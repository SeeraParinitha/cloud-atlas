"""Shared four-scale encoder, cloud head, gated skips, and residual blending."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

# All four factorial variants retain identical heads, losses, channel widths and parameter counts.
VARIANTS = {
    "hsformer": dict(),
    "nafnet_small": dict(),
    "unet": dict(),
    "attention_unet": dict(attention="standard"),
    "resunet": dict(residual=True, blocks=True),
    "aux_no_guidance": dict(mask=True,severity=True,residual=True),
    "mask_noblend": dict(mask=True,severity=True,attention="mask",residual=True),
    "severity_noblend": dict(mask=True,severity=True,attention="severity",residual=True),
    "mask_blend": dict(mask=True,severity=True,attention="mask",residual=True,blend=True),
    "full": dict(mask=True,severity=True,attention="severity",residual=True,blend=True),
    "product_gate": dict(mask=True,severity=True,attention="severity",residual=True,blend=True,product=True),
    "unbounded_full": dict(mask=True,severity=True,attention="severity",residual=True,blend=True,unbounded=True),
}


@dataclass
class ModelConfig:
    variant: str = "full"
    base_channels: int = 32


class Block(nn.Module):
    def __init__(self, cin, cout, residual=False):
        super().__init__()
        groups = next(g for g in (8, 4, 2, 1) if cout % g == 0)
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
        )
        self.shortcut = nn.Conv2d(cin, cout, 1) if residual and cin != cout else nn.Identity()
        self.residual = residual

    def forward(self, x):
        y = self.body(x)
        return F.silu(y + self.shortcut(x) if self.residual else y)


def resize(x, like):
    return F.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=False)


class SeverityAttention(nn.Module):
    def __init__(self, channels, maps):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels + maps, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, feature, maps):
        return feature * self.gate(torch.cat([feature] + [resize(m, feature) for m in maps], 1))


class AttentionGate(nn.Module):
    """Conventional additive attention conditioned on the decoder's gating signal."""

    def __init__(self, skip_channels, gate_channels):
        super().__init__()
        hidden = max(skip_channels // 2, 1)
        self.skip = nn.Conv2d(skip_channels, hidden, 1)
        self.context = nn.Conv2d(gate_channels, hidden, 1)
        self.score = nn.Conv2d(hidden, 1, 1)

    def forward(self, skip, context):
        weight = torch.sigmoid(
            self.score(F.relu(self.skip(skip) + resize(self.context(context), skip)))
        )
        return skip * weight


class CloudRemovalNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        if self.config.variant not in VARIANTS or self.config.base_channels < 2:
            raise ValueError("Unknown variant or base_channels < 2")
        self.flags = VARIANTS[self.config.variant]
        if self.config.variant == "hsformer":
            from .hsformer import HSFormerAdapter
            self.hsformer = HSFormerAdapter()
            return
        if self.config.variant=="nafnet_small":
            from .nafnet import NAFNetSmall
            self.naf=NAFNetSmall(self.config.base_channels)
            return
        c = [self.config.base_channels * 2**i for i in range(4)]
        # Standard U-Net baselines use double convolutions; proposed variants use residual CNN blocks.
        residual_blocks = self.flags.get("mask", False) or self.flags.get("blocks", False)
        self.encoder = nn.ModuleList(
            [Block(3 if i == 0 else c[i - 1], c[i], residual_blocks) for i in range(4)]
        )
        if self.flags.get("mask"):
            self.cloud_fusion = nn.Sequential(nn.Conv2d(sum(c), c[0], 1), nn.SiLU())
            self.mask_head = nn.Conv2d(c[0], 1, 1)
            if self.flags.get("severity"):
                self.severity_head = nn.Conv2d(c[0], 1, 1)
        mode = self.flags.get("attention")
        self.guidance = (
            nn.ModuleList([SeverityAttention(v, 2) for v in c])
            if mode in ("severity", "mask")
            else nn.ModuleList()
        )
        self.gates = (
            nn.ModuleList([AttentionGate(c[i], c[i + 1]) for i in (2, 1, 0)])
            if mode == "standard"
            else nn.ModuleList()
        )
        self.decoder = nn.ModuleList(
            [Block(c[i] + c[i + 1], c[i], residual_blocks) for i in (2, 1, 0)]
        )
        self.output = nn.Conv2d(c[0], 3, 1)

    def forward(self, cloudy):
        if self.config.variant == "hsformer":
            return self.hsformer(cloudy)
        if cloudy.ndim != 4 or cloudy.shape[1] != 3 or min(cloudy.shape[-2:]) < 16:
            raise ValueError("Expected BCHW RGB input at least 16 x 16")
        if self.config.variant=="nafnet_small": return self.naf(cloudy)
        features, x = [], cloudy
        for i, block in enumerate(self.encoder):
            x = block(F.avg_pool2d(x, 2) if i else x)
            features.append(x)
        result = {}
        if self.flags.get("mask"):
            cloud = self.cloud_fusion(torch.cat([resize(f, features[0]) for f in features], 1))
            result["mask_logits"] = self.mask_head(cloud)
            result["mask"] = torch.sigmoid(result["mask_logits"])
            if self.flags.get("severity"):
                result["severity"] = torch.sigmoid(self.severity_head(cloud))
        if self.guidance:
            # Mask controls still train their severity head, but zero its input to attention.
            maps = [result["mask"], result["severity"] if self.flags["attention"]=="severity"
                    else torch.zeros_like(result["severity"])]
            features = [gate(f, maps) for gate, f in zip(self.guidance, features)]
        x = features[-1]
        for k, (block, skip) in enumerate(zip(self.decoder, reversed(features[:-1]))):
            if self.gates:
                skip = self.gates[k](skip, x)
            x = block(torch.cat([resize(x, skip), skip], 1))
        raw = self.output(x)
        delta = raw if self.flags.get("unbounded") else torch.tanh(raw)
        restored = cloudy + delta if self.flags.get("residual") else torch.sigmoid(raw)
        result["residual"] = delta if self.flags.get("residual") else restored-cloudy
        result["reconstruction"] = restored
        gate = result["mask"] if self.flags.get("blend") else torch.ones_like(cloudy[:,:1])
        if self.flags.get("product"):
            gate = gate * result["severity"]
        result["confidence"] = gate  # Historical key; this is an edit gate, NOT calibrated uncertainty.
        result["preclip"] = cloudy + gate * (restored-cloudy)
        result["restored"] = result["preclip"].clamp(0,1)
        return result

    def config_dict(self):
        return asdict(self.config)
