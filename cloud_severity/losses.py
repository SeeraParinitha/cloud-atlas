from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def ssim(x, y):
    """Per-image RGB SSIM: Gaussian 11x11/sigma 1.5, valid windows, data_range=1."""
    size = min(11, x.shape[-2], x.shape[-1])
    size -= 1 - size % 2
    grid = torch.arange(size, device=x.device, dtype=x.dtype) - (size - 1) / 2
    kernel = torch.exp(-grid.square() / (2 * 1.5**2))
    kernel /= kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :]).expand(x.shape[1], 1, size, size)

    def conv(z):
        return F.conv2d(z, kernel, groups=x.shape[1])

    mx, my = conv(x), conv(y)
    vx, vy = (conv(x * x) - mx * mx), (conv(y * y) - my * my)
    covariance = conv(x * y) - mx * my
    score = ((2 * mx * my + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2)
    )
    return score.mean((1, 2, 3))


class Perceptual(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                'Install "cloud-severity[perceptual]" for LPIPS, or explicitly set its weight to 0 for smoke tests.'
            ) from exc
        # Pretrained weights are essential; never silently replace LPIPS with random features.
        self.metric = lpips.LPIPS(net="alex").eval()
        self.metric.requires_grad_(False)

    def forward(self, x, y):
        return self.metric(x.clamp(0, 1) * 2 - 1, y.clamp(0, 1) * 2 - 1).flatten(1).mean(1)


@dataclass
class LossConfig:
    ssim: float = 0.2
    lpips: float = 0.05
    preserve: float = 0.1
    mask: float = 0.1
    severity: float = 0.1
    pseudo_mask_weight: float = 0.25


class HybridLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or LossConfig()
        if any(v < 0 for v in vars(self.config).values()):
            raise ValueError("Loss weights must be nonnegative")
        self.perceptual = Perceptual() if self.config.lpips else None

    def forward(self, out, batch):
        x, target = out["restored"], batch["clear"]
        terms = {
            "l1": F.l1_loss(x, target),
            "ssim": (1 - ssim(x, target)).mean(),
            "preserve": region_mean((x-batch["cloudy"]).abs(), batch["stable_clear"]),
        }
        if self.perceptual is not None:
            terms["lpips"] = self.perceptual(x, target).mean()
        if "mask_logits" in out:
            per_image = F.binary_cross_entropy_with_logits(out["mask_logits"],batch["mask"],reduction="none").mean((1,2,3))
            weights = x.new_tensor([1.0 if src=="dataset" else self.config.pseudo_mask_weight for src in batch["mask_source"]])
            terms["mask"] = (per_image*weights).mean()
        if "severity" in out:
            err=(out["severity"]-batch["severity"]).abs()
            terms["severity"] = 0.5*(region_mean(err,batch["mask"])+region_mean(err,1-batch["mask"]))
        total = terms["l1"] + sum(
            getattr(self.config, name) * value for name, value in terms.items() if name != "l1"
        )
        return total, terms


def region_mean(error,mask):
    """Per-image region-normalized mean; empty regions contribute zero loss."""
    denom=mask.sum((1,2,3))*error.shape[1]
    values=(error*mask).sum((1,2,3))/denom.clamp_min(1)
    return values.mean()