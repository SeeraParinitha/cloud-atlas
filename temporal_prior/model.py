import torch
from torch import nn
from torch.nn import functional as F


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU(),
                         nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU())


class ReferenceFusion(nn.Module):
    """Small learned fusion head on frozen HSFormer output, not upstream HSFormer.

    Inputs are target RGB, frozen restoration, independent reference RGB, a
    target cloud mask, and elapsed days. No reference pixels are directly pasted
    into the output. Observed unmasked pixels are preserved exactly.
    """
    def __init__(self):
        super().__init__()
        self.target = block(7, 24)
        self.reference = block(4, 24)
        self.context = block(48, 48)
        self.decode = block(96, 24)
        self.output = nn.Conv2d(24, 3, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, target, base, reference, mask, days):
        gap = (days.reshape(-1, 1, 1, 1) / 93).clamp(0, 1).expand_as(mask)
        a = self.target(torch.cat((target, base, mask), 1))
        b = self.reference(torch.cat((reference, gap), 1))
        fused = torch.cat((a, b), 1)
        context = F.interpolate(self.context(F.avg_pool2d(fused, 2)),
                                size=target.shape[-2:], mode='bilinear', align_corners=False)
        estimate = (base + self.output(self.decode(torch.cat((fused, context), 1)))).clamp(0, 1)
        return torch.where(mask.bool(), estimate, target)
