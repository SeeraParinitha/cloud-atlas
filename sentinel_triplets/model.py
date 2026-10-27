import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU())


class BoundedFusion(nn.Module):
    """Learned pixelwise mixture, not the failed unbounded residual subtraction.

    The third candidate is a learned RGB reconstruction. The network predicts
    softmax weights for HSFormer, reference and that reconstruction. This can
    learn to rely heavily on the reference; beating reference copy must be tested.
    All masked predictions are convex combinations of values in [0,1].
    Bounds guarantee numeric range only, not correct hidden surface recovery.
    """
    def __init__(self):
        super().__init__()
        self.target = block(8, 24)  # cloudy RGB, HS RGB, cloud mask, input validity
        self.reference = block(4, 24)  # RGB and reference age
        self.context = block(48, 48)
        self.decode = block(96, 24)
        self.head = nn.Conv2d(24, 6, 1)

    def forward(self, cloudy, base, reference, mask, days, valid):
        age = (days.reshape(-1, 1, 1, 1)/60).clamp(0, 1).expand_as(mask)
        a = self.target(torch.cat((cloudy, base, mask, valid), 1))
        b = self.reference(torch.cat((reference, age), 1))
        f = torch.cat((a, b), 1)
        low = F.interpolate(self.context(F.avg_pool2d(f, 2)), size=f.shape[-2:], mode='bilinear', align_corners=False)
        h = self.head(self.decode(torch.cat((f, low), 1)))
        weights = h[:, :3].softmax(1)
        candidate = h[:, 3:].sigmoid()
        mixed = weights[:, 0:1]*base + weights[:, 1:2]*reference + weights[:, 2:3]*candidate
        return torch.where(mask.bool(), mixed, cloudy)*valid
