"""Experimental dense mixture. Gates are mixing coefficients, not confidence."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import block


class SoftFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=block(14,24)
        self.context=block(24,48)
        self.decoder=block(72,24)
        self.head=nn.Conv2d(24,7,1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():self.head.bias.copy_(torch.tensor([3.,-1.,-1.,-1.,0.,0.,0.]))

    def forward(self,cloudy,base,reference,mask,reference_valid,target_valid,probability,days):
        age=(days.reshape(-1,1,1,1)/60).clamp(0,1).expand_as(mask)
        inputs=torch.cat([cloudy*target_valid,base*target_valid,reference*reference_valid,
                          mask,reference_valid,target_valid,probability,age],1)
        f=self.encoder(inputs)
        low=F.interpolate(self.context(F.avg_pool2d(f,2)),size=f.shape[-2:],mode='bilinear',align_corners=False)
        head=self.head(self.decoder(torch.cat([f,low],1)))
        # Smooth learned logits, not the observed image. The provider mask is an input,
        # never a hard rule that forbids edits in missed haze regions.
        logits=F.avg_pool2d(F.pad(head[:,:4],(2,2,2,2),mode='replicate'),5,stride=1)
        logits=torch.cat([logits[:,:2],logits[:,2:3].masked_fill(~reference_valid.bool(),-1e4),logits[:,3:4]],1)
        weights=logits.softmax(1)
        candidate=head[:,4:].sigmoid()
        estimate=weights[:,0:1]*cloudy+weights[:,1:2]*base+weights[:,2:3]*reference+weights[:,3:4]*candidate
        return torch.where(target_valid.bool(),estimate,cloudy),weights
