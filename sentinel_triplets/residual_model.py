"""Experimental signed correction, zero initialized; no production use."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import block

class ResidualFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=block(14,24);self.context=block(24,48);self.decoder=block(72,24)
        self.head=nn.Conv2d(24,3,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,cloudy,base,reference,mask,reference_valid,target_valid,probability,days):
        age=(days.reshape(-1,1,1,1)/60).clamp(0,1).expand_as(mask)
        anchor=torch.where(reference_valid.bool(),reference,base)
        x=torch.cat([cloudy*target_valid,base*target_valid,reference*reference_valid,mask,reference_valid,target_valid,probability,age],1)
        f=self.encoder(x);low=F.interpolate(self.context(F.avg_pool2d(f,2)),size=f.shape[-2:],mode='bilinear',align_corners=False)
        correction=self.head(self.decoder(torch.cat([f,low],1))).tanh()
        pred=(anchor+correction).clamp(0,1)
        return torch.where(target_valid.bool(),pred,cloudy),correction
