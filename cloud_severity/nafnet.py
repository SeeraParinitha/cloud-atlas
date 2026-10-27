"""Compact NAFNet adaptation from Chen et al. (ECCV 2022), arXiv:2204.04676.
Reference: https://github.com/megvii-research/NAFNet (MIT).
This is an independently written small adaptation, not the paper's trained model.
"""
import torch
from torch import nn
from torch.nn import functional as F

class LayerNorm2d(nn.Module):
    def __init__(self,c):
        super().__init__();self.weight=nn.Parameter(torch.ones(1,c,1,1));self.bias=nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x):
        mean=x.mean(1,keepdim=True);var=(x-mean).square().mean(1,keepdim=True)
        return (x-mean)/torch.sqrt(var+1e-6)*self.weight+self.bias

class SimpleGate(nn.Module):
    def forward(self,x):
        a,b=x.chunk(2,dim=1);return a*b

class NAFBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.norm1=LayerNorm2d(c);self.conv1=nn.Conv2d(c,2*c,1);self.dw=nn.Conv2d(2*c,2*c,3,padding=1,groups=2*c)
        self.sg=SimpleGate();self.sca=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,c,1));self.conv2=nn.Conv2d(c,c,1)
        self.norm2=LayerNorm2d(c);self.ffn1=nn.Conv2d(c,2*c,1);self.ffn2=nn.Conv2d(c,c,1)
        self.beta=nn.Parameter(torch.zeros(1,c,1,1));self.gamma=nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x):
        y=self.sg(self.dw(self.conv1(self.norm1(x))));y=self.conv2(y*self.sca(y));z=x+self.beta*y
        return z+self.gamma*self.ffn2(self.sg(self.ffn1(self.norm2(z))))

class NAFNetSmall(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.intro=nn.Conv2d(3,width,3,padding=1);self.ending=nn.Conv2d(width,3,3,padding=1)
        self.enc=nn.ModuleList();self.down=nn.ModuleList();self.up=nn.ModuleList();self.dec=nn.ModuleList();c=width
        for _ in range(4):
            self.enc.append(NAFBlock(c));self.down.append(nn.Conv2d(c,2*c,2,stride=2));c*=2
        self.middle=NAFBlock(c)
        for _ in range(4):
            self.up.append(nn.Sequential(nn.Conv2d(c,2*c,1,bias=False),nn.PixelShuffle(2)));c//=2;self.dec.append(NAFBlock(c))
    def forward(self,x):
        h,w=x.shape[-2:];z=F.pad(x,(0,(-w)%16,0,(-h)%16));z=self.intro(z);skips=[]
        for enc,down in zip(self.enc,self.down):z=enc(z);skips.append(z);z=down(z)
        z=self.middle(z)
        for up,dec,skip in zip(self.up,self.dec,reversed(skips)):z=dec(up(z)+skip)
        residual=self.ending(z)[...,:h,:w];pre=x+residual
        return dict(restored=pre.clamp(0,1),preclip=pre,residual=residual,reconstruction=pre,confidence=torch.ones_like(x[:,:1]))