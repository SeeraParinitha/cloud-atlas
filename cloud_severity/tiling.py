"""Overlap-tile inference, shared by the website and command-line deployment."""
import torch
from torch.nn import functional as F


def starts(length, tile, stride):
    result=list(range(0,max(length-tile,0)+1,stride))
    if result[-1]!=max(0,length-tile): result.append(max(0,length-tile))
    return result


@torch.no_grad()
def restore_tiled(model, image, device='cpu', tile=256, overlap=64):
    if image.ndim!=3 or image.shape[0]!=3 or min(image.shape[1:])<1:
        raise ValueError('Expected nonempty CHW RGB tensor')
    if not 0<=overlap<tile or not torch.isfinite(image).all(): raise ValueError('Invalid tiling or pixels')
    h,w=image.shape[-2:]
    x=F.pad(image[None],(0,max(0,tile-w),0,max(0,tile-h)),mode='replicate')[0]
    hh,ww=x.shape[-2:]; weight=torch.hann_window(tile,periodic=False).clamp_min(.05)
    weight=(weight[:,None]*weight[None,:])[None]
    total={}; denom=torch.zeros(1,hh,ww)
    model.eval()
    for y in starts(hh,tile,tile-overlap):
        for xx in starts(ww,tile,tile-overlap):
            prediction=model(x[:,y:y+tile,xx:xx+tile][None].to(device))
            for key in ['restored','mask','severity']:
                if key not in prediction: continue
                value=prediction[key][0].detach().float().cpu()
                if not torch.isfinite(value).all(): raise RuntimeError('Model returned non-finite pixels')
                if key not in total: total[key]=torch.zeros(value.shape[0],hh,ww)
                total[key][:,y:y+tile,xx:xx+tile]+=value*weight
            denom[:,y:y+tile,xx:xx+tile]+=weight
    return {k:(v/denom)[:,:h,:w].clamp(0,1) for k,v in total.items()}
