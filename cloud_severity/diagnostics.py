"""Reference-conditioned synthetic opacity probe, not validation of real optical thickness."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.stats import spearmanr
from .data import PairedDataset,PseudoConfig
from .engine import load_model

def opacity_probe(checkpoint,manifest,output,device='cpu',max_images=24):
    net,state=load_model(checkpoint,torch.device(device));cfg=state['config']
    ds=PairedDataset(manifest,'test',cfg['size'],False,PseudoConfig(**cfg['pseudo']))
    rows=[];generator=torch.Generator().manual_seed(2026)
    with torch.no_grad():
        for i in range(min(max_images,len(ds))):
            item=ds[i];clear=item['clear'][None].to(device);h,w=clear.shape[-2:]
            # Fixed smooth spatial support shared over opacity levels for this image.
            field=F.interpolate(torch.rand(1,1,4,4,generator=generator),size=(h,w),mode='bilinear',align_corners=False).to(device)
            support=field>.5
            if not support.any():support[:]=True
            for alpha in [0.,.15,.3,.5,.7]:
                opacity=support*alpha;cloudy=(1-opacity)*clear+opacity
                out=net(cloudy)
                rows.append(dict(id=item['id'],alpha=alpha,predicted_severity=float(out['severity'][support].mean()),
                    restoration_mae=float((out['restored']-clear).abs().mean())))
    table=pd.DataFrame(rows);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    table.to_csv(output/'synthetic_opacity_probe.csv',index=False)
    # Per-image ranks avoid confounding opacity with varying land appearance across images.
    correlations=[float(spearmanr(g.alpha,g.predicted_severity).statistic) for _,g in table.groupby('id')]
    pd.DataFrame([dict(mean_within_image_spearman=float(np.nanmean(correlations)),images=len(correlations),
                      interpretation='Synthetic white-cloud mixing probe only; not a real opacity calibration')]).to_csv(output/'probe_summary.csv',index=False)
    return table