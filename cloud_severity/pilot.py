"""One forward/backward feasibility check; does not measure restoration accuracy."""
import json
import time
from pathlib import Path
import torch
from cloud_severity.model import CloudRemovalNet, ModelConfig
from cloud_severity.engine import seed_all, sync
from cloud_severity.losses import HybridLoss

if __name__=='__main__':
    seed_all(123); torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=CloudRemovalNet(ModelConfig('hsformer')).to(device)
    x=torch.rand(1,3,256,256,device=device)
    target=(x*.8).clamp(0,1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
    criterion=HybridLoss().to(device)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    sync(device); start=time.perf_counter()
    out=model(x); loss,_=criterion(out,dict(clear=target,cloudy=x,stable_clear=torch.ones_like(x[:,:1])))
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    optimizer.step(); sync(device)
    report=dict(device=str(device),parameters=sum(p.numel() for p in model.parameters()),
                forward_backward_seconds=time.perf_counter()-start,loss=float(loss.detach()),
                output_shape=list(out['restored'].shape),finite_gradients=True,
                peak_cuda_memory_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else None,
                interpretation='Synthetic feasibility only; not research quality or deployment latency')
    Path('runs').mkdir(exist_ok=True)
    Path('runs/hsformer_pilot.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
