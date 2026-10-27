"""Frozen HSFormer + bounded real-triplet fusion, with geographic holdouts."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from cloud_severity.engine import load_model
from temporal_prior.train import PARENT_SHA,atomic_save
from temporal_prior.real_review import panel
from .data import validate,load,digest
from .model import BoundedFusion


def metrics(pred,b):
    valid=b['eval_valid']; error=(pred-b['label']).abs()
    def region(mask):
        n=float(mask.sum())*3
        return float((error*mask).sum()/n) if n else None
    mse=((pred-b['label']).square()*valid).sum()/(3*valid.sum()).clamp_min(1)
    changed=((pred<=1e-6).all(1,keepdim=True)&(b['cloudy']>1e-6).any(1,keepdim=True))*valid.bool()
    return {'proxy_psnr':float(-10*torch.log10(mse.clamp_min(1e-12))),
            'proxy_cloud_mae':region(b['mask']*valid),
            'proxy_high_cloud_mae':region(b['high_cloud']*valid),
            'new_black_fraction':float(changed.sum()/valid.sum().clamp_min(1))}


def run(data,parent,output,epochs=12,hours=2):
    data,parent,output=Path(data),Path(parent),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    protocol=validate(data)
    if digest(parent)!=PARENT_SHA: raise ValueError('Incorrect parent: use completed Sentinel fine-tuning (4) HSFormer')
    if not torch.cuda.is_available(): raise RuntimeError('Enable a T4 GPU after data ready=true')
    torch.manual_seed(321);torch.set_num_threads(4);device='cuda';start=time.monotonic()
    code=b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))
    cfg={'version':1,'model':'bounded_fusion_v1','parent_sha256':PARENT_SHA,'manifest_sha256':digest(data/'manifest.json'),
         'code_sha256':hashlib.sha256(code).hexdigest(),'epochs':epochs,'patience':4,'lr':.0002,'batch':1,'accumulation':4,'seed':321}
    cfgpath=output/'config.json'
    if cfgpath.exists() and json.loads(cfgpath.read_text())!=cfg: raise ValueError('Different experiment; use a fresh run folder')
    cfgpath.write_text(json.dumps(cfg,indent=2))
    status={'state':'caching','website_ready':False,'same_date_accuracy_validated':False,'config':cfg}
    def save_status(state):
        status.update(state=state,elapsed_seconds=time.monotonic()-start)
        (output/'status.json').write_text(json.dumps(status,indent=2));return status
    save_status('caching')
    cache=output/'cache';cache.mkdir(exist_ok=True)
    net=BoundedFusion().to(device)
    optim=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    samples={s:[] for s in ('train','validation','test')}
    # Cache all observed inputs without using test labels for optimization/selection.
    backbone,_=load_model(parent,device)
    for p in backbone.parameters():p.requires_grad_(False)
    for r in protocol['records']:
        p=cache/(r['id']+'.pt');samples[r['split']].append((r,p))
        if not p.exists():
            b=load(data,r)
            with torch.no_grad():b['base']=backbone(b['cloudy'][None].to(device))['restored'][0].cpu()*b['valid']
            atomic_save(b,p)
        if time.monotonic()-start>hours*3600-180:return save_status('paused')
    del backbone;torch.cuda.empty_cache()
    def get(p):return {k:v[None].to(device) for k,v in torch.load(p,weights_only=True).items()}
    def forward(b):return net(b['cloudy'],b['base'],b['reference'],b['mask'],b['days'],b['valid'])
    def assess(split,export=False):
        net.eval();rows=[]
        with torch.no_grad():
            for i,(r,p) in enumerate(samples[split]):
                b=get(p);pred=forward(b)
                if not torch.isfinite(pred).all():raise ValueError('Nonfinite prediction; do not use this checkpoint')
                outputs={'cloudy':b['cloudy'],'hsformer':b['base'],
                         'reference_copy':torch.where(b['mask'].bool(),b['reference'],b['cloudy']),
                         'bounded_fusion':pred}
                for name,value in outputs.items():
                    rows.append({'id':r['id'],'group':r['group'],'method':name,
                                 **{k:r[k] for k in ('cloudy_date','reference_date','label_date')},**metrics(value,b)})
                if export and i<8:
                    arrays=[a[0].cpu().numpy() for a in [b['cloudy'],b['reference'],b['base'],pred,b['label']]]
                    panel(arrays,['Real cloudy input','Earlier reference','Single HSFormer','Bounded learned fusion','Different-date proxy label'],output/(split+'_'+r['id']+'.png'))
        groups=sorted({r['group'] for r in rows});keys=('proxy_psnr','proxy_cloud_mae','proxy_high_cloud_mae','new_black_fraction')
        summary={};group_metrics=[]
        for name in outputs:
            summary[name]={}
            for key in keys:
                means=[]
                for g in groups:
                    vals=[r[key] for r in rows if r['method']==name and r['group']==g and r[key] is not None]
                    if vals:means.append(float(np.mean(vals)))
                summary[name][key]=float(np.mean(means)) if means else None
            for g in groups:
                group_metrics.append({'group':g,'method':name,**{key:float(np.mean(v)) if (v:=[r[key] for r in rows if r['method']==name and r['group']==g and r[key] is not None]) else None for key in keys}})
        if export:
            (output/(split+'_per_sample.json')).write_text(json.dumps(rows,indent=2))
            (output/(split+'_per_location.json')).write_text(json.dumps(group_metrics,indent=2))
        return summary
    first=0;best=float('inf');stale=0;history=[]
    if (output/'last.pt').exists():
        old=torch.load(output/'last.pt',map_location=device,weights_only=True)
        net.load_state_dict(old['model']);optim.load_state_dict(old['optimizer'])
        first,best,stale,history=old['epoch'],old['best'],old['stale'],old['history']
    for epoch in range(first,epochs):
        if stale>=cfg['patience']:break
        net.train();optim.zero_grad(set_to_none=True)
        rng=np.random.default_rng(321+epoch);order=rng.permutation(len(samples['train']))
        for i,index in enumerate(order):
            b=get(samples['train'][index][1]);rotation=int(rng.integers(4));flip=bool(rng.integers(2))
            for k,v in b.items():
                if v.ndim==4:b[k]=torch.rot90(v,rotation,(-2,-1)).flip(-1) if flip else torch.rot90(v,rotation,(-2,-1))
            pred=forward(b);supervised=b['mask']*b['loss_valid']
            loss=((pred-b['label']).abs()*supervised).sum()/(3*supervised.sum()).clamp_min(1)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss; training stopped before checkpoint update')
            (loss/min(4,len(order)-(i//4)*4)).backward()
            if (i+1)%4==0 or i+1==len(order):
                torch.nn.utils.clip_grad_norm_(net.parameters(),1);optim.step();optim.zero_grad(set_to_none=True)
            if time.monotonic()-start>hours*3600-180:return save_status('paused')
        result=assess('validation');score=result['bounded_fusion']['proxy_cloud_mae']
        history.append({'epoch':epoch+1,'validation':result})
        if score<best:
            best=score;stale=0
            atomic_save({'model':net.state_dict(),'epoch':epoch+1,'config':cfg},output/'best.pt')
        else:stale+=1
        atomic_save({'model':net.state_dict(),'optimizer':optim.state_dict(),'epoch':epoch+1,'best':best,
                     'stale':stale,'history':history},output/'last.pt')
        print('Epoch',epoch+1,'validation proxy cloud MAE:',score,flush=True);save_status('training')
    chosen=torch.load(output/'best.pt',map_location=device,weights_only=True)
    net.load_state_dict(chosen['model'])
    validation=assess('validation',True)
    # Test is evaluated only after the validation-selected checkpoint is fixed.
    test=assess('test',True)
    (output/'history.json').write_text(json.dumps(history,indent=2))
    status.update(selected_epoch=chosen['epoch'],validation=validation,test=test,
                  test_beats_both_proxy_baselines=all(test['bounded_fusion']['proxy_cloud_mae']<test[n]['proxy_cloud_mae'] for n in ('hsformer','reference_copy')),
                  notice='Small location-disjoint API dataset with different-date proxy labels. Bounds prevent unbounded subtraction, not hallucinations. No automatic web deployment.')
    return save_status('experiment_complete')
