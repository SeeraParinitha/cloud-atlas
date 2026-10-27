import copy,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .engine import train,evaluate,save_json,checked_manifest
from .audit import paired_bootstrap

CORE=['full','mask_blend','severity_noblend','mask_noblend','resunet','unet','attention_unet','nafnet_small']
EXTENDED=['aux_no_guidance','product_gate','unbounded_full','no_preserve','no_lpips']

def code_hash():
    root=Path(__file__).parent;h=hashlib.sha256()
    for p in sorted(root.glob('*.py')):h.update(p.name.encode()+p.read_bytes())
    external=root.parent/'third_party'/'hsformer_runtime'
    for p in sorted(external.glob('*.py')):h.update(p.name.encode()+p.read_bytes())
    return h.hexdigest()

def run_config(base,variant,seed,root):
    c=copy.deepcopy(base);c['seed']=seed;c['output']=str(Path(root)/variant/f'seed_{seed}')
    c['model']['variant']='full' if variant in ['no_preserve','no_lpips'] else variant
    if variant=='no_preserve':c['loss']['preserve']=0
    if variant=='no_lpips':c['loss']['lpips']=0
    c['experiment_name']=variant
    return c

def check_config(current,old):
    for k in ['seed','epochs','size','batch_size','model','loss','pseudo','severity_bins','learning_rate','weight_decay','code_sha256','data_protocol','experiment_name']:
        if current.get(k)!=old.get(k):raise ValueError(f'Existing run differs in {k}; start a new experiment name.')

def run_queue(base,variants,seeds,root,max_runs=1):
    root=Path(root);started=0;rows=[];manifest_hash=checked_manifest(base['manifest'])
    for variant in variants:
        for seed in seeds:
            c=run_config(base,variant,seed,root);out=Path(c['output']);last=out/'last.pt';best=out/'best.pt'
            completed=False;state=None
            if last.exists():
                state=torch.load(last,map_location='cpu',weights_only=True)
                check_config(c,state['config'])
                if state['manifest_sha256']!=manifest_hash:raise ValueError('Dataset content or split changed')
                if not best.exists():raise FileNotFoundError(f'Restore best.pt as well as last.pt: {out}')
                completed=state['epoch']+1>=c['epochs']
            if not completed and started<max_runs:
                print(f'Running {variant}, seed {seed}',flush=True)
                train(c,resume=str(last) if last.exists() else None);started+=1
                state=torch.load(last,map_location='cpu',weights_only=True);completed=state['epoch']+1>=c['epochs']
            rows.append(dict(variant=variant,seed=seed,epochs=state['epoch']+1 if state else 0,target=c['epochs'],status='complete' if completed else 'partial' if state else 'pending'))
    table=pd.DataFrame(rows);table.to_csv(root/'run_status.csv',index=False)
    return table

def evaluate_queue(base,variants,seeds,root):
    root=Path(root);reports={};rows=[]
    # Preflight entire declared plan before opening the test split.
    for v in variants:
        for seed in seeds:
            c=run_config(base,v,seed,root);out=Path(c['output'])
            if not (out/'last.pt').exists():raise RuntimeError('Finish every declared run before final test evaluation')
            st=torch.load(out/'last.pt',map_location='cpu',weights_only=True);check_config(c,st['config'])
            if st['epoch']+1<c['epochs']:raise RuntimeError('Incomplete run: '+str(out))
    for v in variants:
        for seed in seeds:
            c=run_config(base,v,seed,root);out=Path(c['output']);report=evaluate(out/'best.pt',base['manifest'],out/'test',device=base['device'],with_lpips=True)
            reports[v,seed]=report
            for subset,values in [('ALL',report['overall'])]+list(report['by_dataset'].items()):
                row=dict(variant=v,seed=seed,subset=subset,parameters=report['parameters'])
                row.update({k:values.get(k) for k in ['psnr','ssim','rmse','lpips','psnr_gain','cloud_mae','clear_mae','clear_change_mae','stable_clear_change_mae','identity_change_mae','edge_mae','inference_ms']})
                rows.append(row)
    per_seed=pd.DataFrame(rows);per_seed.to_csv(root/'results_per_seed.csv',index=False)
    metrics=[c for c in per_seed.columns if c not in ['variant','seed','subset','parameters']]
    per_seed.groupby(['variant','subset'])[metrics].agg(['mean','std','count']).to_csv(root/'results_mean_sd.csv')
    contrasts=[]
    for reference in ['mask_blend','severity_noblend','resunet','nafnet_small','unbounded_full','product_gate','no_preserve','no_lpips']:
        if 'full' not in variants or reference not in variants:continue
        def averaged(v):
            x=pd.concat([pd.DataFrame(reports[v,s]['samples']) for s in seeds])
            return x.groupby(['id','group','dataset'],as_index=False)[['psnr','lpips','cloud_mae','stable_clear_change_mae']].mean()
        a,b=averaged('full'),averaged(reference)
        for subset in ['ALL']+sorted(a.dataset.unique().tolist()):
            aa=a if subset=='ALL' else a[a.dataset==subset];bb=b if subset=='ALL' else b[b.dataset==subset]
            for metric in ['psnr','lpips','cloud_mae','stable_clear_change_mae']:
                contrasts.append(dict(comparison='full minus '+reference,subset=subset,metric=metric,**paired_bootstrap(aa,bb,metric)))
    pd.DataFrame(contrasts).to_csv(root/'paired_bootstrap.csv',index=False)
    # Difference-in-differences for the guidance x blending factorial, evaluated per seed.
    factorial=[]
    names=['full','mask_blend','severity_noblend','mask_noblend']
    if all(v in variants for v in names):
        for seed in seeds:
            for metric in ['psnr','lpips','cloud_mae','stable_clear_change_mae']:
                vals={v:reports[v,seed]['overall'].get(metric) for v in names}
                if all(isinstance(x,(int,float)) for x in vals.values()):
                    factorial.append(dict(seed=seed,metric=metric,interaction=(vals['full']-vals['mask_blend'])-(vals['severity_noblend']-vals['mask_noblend'])))
    pd.DataFrame(factorial).to_csv(root/'factorial_interactions.csv',index=False)
    return per_seed,reports