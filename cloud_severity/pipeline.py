"""Single invocation: download -> train -> validate/select -> test -> export."""
import copy
import json
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
import pandas as pd
import torch
from .cli import calibrate
from .data import pair_records,split_records,write_manifest
from .discovery import discover_rice
from .rice_masks import prepare_rice_masks
from .engine import train,evaluate,checked_manifest,save_json
from .experiments import code_hash
from .selection import select_model,export_model


def archive_project(root=Path('.')):
    target=root/'cloud_shadow_complete.zip'
    temp=root/'cloud_shadow_complete.tmp'
    folders=['cloud_severity','web','configs','scripts','docs','tests','runs','deployment','data/binary_masks','third_party/hsformer_runtime']
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in folders:
            for p in (root/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.tmp': z.write(p,str(p.relative_to(root)))
        for name in ['requirements.txt','requirements-web.txt','README.md','Dockerfile','.dockerignore','.env.example','data/rice.json','data/calibration.json','data/archive_provenance.json','data/sources.json']:
            p=root/name
            if p.is_file(): z.write(p,name)
    temp.replace(target)
    return str(target.resolve())


def run(config_path='configs/run_all.json'):
    plan=json.loads(Path(config_path).read_text()); started=time.monotonic()
    summary={'state':'running','stages':{},'plan':plan}
    Path('runs').mkdir(exist_ok=True)
    try:
        subprocess.check_call([sys.executable,'scripts/prepare_hsformer.py'])
        print('Preparing verified RICE pairs and frozen calibration',flush=True)
        torch.set_num_threads(4)
        if not Path('data/RICE_DATASET').is_dir():
            subprocess.check_call([sys.executable,'-m','cloud_severity.download','--output','data'])
        if not Path('data/rice.json').exists():
            sources=discover_rice('data/RICE_DATASET'); records=[]
            for s in sources: records.extend(pair_records(s['cloudy'],s['clear'],s['name'],s.get('masks')))
            prepare_rice_masks(records,'data/binary_masks'); split_records(records,42)
            write_manifest('data/rice.json',records,42)
        base=json.loads(Path('configs/research.json').read_text())
        manifest_hash=checked_manifest(base['manifest'])
        if not Path('data/calibration.json').exists():
            cal=calibrate(base['manifest'],base['size']); cal.update(manifest_sha256=manifest_hash,size=base['size'])
            save_json('data/calibration.json',cal)
        cal=json.loads(Path('data/calibration.json').read_text())
        if cal['manifest_sha256']!=manifest_hash or cal['size']!=base['size']: raise ValueError('Frozen calibration differs from dataset')
        base.update({k:cal[k] for k in ['pseudo','severity_bins']}); base['code_sha256']=code_hash()
        base['epochs']=plan['epochs']; base['seed']=123
        variants=['full','hsformer']+(['mask_blend'] if plan.get('include_ablation') else [])
        summary['stages']['data']='complete'
        # Download LPIPS weights before spending hours training. Failure stops early.
        from .losses import Perceptual
        perceptual=Perceptual(); del perceptual
        for variant in variants:
            print('Training/resuming '+variant,flush=True)
            cfg=copy.deepcopy(base); cfg['model']['variant']=variant; cfg['output']=f'runs/{variant}'
            if variant=='hsformer':
                cfg['batch_size']=1; cfg['accumulation_steps']=4
                cfg['hsformer_provenance']=json.loads(Path('third_party/hsformer_runtime/provenance.json').read_text())
            else: cfg['accumulation_steps']=1
            remaining=plan['session_hours']*3600-(time.monotonic()-started)
            if remaining<1800:
                summary['state']='paused_session_budget'; return summary
            cfg['session_seconds']=remaining-1800
            last=Path(cfg['output'])/'last.pt'
            done=False
            if last.exists():
                state=torch.load(last,map_location='cpu',weights_only=True)
                if state['manifest_sha256']!=manifest_hash: raise ValueError('Saved dataset changed')
                for key in ['seed','epochs','model','loss','pseudo','size','batch_size','accumulation_steps','code_sha256','learning_rate','weight_decay']:
                    if state['config'].get(key)!=cfg.get(key): raise ValueError('Saved configuration changed: '+key)
                done=state['epoch']+1>=cfg['epochs']
            if not done:
                completed=train(cfg,resume=str(last) if last.exists() else None)
                if completed is None:
                    summary['state']='paused_session_budget'; summary['stages'][variant]='partial'; return summary
            summary['stages'][variant]='trained'
        reports={}
        for variant in variants:
            reports[variant]=evaluate(f'runs/{variant}/best.pt',base['manifest'],f'runs/{variant}/validation',split='validation')
        winner,ranking=select_model({v:reports[v] for v in ['full','hsformer']})
        decision={'winner':winner,'rule':'Lowest mean validation rank across PSNR, SSIM, LPIPS and stable-clear change MAE; ties: PSNR, then latency. Checkpoints individually selected by validation PSNR.',
                  'ranking':ranking,'single_seed':123,'test_used_for_selection':False}
        save_json('runs/selection.json',decision); pd.DataFrame(ranking).to_csv('runs/validation_comparison.csv',index=False)
        export_model(f'runs/{winner}/best.pt',decision)
        for variant in variants:
            evaluate(f'runs/{variant}/best.pt',base['manifest'],f'runs/{variant}/test')
        from .reporting import plots
        plots(variants)
        summary['winner']=winner; summary['stages']['rice_test']='complete'
        whu=plan.get('whus2_manifest'); protocol=plan.get('whus2_protocol')
        if whu and protocol:
            for variant in variants:
                subprocess.check_call([sys.executable,'-m','cloud_severity.cli','evaluate','--checkpoint',f'runs/{variant}/best.pt','--manifest',whu,'--output',f'runs/{variant}/whus2cr','--split','test','--external-protocol',protocol])
            summary['stages']['whus2cr']='complete'
        else: summary['stages']['whus2cr']='not run: aligned official test manifest/protocol not supplied'
        summary['state']='training_and_rice_evaluation_complete'
        return summary
    except Exception as exc:
        summary.update(state='failed',error=str(exc)); traceback.print_exc()
        return summary
    finally:
        summary['elapsed_seconds']=time.monotonic()-started
        save_json('runs/pipeline_status.json',summary)
        print('Saved project archive:',archive_project())
        print(json.dumps(summary,indent=2))

if __name__=='__main__': run()
