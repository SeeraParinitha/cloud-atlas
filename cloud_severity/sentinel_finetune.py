"""Budgeted Sentinel-2 adaptation of the two saved RICE models, without test access."""
import copy
import json
import math
import shutil
import time
import traceback
import zipfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .engine import image_of, load_model, save_json, seed_all, to_device
from .losses import HybridLoss, LossConfig
from .metrics import average, restoration_metrics, region_metrics
from .selection import export_model
from .sentinel_data import SentinelPairs, calibrate, digest, prepare
from .sentinel_validation import checkpoints
from .tiling import restore_tiled
from .remote_validation import set_download_deadline

DEFAULTS = dict(train_count=512, val_count=64, epochs=8, patience=3, seed=123,
    learning_rate=5e-5, weight_decay=1e-4, accumulation_steps=4,
    session_hours=6.0, reserve_seconds=600, require_cuda=True, prepare_only=False)


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def assess(model, dataset, device, perceptual, output, deadline=float('inf')):
    """Full native patches, exactly the service's overlap tiling; no resize."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(len(dataset)):
        if time.monotonic() >= deadline:
            raise TimeoutError('Evaluation reached the session time budget')
        batch = dataset[i]
        x, target = batch['cloudy'], batch['clear']
        pred = restore_tiled(model, x, device)['restored']
        row = dict(id=batch['id'], **restoration_metrics(pred, target))
        row.update({'input_'+k:v for k,v in restoration_metrics(x,target).items()})
        row.update(region_metrics(pred, target, x, batch['stable_clear'], 'stable_clear_'))
        row['mae'] = float((pred-target).abs().mean())
        row['input_mae'] = float((x-target).abs().mean())
        row['psnr_gain'] = row['psnr']-row['input_psnr']
        if perceptual is not None:
            y = target[None].to(device)
            row['lpips'] = float(perceptual(pred[None].to(device), y).mean())
            row['input_lpips'] = float(perceptual(x[None].to(device), y).mean())
        rows.append(row)
        if i < 4:
            from PIL import Image, ImageDraw
            w, h = x.shape[-1], x.shape[-2]
            panel = Image.new('RGB', (w*3,h+24), 'white')
            draw = ImageDraw.Draw(panel)
            for j,(label,value) in enumerate([('Cloudy',x),('Prediction',pred),('Different-date reference',target)]):
                panel.paste(image_of(value),(j*w,24)); draw.text((j*w+4,4),label,fill='black')
            panel.save(output/f'comparison_{i:02d}.png')
    summary = average(rows)
    summary['psnr_improved_fraction'] = sum(r['psnr_gain']>0 for r in rows)/len(rows)
    pd.DataFrame(rows).to_csv(output/'per_image.csv',index=False)
    save_json(output/'metrics.json',summary)
    return summary


def eligible(metrics, baseline):
    """Prespecified validation gate; passing means candidate, not proven live recovery."""
    required = ['psnr','input_psnr','ssim','input_ssim','rmse','input_rmse','lpips',
                'input_lpips','stable_clear_change_mae','psnr_improved_fraction']
    if any(not isinstance(metrics.get(k),(int,float)) or not math.isfinite(metrics[k]) for k in required):
        return False
    return (metrics['psnr'] >= metrics['input_psnr']+.5
        and metrics['psnr'] >= baseline['psnr']+.1
        and metrics['ssim'] >= metrics['input_ssim']
        and metrics['rmse'] < metrics['input_rmse']
        and metrics['lpips'] <= metrics['input_lpips']
        and metrics['stable_clear_change_mae'] <= .01
        and metrics['stable_clear_change_mae_n'] >= max(1,metrics['count']//2)
        and metrics['psnr_improved_fraction'] >= .6)


def train_one(source, train_data, val_data, outdir, options, manifest_hash,
              pseudo, code_hash, device, deadline):
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    model, original = load_model(source, device)
    config = copy.deepcopy(original['config'])
    config['pseudo'] = asdict(pseudo)
    config['sentinel_finetuning'] = {k:v for k,v in options.items() if k not in ('session_hours','reserve_seconds','require_cuda','prepare_only')}
    config['sentinel_finetuning'].update(source_sha256=digest(source), code_sha256=code_hash,
        preprocessing='clip(DN / 10000, 0, 1); native 256 crops', identity_probability=.1)
    # Original loss implementation and source loss weights are retained.
    criterion = HybridLoss(LossConfig(**config.get('loss',{}))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=options['weight_decay'])
    last = outdir/'last.pt'
    if last.exists():
        state = torch.load(last,map_location='cpu',weights_only=True)
        saved_config = copy.deepcopy(state['config'])
        old_hash = saved_config.get('sentinel_finetuning',{}).get('code_sha256')
        approval_path = Path(__file__).with_name('resume_compatibility.json')
        approval = json.loads(approval_path.read_text()) if approval_path.exists() else {}
        compatible = old_hash != code_hash and approval.get('old_code_sha256') == old_hash and approval.get('new_code_sha256') == code_hash
        if compatible:
            saved_config['sentinel_finetuning']['code_sha256'] = code_hash
            if state['manifest_sha256'] == manifest_hash and saved_config == config:
                save_json(outdir/'resume_compatibility.json',dict(old_code_sha256=old_hash,
                    new_code_sha256=code_hash,completed_epochs=state['epoch'],
                    reason='Approved no_grad validation fix; model, loss, optimizer and dataset configuration unchanged'))
                state['config'] = saved_config
        if state['manifest_sha256'] != manifest_hash or state['config'] != config:
            raise ValueError('Resume data/code/configuration changed; refusing to mix experiments')
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
    else:
        baseline = assess(model,val_data,device,criterion.perceptual,outdir/'baseline',deadline)
        state = dict(config=config,manifest_sha256=manifest_hash,epoch=0,history=[],
            best_psnr=baseline['psnr'],bad_epochs=0,baseline=baseline,completed=False)
        state.update(model_state=model.state_dict(),optimizer_state=optimizer.state_dict())
        atomic_checkpoint(last,state)
        atomic_checkpoint(outdir/'best.pt',state)
    if state['completed']:
        print(f"Reusing completed {config['model']['variant']}: {state['epoch']} epochs; no retraining",flush=True)
        return outdir/'best.pt', state['baseline']
    for epoch in range(state['epoch'],options['epochs']):
        if time.monotonic() >= deadline:
            raise TimeoutError('Fine-tuning paused before next epoch')
        started = time.monotonic()
        seed_all(options['seed']+epoch)
        loader = DataLoader(train_data,batch_size=1,shuffle=True,num_workers=0)
        model.train(); optimizer.zero_grad(set_to_none=True)
        total = 0.0
        accumulation = options['accumulation_steps']
        for i,batch in enumerate(loader):
            if time.monotonic() >= deadline:
                # last.pt is the previous completed epoch; partial updates are discarded on resume.
                raise TimeoutError('Fine-tuning paused mid-epoch; resume repeats this epoch')
            batch = to_device(batch,device)
            loss,_ = criterion(model(batch['cloudy']),batch)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite fine-tuning loss')
            group_start = (i//accumulation)*accumulation
            group_size = min(accumulation,len(loader)-group_start)
            (loss/group_size).backward()
            if (i+1)%accumulation == 0 or i+1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
        scores = assess(model,val_data,device,criterion.perceptual,outdir/'latest_validation',deadline)
        improved = scores['psnr'] > state['best_psnr']
        bad = 0 if improved else state['bad_epochs']+1
        row = dict(epoch=epoch+1,train_loss=total/len(loader),validation_psnr=scores['psnr'],
                   validation_ssim=scores['ssim'],epoch_seconds=time.monotonic()-started)
        row['estimated_remaining_model_hours'] = row['epoch_seconds']*(options['epochs']-epoch-1)/3600
        state.update(epoch=epoch+1,history=state['history']+[row],
            best_psnr=max(state['best_psnr'],scores['psnr']),bad_epochs=bad,
            completed=epoch+1>=options['epochs'] or bad>=options['patience'],
            model_state=model.state_dict(),optimizer_state=optimizer.state_dict())
        atomic_checkpoint(last,state)
        if improved:
            atomic_checkpoint(outdir/'best.pt',state)
        save_json(outdir/'history.json',state['history'])
        print(json.dumps(dict(model=config['model']['variant'],**row)),flush=True)
        if state['completed']:
            break
    return outdir/'best.pt',state['baseline']


def package(root='.'):
    root = Path(root).resolve()
    target = root/'cloud_shadow_sentinel_finetune.zip'
    folders = ['finetune','data/sentinel','third_party/hsformer_runtime','cloud_severity','web']
    status_path = root/'finetune/status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('state') == 'candidate_exported':
        folders.append('deployment')
    with zipfile.ZipFile(target.with_suffix('.tmp'),'w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for folder in folders:
            for path in (root/folder).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.tmp','.part','.pyc'):
                    archive.write(path,path.relative_to(root).as_posix())
        for name in ['requirements-web.txt','README_FINETUNE.md']:
            if (root/name).exists():
                archive.write(root/name,name)
    target.with_suffix('.tmp').replace(target)
    return target


def run(source, options=None):
    cfg = {**DEFAULTS,**(options or {})}
    if cfg['epochs']<1 or cfg['patience']<1 or cfg['accumulation_steps']<1 or cfg['session_hours']<=0:
        raise ValueError('Invalid fine-tuning configuration')
    root = Path.cwd()
    output = root/'finetune'
    output.mkdir(exist_ok=True)
    deadline = time.monotonic()+cfg['session_hours']*3600-cfg['reserve_seconds']
    status = dict(state='incomplete',options=cfg,test_evaluated=False,live_satellite_validated=False)
    save_json(output/'status.json',status)
    try:
        set_download_deadline(deadline)
        if cfg['require_cuda'] and not cfg['prepare_only'] and not torch.cuda.is_available():
            raise RuntimeError('Enable Kaggle GPU T4 before this fine-tuning run')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(4); seed_all(cfg['seed'])
        paths = checkpoints(source,root)
        if Path(source).is_dir():
            runtime = Path(source)/'third_party/hsformer_runtime'
            target = root/'third_party/hsformer_runtime'
            if not target.exists(): shutil.copytree(runtime,target)
        protocol = prepare(root/'data/sentinel',cfg['train_count'],cfg['val_count'],cfg['seed'],deadline)
        if cfg['prepare_only']:
            status.update(state='data_prepared',message='Data verified; attach this ZIP to the GPU fine-tuning notebook with the original weights.')
            return status
        manifest_hash = digest(root/'data/sentinel/manifest.json')
        pseudo = calibrate(root/'data/sentinel',protocol)
        save_json(output/'pseudo_calibration.json',asdict(pseudo))
        train_data = SentinelPairs(root/'data/sentinel',protocol,'train',pseudo,True)
        val_data = SentinelPairs(root/'data/sentinel',protocol,'validation',pseudo,False)
        import hashlib
        code_files = sorted(Path(__file__).parent.glob('*.py')) + sorted((root/'third_party/hsformer_runtime').glob('*.py'))
        code_hash = hashlib.sha256(b''.join(p.parent.name.encode()+p.name.encode()+p.read_bytes() for p in code_files)).hexdigest()
        candidates, baselines, reports = {},{},{}
        for variant,path in paths.items():
            best,baseline = train_one(path,train_data,val_data,output/variant,cfg,manifest_hash,pseudo,code_hash,device,deadline)
            candidates[variant],baselines[variant] = best,baseline
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Both models must finish their common max-epoch/patience policy before selection.
        for variant,path in candidates.items():
            model,state = load_model(path,device)
            criterion = HybridLoss(LossConfig(**state['config'].get('loss',{}))).to(device)
            reports[variant] = assess(model,val_data,device,criterion.perceptual,output/variant/'final_validation',deadline)
            del model,criterion,state
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        rows = [dict(model=v,eligible=eligible(m,baselines[v]),**m) for v,m in reports.items()]
        pd.DataFrame(rows).to_csv(output/'comparison.csv',index=False)
        qualified = [v for v in reports if eligible(reports[v],baselines[v])]
        decision = dict(protocol='WHUS2-CRv RGB validation subset',baselines=baselines,metrics=reports,
            gate='PSNR >= input+0.5 dB and parent+0.1 dB; SSIM >= input; RMSE < input; LPIPS <= input; stable-change <= .01 with >=half patches covered; >=60% patches improve PSNR',
            qualified=qualified,final_test_evaluated=False,requires_visual_and_live_review=True)
        if qualified:
            winner = sorted(qualified,key=lambda v:(-reports[v]['psnr'],-reports[v]['ssim'],reports[v]['lpips']))[0]
            decision['winner'] = winner
            # This is a separate notebook workspace, never the original deployment directory.
            export_model(candidates[winner],decision,root/'deployment')
            pp = json.loads((root/'deployment/preprocessing.json').read_text())
            pp.update(satellite_reflectance_ceiling=1.0,satellite_scaling='clip(reflectance, 0, 1)',
                satellite_domain_validated=False,validation_scope='Small WHUS2-CRv RGB validation subset; not live or final test verification',
                training_dataset='WHUS2-CRv',candidate_requires_visual_review=True)
            save_json(root/'deployment/preprocessing.json',pp)
            status.update(state='candidate_exported',winner=winner)
        else:
            status.update(state='completed_no_qualified_model',winner=None)
        save_json(output/'decision.json',decision)
    except TimeoutError as exc:
        status.update(state='paused_budget',message=str(exc))
    except Exception as exc:
        status.update(state='failed',message=f'{type(exc).__name__}: {exc}')
        (output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print(status['message'],flush=True)
    finally:
        set_download_deadline(float('inf'))
        save_json(output/'status.json',status)
        print('Saved resumable output:',package(root),flush=True)
    return status
