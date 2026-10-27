"""Bounded synthetic-cloud pilot. Frozen HSFormer plus a trainable fusion head."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from cloud_severity.engine import load_model
from .data import validate, example
from .model import ReferenceFusion
from .prepare import sha

PARENT_SHA = '6c74416cc9521998105d155d7a3425095198d0efa510779f8fe537fcfcb43830'


def atomic_save(value, path):
    path = Path(path); tmp = path.with_suffix('.tmp')
    torch.save(value, tmp); tmp.replace(path)


def metrics(pred, target, mask, opaque, valid=None):
    valid = torch.ones_like(mask) if valid is None else valid
    mask = mask*valid; opaque = opaque*valid
    error = (pred-target).abs()
    mse = ((pred-target).square()*valid).sum()/(3*valid.sum()).clamp_min(1)
    return {'psnr': float(-10*torch.log10(mse.clamp_min(1e-12))),
            'cloud_mae': float((error*mask).sum()/(3*mask.sum()).clamp_min(1)),
            'opaque_mae': float((error*opaque).sum()/(3*opaque.sum()).clamp_min(1))}


def run(data, parent, output, epochs=6, hours=1.5):
    data, parent, output = Path(data), Path(parent), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = validate(data)
    if sha(parent) != PARENT_SHA:
        raise ValueError('Use HSFormer best.pt from the completed Sentinel (4) ZIP, not the refinement or failed epoch-zero checkpoint')
    if not torch.cuda.is_available(): raise RuntimeError('Enable Kaggle GPU T4 for the pilot')
    torch.set_num_threads(4); torch.manual_seed(321)
    device = torch.device('cuda'); start = time.monotonic()
    code = b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))
    config = {'version': 2, 'manifest_sha256': sha(data/'manifest.json'), 'parent_sha256': sha(parent),
              'code_sha256': hashlib.sha256(code).hexdigest(), 'epochs': epochs,
              'lr': .0002, 'batch_size': 1, 'accumulation': 4, 'samples_per_pair': 16,
              'seed': 321, 'notice': 'Synthetic clouds, oracle masks, L2A only; not validated on real thick clouds'}
    path = output/'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Changed experiment; use a new output folder')
    path.write_text(json.dumps(config, indent=2))
    cache = output/'cache'; cache.mkdir(exist_ok=True)
    model, _ = load_model(parent, device)
    for p in model.parameters(): p.requires_grad_(False)
    samples = {'train': [], 'validation': []}
    status = {'state': 'caching', 'real_cloud_validated': False, 'website_ready': False}

    def save_status(state):
        status.update(state=state, elapsed_seconds=time.monotonic()-start)
        (output/'status.json').write_text(json.dumps(status, indent=2))
        return status

    for i, row in enumerate(protocol['records']):
        for j in range(16):
            p = cache/f'{i:04d}-{j:02d}.pt'; samples[row['split']].append(p)
            if not p.exists():
                batch = example(data, row, seed=321+i*1000+j)
                with torch.no_grad():
                    batch['base'] = model(batch['cloudy'][None].to(device))['restored'][0].cpu()*batch['valid']
                batch['id'] = row['id']; atomic_save(batch, p)
            if time.monotonic()-start > hours*3600-120: return save_status('paused')
        print('Cached pair', i+1, '/', len(protocol['records']), flush=True)
    del model; torch.cuda.empty_cache()
    net = ReferenceFusion().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=1e-4)
    first = 0; best = float('inf'); history = []
    if (output/'last.pt').exists():
        state = torch.load(output/'last.pt', map_location=device, weights_only=True)
        net.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        first, best, history = state['epoch'], state['best'], state['history']

    def load(p):
        b = torch.load(p, weights_only=True)
        return {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v for k, v in b.items()}

    def predict(b):
        return net(b['cloudy'], b['base'], b['reference'], b['mask'], b['days'])

    def assess(save=False):
        rows = []; net.eval()
        with torch.no_grad():
            for i, p in enumerate(samples['validation']):
                b = load(p); pred = predict(b)
                outputs = {'cloudy': b['cloudy'], 'single_hsformer': b['base'],
                           'reference_copy': torch.where(b['mask'].bool(), b['reference'], b['cloudy']),
                           'temporal_fusion': pred}
                for name, value in outputs.items():
                    rows.append({'sample': p.stem, 'group_pair': b['id'], 'method': name,
                                 **metrics(value, b['target'], b['mask'], b['opaque'], b['valid'])})
                if save and i % 16 == 0:
                    panels = [b['cloudy'], b['reference'], b['base'], pred, b['target']]
                    arr = torch.cat([p*b['valid'] for p in panels], -1)[0].cpu().clamp(0, 1).pow(1/2.2)
                    Image.fromarray((arr.permute(1, 2, 0).numpy()*255).round().astype('uint8')).save(output/f'comparison_{i:03d}.png')
        summary = {name: {key: float(np.mean([r[key] for r in rows if r['method'] == name]))
                         for key in ('psnr', 'cloud_mae', 'opaque_mae')} for name in outputs}
        if save: (output/'per_sample.json').write_text(json.dumps(rows, indent=2))
        return summary

    for epoch in range(first, epochs):
        net.train(); opt.zero_grad(set_to_none=True)
        order = np.random.default_rng(321+epoch).permutation(len(samples['train']))
        for step, idx in enumerate(order):
            b = load(samples['train'][idx]); pred = predict(b)
            loss = ((pred-b['target']).abs()*b['mask']).sum()/(3*b['mask'].sum()).clamp_min(1)
            group_start = (step//4)*4
            (loss/min(4, len(order)-group_start)).backward()
            if (step+1) % 4 == 0 or step+1 == len(order):
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1); opt.step(); opt.zero_grad(set_to_none=True)
            if time.monotonic()-start > hours*3600-120:
                # Only completed epochs are resumable; current epoch will be repeated.
                return save_status('paused')
        report = assess(); score = report['temporal_fusion']['cloud_mae']
        history.append({'epoch': epoch+1, 'metrics': report})
        if score < best:
            best = score; atomic_save({'model': net.state_dict(), 'epoch': epoch+1, 'config': config}, output/'best.pt')
        atomic_save({'model': net.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch+1,
                     'best': best, 'history': history}, output/'last.pt')
        print('Epoch', epoch+1, report, flush=True)
        save_status('training')
    net.load_state_dict(torch.load(output/'best.pt', map_location=device, weights_only=True)['model'])
    report = assess(save=True)
    improved = all(report['temporal_fusion'][k] < report[n][k] for n in ('single_hsformer', 'reference_copy')
                   for k in ('cloud_mae', 'opaque_mae'))
    status.update(comparison=report, better_than_both_baselines=improved,
                  next_step='Real-cloud temporal validation required before any website deployment')
    (output/'history.json').write_text(json.dumps(history, indent=2))
    return save_status('pilot_complete')
