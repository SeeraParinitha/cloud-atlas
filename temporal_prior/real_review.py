"""Real-cloud qualitative evaluation on held-out locations, without new downloads."""
import hashlib
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from rasterio.io import MemoryFile
from clear_sky.service import quality, ClearClient, buffer_mask
from cloud_severity.engine import load_model
from .model import ReferenceFusion
from .prepare import sha, AREAS, bbox_for
from .train import PARENT_SHA


def collect(output, client=None):
    """Bounded API collection; cache each acquisition, including interrupted runs."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    client = client or ClearClient()
    cfgpath = output/'collection.json'
    if cfgpath.exists(): cfg = json.loads(cfgpath.read_text())
    else:
        end = datetime.now(timezone.utc).date()-timedelta(days=1)
        cfg = {'start': (end-timedelta(days=90)).isoformat(), 'end': end.isoformat(),
               'locations': ['adelaide', 'fresno'], 'max_requests_per_location': 16}
        cfgpath.write_text(json.dumps(cfg, indent=2))
    for name, lat, lon, _ in AREAS:
        if name not in cfg['locations']: continue
        folder = output/'temporal_data'/name; folder.mkdir(parents=True, exist_ok=True)
        catalog_path = folder/'catalog.json'; bbox = bbox_for(lat, lon)
        if not catalog_path.exists(): catalog_path.write_text(json.dumps(client.search(bbox, cfg['start'], cfg['end'])))
        catalog = json.loads(catalog_path.read_text())
        ranked = sorted(catalog['scenes'], key=lambda s: (s.get('cloud_cover') if s.get('cloud_cover') is not None else 50, s['datetime']))
        # Both low- and high-cloud scenes, chosen before model inference.
        selected = {s['datetime']: s for s in ranked[:8]+ranked[-8:]}
        for stamp in sorted(selected):
            key = hashlib.sha256(stamp.encode()).hexdigest()[:16]
            path = folder/(key+'.tif')
            if not path.exists():
                raw = client.fetch(bbox, stamp); read_scene(raw)
                temp = path.with_suffix('.tmp'); temp.write_bytes(raw); temp.replace(path)
            print('Cached real observation:', name, stamp, flush=True)
    archive = output/'observations.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in (output/'temporal_data').rglob('*'):
            if p.is_file() and p.suffix != '.tmp': z.write(p, p.relative_to(output))
    return archive


def read_scene(raw):
    rgb, risk, stats = quality(raw)
    with MemoryFile(raw) as mem:
        with mem.open() as ds:
            a = ds.read(); grid = [str(ds.crs), list(ds.transform), ds.width, ds.height]
    valid = ((a[5] > .5) & np.isfinite(a).all(0) & (a[3] >= 0) & (a[3] <= 11)
             & (a[3] == np.floor(a[3])) & (a[4] >= 0) & (a[4] <= 100))
    cloud = np.isin(a[3], [3, 8, 9, 10]) | (a[4] >= 40)
    cloud = buffer_mask(cloud)
    usable = valid & (np.isin(a[3], [4, 5, 6]) | cloud)
    return dict(rgb=rgb, risk=risk, cloud=cloud, usable=usable, grid=grid)


def select_cases(archive, groups, max_cases=3):
    cases = []
    with zipfile.ZipFile(archive) as z:
        for group in groups:
            catalog = json.loads(z.read(f'temporal_data/{group}/catalog.json'))
            scenes = []
            for item in catalog['scenes']:
                stamp = item['datetime']; key = hashlib.sha256(stamp.encode()).hexdigest()[:16]
                name = f'temporal_data/{group}/{key}.tif'
                if name not in z.namelist(): continue
                scenes.append(dict(datetime=stamp, **read_scene(z.read(name))))
            # Selection is fixed before loading models and uses provider masks only.
            targets = sorted(scenes, key=lambda s: s['datetime'], reverse=True)
            selected = 0
            for target in targets:
                refs = sorted((s for s in scenes if s['datetime'][:10] < target['datetime'][:10]),
                              key=lambda s: s['datetime'], reverse=True)
                chosen = None
                for ref in refs:
                    if target['grid'] != ref['grid']: continue
                    h, w = target['cloud'].shape
                    options = []
                    for y in sorted(set(range(0, h-255, 64)) | {h-256}):
                        for x in sorted(set(range(0, w-255, 64)) | {w-256}):
                            if min(y, x) < 0: continue
                            sl = np.s_[y:y+256, x:x+256]
                            usable = target['usable'][sl] & ~ref['risk'][sl]
                            fraction = float((target['cloud'][sl] & usable).mean())
                            if usable.mean() >= .99 and fraction >= .10:
                                options.append((fraction, y, x))
                    if options:
                        fraction, y, x = max(options)
                        chosen = dict(group=group, target=target, reference=ref, y=y, x=x,
                                      provider_cloud_shadow_fraction=fraction)
                        break  # Nearest earlier reference with sufficient usable coverage.
                if chosen:
                    cases.append(chosen); selected += 1
                if selected >= max_cases: break
    return cases


def panel(images, labels, path):
    canvas = Image.new('RGB', (256*len(images), 282), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (a, label) in enumerate(zip(images, labels)):
        rgb = np.clip(a, 0, 1)**(1/2.2)
        canvas.paste(Image.fromarray((rgb.transpose(1, 2, 0)*255).round().astype('uint8')), (i*256, 26))
        draw.text((i*256+5, 5), label, fill='black')
    canvas.save(path)


def run(data_zip, pilot_zip, parent, manifest, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(manifest).read_text())
    held_out = sorted({r['group'] for r in protocol['records'] if r['split'] == 'validation'})
    if sha(parent) != PARENT_SHA: raise ValueError('Incorrect frozen parent')
    with zipfile.ZipFile(pilot_zip) as z:
        status = json.loads(z.read('pilot_results/status.json'))
        cfg = json.loads(z.read('pilot_results/config.json'))
        if status['state'] != 'pilot_complete' or cfg['manifest_sha256'] != sha(manifest):
            raise ValueError('Incomplete pilot or different location split')
        weight = output/'fusion.pt'; weight.write_bytes(z.read('pilot_results/best.pt'))
    cases = select_cases(data_zip, held_out)
    if not cases:
        report = {'state': 'no_suitable_pairs', 'website_ready': False, 'cases': [],
                  'notice': 'No qualifying pairs among bounded observations. This is not proof that no reference exists.'}
        (output/'report.json').write_text(json.dumps(report, indent=2))
        return report
    torch.set_num_threads(2); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base, _ = load_model(parent, device)
    fusion = ReferenceFusion().to(device)
    checkpoint = torch.load(weight, map_location=device, weights_only=True)
    if checkpoint['config'] != cfg: raise ValueError('Checkpoint/config mismatch')
    fusion.load_state_dict(checkpoint['model']); fusion.eval()
    rows = []
    for i, case in enumerate(cases):
        y, x = case['y'], case['x']; sl = np.s_[y:y+256, x:x+256]
        t, r = case['target'], case['reference']
        valid = t['usable'][sl] & ~r['risk'][sl]
        mask = t['cloud'][sl] & valid
        target = np.clip(t['rgb'][:, sl[0], sl[1]], 0, 1)*valid
        reference = np.clip(r['rgb'][:, sl[0], sl[1]], 0, 1)*valid
        days = (datetime.fromisoformat(t['datetime'].replace('Z', '+00:00'))-
                datetime.fromisoformat(r['datetime'].replace('Z', '+00:00'))).total_seconds()/86400
        def tensor(a): return torch.as_tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            single = base(tensor(target))['restored']*tensor(valid[None])
            restored = fusion(tensor(target), single, tensor(reference), tensor(mask[None]),
                              torch.tensor([days], device=device))[0].cpu().numpy()*valid
        single = single[0].cpu().numpy()
        copied = np.where(mask[None], reference, target)
        ident = f'{i:02d}-{case["group"]}'
        panel([target, reference, single, restored, copied],
              ['Real cloudy target', 'Earlier reference', 'Single HSFormer', 'Learned temporal fusion', 'Reference-copy baseline'],
              output/(ident+'.png'))
        np.savez_compressed(output/(ident+'.npz'), target=target, reference=reference,
                            single=single, fusion=restored, copy=copied, valid=valid, cloud_mask=mask)
        Image.fromarray(mask.astype('uint8')*255).save(output/(ident+'-mask.png'))
        clear = valid & ~mask
        row = dict(id=ident, target_date=t['datetime'], reference_date=r['datetime'], days_gap=days,
                   source_window=[y,x,256,256], valid_fraction=float(valid.mean()),
                   provider_cloud_shadow_fraction=float(mask.mean()),
                   observed_clear_change_mae=float(np.abs(restored-target)[:, clear].mean()) if clear.any() else None,
                   psnr=None, ground_truth_available=False)
        rows.append(row); print(row, flush=True)
    report = dict(state='qualitative_review_complete', website_ready=False,
                  real_cloud_accuracy_validated=False, fusion_epoch=checkpoint['epoch'],
                  parent_sha256=sha(parent), fusion_sha256=sha(weight), held_out_locations=held_out,
                  source_archive_sha256=sha(data_zip), cases=rows,
                  notice='Actual L2A clouds, provider masks, earlier references. No same-date clear truth; no PSNR. Fixed model; no retraining. Grid alignment checked, not subpixel registration.')
    (output/'report.json').write_text(json.dumps(report, indent=2))
    return report
