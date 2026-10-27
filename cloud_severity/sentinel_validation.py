"""Small, reproducible WHUS2-CRv RGB validation probe; never selects deployment."""
import argparse
import hashlib
import json
import random
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image, ImageDraw

from .engine import load_model, save_json
from .metrics import restoration_metrics
from .remote_validation import ValidationZip
from .tiling import restore_tiled


def prepare(root, count=24):
    if not 1 <= count <= 256:
        raise ValueError('Diagnostic sample count must be between 1 and 256')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'manifest.json'
    if manifest.exists():
        protocol = json.loads(manifest.read_text())
        if protocol['requested_count'] != count:
            raise ValueError('Use a new output folder when changing sample count')
        for row in protocol['pairs']:
            for key in ('cloudy', 'clear'):
                path = root / row[key]
                if hashlib.sha256(path.read_bytes()).hexdigest() != row[key+'_sha256']:
                    raise ValueError('Cached validation sample changed')
        return protocol
    archive = ValidationZip()
    groups = defaultdict(list)
    for name in sorted(archive.members):
        if name.startswith('val/clearDNclips/10m/') and name.endswith('.tif'):
            cloudy = name.replace('/clearDNclips/', '/cloudDNclips/')
            if cloudy not in archive.members:
                raise ValueError('Missing official pair: '+name)
            groups[name.split('/')[-2]].append(name)
    rng = random.Random(123)
    group_names = sorted(groups)
    rng.shuffle(group_names)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count:
        progress = False
        for group in group_names:
            if groups[group] and len(selected) < count:
                selected.append(groups[group].pop())
                progress = True
        if not progress:
            raise ValueError('Requested more samples than available')
    def fetch_pair(item):
        i, clear = item
        row = {'id': '/'.join(clear.split('/')[-2:]), 'group': clear.split('/')[-2]}
        for key, member in [('clear', clear), ('cloudy', clear.replace('/clearDNclips/', '/cloudDNclips/'))]:
            path = root / f'{i:03d}_{key}.tif'
            raw = path.read_bytes() if path.exists() else b''
            info = archive.members[member]
            if len(raw) != info['unpacked'] or zlib.crc32(raw) != info['crc']:
                raw = archive.read(member)
                path.write_bytes(raw)
            row[key] = path.name
            row[key+'_sha256'] = hashlib.sha256(raw).hexdigest()
            row[key+'_member'] = member
        print(f'Validated downloaded pair {i+1}/{count}', flush=True)
        return row
    # Bounded I/O concurrency; map preserves deterministic manifest order.
    with ThreadPoolExecutor(max_workers=3) as pool:
        pairs = list(pool.map(fetch_pair, enumerate(selected)))
    protocol = dict(dataset='WHUS2-CRv', split='validation', requested_count=count,
        source='https://zenodo.org/records/8035349', sample_seed=123,
        sampling='Round robin over shuffled official scene folders; not a full benchmark',
        rgb_indices_one_based=[3, 2, 1], dn_divisor=10000,
        band_evidence='https://github.com/Neooolee/WHUS2-CRv/blob/main/allnet.py (get_write_picture)',
        scale_evidence='https://github.com/Neooolee/WHUS2-CRv/blob/main/alltrain.py (scale=10000)',
        alignment='Official paired patch correspondence, equal shape checked; TIFFs lack georeferencing. No independent subpixel registration validation.',
        pairs=pairs)
    save_json(manifest, protocol)
    return protocol


class InvalidPairError(ValueError):
    """A specific pair fails conservative pixel/shape checks, not a network error."""


def read_pair(root, row):
    arrays = []
    for key in ('cloudy', 'clear'):
        with rasterio.open(Path(root)/row[key]) as source:
            if source.count != 4 or source.dtypes != ('uint16',)*4:
                raise ValueError('Expected official four-band 10 m uint16 patches')
            if not source.read_masks([3, 2, 1]).all():
                raise InvalidPairError('Invalid pixels: prepare an explicit validity-mask evaluation')
            array = source.read([3, 2, 1]).astype(np.float32)/10000
        if not np.isfinite(array).all() or np.any(np.all(array == 0, axis=0)):
            raise InvalidPairError('Nonfinite or all-zero RGB pixels; do not score as ground truth')
        arrays.append(torch.from_numpy(array))
    if arrays[0].shape != arrays[1].shape:
        raise InvalidPairError('Pair shape mismatch')
    return arrays


def checkpoints(source, destination):
    source, destination = Path(source), Path(destination)
    required = [f'runs/{v}/best.pt' for v in ('full', 'hsformer')]
    if source.is_dir():
        if not all((source/n).is_file() for n in required):
            raise ValueError('Attach the full training output with BOTH runs/*/best.pt files')
        return {v: source/f'runs/{v}/best.pt' for v in ('full', 'hsformer')}
    with zipfile.ZipFile(source) as archive:
        matches = [n for n in archive.namelist() if n.endswith(required[0])]
        if len(matches) != 1:
            raise ValueError('Expected one original full training archive')
        prefix = matches[0][:-len(required[0])]
        names = required + [n[len(prefix):] for n in archive.namelist()
            if n.startswith(prefix+'third_party/hsformer_runtime/') and n.endswith('.py')]
        for name in names:
            target = (destination/name).resolve()
            if destination.resolve() not in target.parents or '\\' in name:
                raise ValueError('Unsafe archive path')
            info = archive.getinfo(prefix+name)
            if info.file_size > 512*1024**2 or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unexpected checkpoint member')
            data = archive.read(info)
            if target.exists() and target.read_bytes() != data:
                raise ValueError('Different checkpoint/runtime already exists: '+str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return {v: destination/f'runs/{v}/best.pt' for v in ('full', 'hsformer')}


def evaluate(source, output='runs/sentinel_probe', count=24, ceilings=(.4, 1.0), device=None):
    if not ceilings or any(not np.isfinite(c) or c <= 0 for c in ceilings):
        raise ValueError('Scaling ceilings must be finite and positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/'status.json', dict(state='incomplete', count=count, ceilings=list(ceilings)))
    paths = checkpoints(source, Path.cwd())
    protocol = prepare(output/'data', count)
    # In the standalone notebook the working directory is the project directory.
    if Path(source).is_dir():
        import shutil
        runtime = Path(source)/'third_party/hsformer_runtime'
        target = Path('third_party/hsformer_runtime')
        if not target.exists():
            shutil.copytree(runtime, target)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    rows = []
    for variant, path in paths.items():
        model, state = load_model(path, torch.device(device))
        if state['config']['model']['variant'] != variant:
            raise ValueError('Wrong model variant in checkpoint')
        for ceiling in ceilings:
            for index, pair in enumerate(protocol['pairs']):
                cloudy, clear = read_pair(output/'data', pair)
                x, target = (cloudy/ceiling).clamp(0, 1), (clear/ceiling).clamp(0, 1)
                prediction = restore_tiled(model, x, device)['restored']
                result = dict(model=variant, ceiling=ceiling, id=pair['id'], group=pair['group'])
                result.update(restoration_metrics(prediction, target))
                baseline = restoration_metrics(x, target)
                result.update({'input_'+k: v for k, v in baseline.items()})
                result['psnr_gain'] = result['psnr']-baseline['psnr']
                result['ssim_gain'] = result['ssim']-baseline['ssim']
                # Same un-clipped physical reference across scaling experiments.
                result['reflectance_mae'] = float((prediction*ceiling-clear).abs().mean())
                result['input_reflectance_mae'] = float((cloudy-clear).abs().mean())
                result['input_clipped_fraction'] = float((cloudy>=ceiling).any(0).float().mean())
                result['target_clipped_fraction'] = float((clear>=ceiling).any(0).float().mean())
                result['edit_mae'] = float((prediction-x).abs().mean())
                rows.append(result)
                pd.DataFrame(rows).to_csv(output/'per_image.csv', index=False)
                if index < 6:
                    tiles = [Image.fromarray((a.permute(1,2,0).numpy()*255).round().astype('uint8'))
                             for a in (x, prediction, target)]
                    w, h = tiles[0].size
                    panel = Image.new('RGB', (w*3, h+24), 'white')
                    draw = ImageDraw.Draw(panel)
                    for j, (tile, label) in enumerate(zip(tiles, ['Cloudy', variant+' prediction', 'Reference (different date)'])):
                        panel.paste(tile, (j*w, 24)); draw.text((j*w+4, 4), label, fill='black')
                    panel.save(output/f'{variant}_scale{ceiling}_{index:02d}.png')
                print(f'{variant} scale={ceiling}: {index+1}/{count}', flush=True)
        del model, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    numeric = table.select_dtypes(include='number').columns.difference(['ceiling'])
    summary = table.groupby(['model','ceiling'])[numeric].mean().reset_index()
    summary.to_csv(output/'summary.csv', index=False)
    save_json(output/'status.json', dict(state='validation_probe_complete', dataset='WHUS2-CRv',
        count=count, ceilings=list(ceilings), checkpoints={v:hashlib.sha256(p.read_bytes()).hexdigest() for v,p in paths.items()},
        limitations=['Small RGB validation subset, not official full test results',
                    'Compare RGB scores within a scale; targets change across scales',
                    'Temporal differences and residual misregistration may affect scores',
                    'No cloud-mask ground truth or thick-cloud guarantee',
                    'No automatic model selection, deployment modification, or training']))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', default='runs/sentinel_probe')
    parser.add_argument('--count', type=int, default=24)
    args = parser.parse_args()
    print(evaluate(args.source, args.output, args.count).to_string(index=False))
