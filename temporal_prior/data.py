import json
from datetime import datetime
from pathlib import Path
import numpy as np
import rasterio
import torch
from torch.nn import functional as F
from .prepare import sha


def validate(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if not manifest.get('ready'): raise ValueError('Insufficient clear temporal pairs: finish CPU preparation first')
    if manifest.get('validity_policy') not in (None, 'joint_clear_mask_v1'): raise ValueError('Unknown validity policy')
    groups = {}; contents = {}; ids = set()
    for r in manifest['records']:
        split = r['split']
        if split not in ('train', 'validation'): raise ValueError('Unknown split')
        if r['id'] in ids: raise ValueError('Duplicate sample ID')
        ids.add(r['id'])
        if r['group'] in groups and groups[r['group']] != split: raise ValueError('Location leakage')
        groups[r['group']] = split
        a, b = r['reference'], r['target']
        if a['datetime'][:10] >= b['datetime'][:10]: raise ValueError('Reference must be from an earlier date')
        if a['grid'] != b['grid'] or a['pixels'] == b['pixels']: raise ValueError('Invalid independent reference')
        for item in (a, b):
            path = (root/item['path']).resolve()
            if root not in path.parents or sha(path) != item['sha256']: raise ValueError('Changed or unsafe data file')
            with rasterio.open(path) as ds:
                rgb = ds.read([1, 2, 3])
                if [str(ds.crs), list(ds.transform), ds.width, ds.height] != item['grid']: raise ValueError('Changed grid')
            import hashlib
            if hashlib.sha256(rgb.tobytes()).hexdigest() != item['pixels']: raise ValueError('Pixel checksum mismatch')
            if not np.isfinite(rgb).all() or min(rgb.shape[-2:]) < 256: raise ValueError('Invalid image')
            h = item['pixels']
            if h in contents and contents[h] != split: raise ValueError('Image leakage across splits')
            contents[h] = split
        if manifest.get('validity_policy') == 'joint_clear_mask_v1':
            from clear_sky.service import quality
            item = r['validity']; path = (root/item['path']).resolve()
            if root not in path.parents or sha(path) != item['sha256']: raise ValueError('Invalid validity mask checksum/path')
            with rasterio.open(path) as ds:
                usable = ds.read(1)
                if [str(ds.crs), list(ds.transform), ds.width, ds.height] != b['grid']: raise ValueError('Mask grid mismatch')
            if not np.isin(usable, [0, 1]).all() or usable.mean() < .99: raise ValueError('Invalid usable pixel mask')
            for item in (a, b):
                _, risk, _ = quality((root/item['path']).read_bytes())
                if (risk & usable.astype(bool)).any(): raise ValueError('Flagged pixels included in training labels')
    if sum(v == 'train' for v in groups.values()) < 4 or sum(v == 'validation' for v in groups.values()) < 2:
        raise ValueError('Need at least four training and two validation locations')
    return manifest


def example(root, record, seed):
    """Fixed independent RNG; the label never enters reference features."""
    rng = np.random.default_rng(seed)
    arrays = []
    for kind in ('target', 'reference'):
        with rasterio.open(Path(root)/record[kind]['path']) as ds:
            arrays.append(torch.from_numpy(ds.read([1, 2, 3]).astype('float32')).clamp(0, 1))
    h, w = arrays[0].shape[-2:]
    y, x = int(rng.integers(h-255)), int(rng.integers(w-255))
    target, reference = [a[:, y:y+256, x:x+256] for a in arrays]
    valid = torch.ones_like(target[:1])
    if 'validity' in record:
        with rasterio.open(Path(root)/record['validity']['path']) as ds:
            valid = torch.from_numpy(ds.read([1]).astype('float32'))[:, y:y+256, x:x+256]
    # Unknown pixels are excluded, not used as clear labels or temporal features.
    target = target*valid; reference = reference*valid
    field = torch.tensor(rng.random((1, 1, 8, 8)), dtype=torch.float32)
    field = F.interpolate(field, (256, 256), mode='bicubic', align_corners=False)[0]
    alpha = ((field-.30)*2.8).clamp(0, 1)
    # Include fully opaque clouds, with soft edges elsewhere.
    alpha[:, 80:176, 80:176] = 1
    mask = (alpha > 0).float()*valid
    cloudy = (target*(1-alpha)+float(rng.uniform(.75, 1))*alpha)*valid
    gap = (datetime.fromisoformat(record['target']['datetime'].replace('Z', '+00:00'))-
           datetime.fromisoformat(record['reference']['datetime'].replace('Z', '+00:00'))).total_seconds()/86400
    return {'target': target, 'reference': reference, 'cloudy': cloudy, 'mask': mask,
            'opaque': (alpha >= .99).float()*valid, 'valid': valid,
            'days': torch.tensor(gap, dtype=torch.float32)}
