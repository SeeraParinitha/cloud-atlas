"""Recover completely unflagged common patches from cached preparation imagery."""
import hashlib
import itertools
import json
import zipfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window, transform, bounds
from clear_sky.service import quality
from .prepare import sha


def common_windows(risk_a, risk_b, size=256, max_flagged=0):
    if risk_a.shape != risk_b.shape: raise ValueError('Different mask grids')
    h, w = risk_a.shape
    if min(h, w) < size: return []
    risk = np.logical_or(risk_a, risk_b).astype('int64')
    integral = np.pad(risk, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    candidates = []
    for y in sorted(set(range(0, h-size+1, 32)) | {h-size}):
        for x in sorted(set(range(0, w-size+1, 32)) | {w-size}):
            count = integral[y+size, x+size]-integral[y, x+size]-integral[y+size, x]+integral[y, x]
            if count <= size*size*max_flagged: candidates.append((int(count), y, x))
    return [(y, x) for count, y, x in sorted(candidates)]


def recover(archive, destination):
    """Read archive without executing its code; emit standard compatible manifest."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'manifest.json').exists(): raise ValueError('Use a fresh recovery destination')
    records = []; reports = []
    with zipfile.ZipFile(archive) as z:
        config = json.loads(z.read('temporal_data/config.json'))
        for name, lat, lon, split in config['areas']:
            catalog = json.loads(z.read(f'temporal_data/{name}/catalog.json'))
            available = []
            for scene in catalog['scenes']:
                stamp = scene['datetime']; key = hashlib.sha256(stamp.encode()).hexdigest()[:16]
                member = f'temporal_data/{name}/{key}.tif'
                if member not in z.namelist(): continue
                raw = z.read(member); rgb, risk, stats = quality(raw)
                with MemoryFile(raw) as mem:
                    with mem.open() as ds:
                        available.append({'stamp': stamp, 'key': key, 'raw': raw,
                                          'risk': risk, 'rgb': rgb, 'profile': ds.profile.copy(),
                                          'source_sha256': hashlib.sha256(raw).hexdigest()})
            available.sort(key=lambda a: a['stamp'])
            options = []
            for a, b in itertools.combinations(available, 2):
                if a['stamp'][:10] == b['stamp'][:10]: continue
                if any(a['profile'][k] != b['profile'][k] for k in ('crs', 'transform', 'height', 'width')): continue
                windows = common_windows(a['risk'], b['risk'], max_flagged=.01)
                if windows: options.append((a, b, windows))
            used_positions = []; selected = 0
            # First choose spatially diverse patches, then different date pairs.
            for pass_index in (0, 1):
                for a, b, windows in options:
                    if selected >= 4: break
                    if any(r.get('source_dates') == [a['stamp'], b['stamp']] and r['group'] == name for r in records): continue
                    choices = [p for p in windows if pass_index or all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) >= 128 for q in used_positions)]
                    if not choices: continue
                    y, x = choices[0]
                    if np.array_equal(a['rgb'][:, y:y+256, x:x+256], b['rgb'][:, y:y+256, x:x+256]): continue
                    window = Window(x, y, 256, 256)
                    record = {'id': f'{name}-crop-{selected}', 'group': name, 'split': split,
                              'source_dates': [a['stamp'], b['stamp']], 'source_window': [y, x, 256, 256]}
                    for kind, item in (('reference', a), ('target', b)):
                        profile = item['profile'].copy()
                        profile.update(width=256, height=256, transform=transform(window, profile['transform']))
                        path = destination/name/f'{selected}-{kind}.tif'; path.parent.mkdir(exist_ok=True)
                        with MemoryFile(item['raw']) as mem:
                            with mem.open() as ds: pixels = ds.read(window=window)
                        with rasterio.open(path, 'w', **profile) as ds: ds.write(pixels)
                    # Re-run provider masks; store the joint mask from full images
                    # so buffering at crop borders is retained conservatively.
                        rgb, risk, stats = quality(path.read_bytes())
                        record[kind] = {'path': path.relative_to(destination).as_posix(), 'sha256': sha(path),
                                        'datetime': item['stamp'], 'pixels': hashlib.sha256(rgb.tobytes()).hexdigest(),
                                        'grid': [str(profile['crs']), list(profile['transform']), 256, 256],
                                        'source_sha256': item['source_sha256']}
                    record['bbox'] = list(bounds(window, a['profile']['transform']))
                    valid = (~(a['risk'] | b['risk']))[y:y+256, x:x+256].astype('uint8')
                    if valid.mean() < .99: raise ValueError('Insufficient usable pixels')
                    valid_path = destination/name/f'{selected}-valid.tif'
                    mask_profile = profile.copy(); mask_profile.update(count=1, dtype='uint8', nodata=None)
                    with rasterio.open(valid_path, 'w', **mask_profile) as ds: ds.write(valid, 1)
                    record['validity'] = {'path': valid_path.relative_to(destination).as_posix(),
                                          'sha256': sha(valid_path), 'fraction': float(valid.mean())}
                    records.append(record); used_positions.append((y, x)); selected += 1
            reports.append({'group': name, 'split': split, 'cached_dates': len(available),
                            'date_pairs_with_clear_windows': len(options), 'exported_pairs': selected})
            print(reports[-1], flush=True)
    # No model has trained on these data yet. A fixed backup holdout is allowed
    # only before training, based on availability, never reconstruction scores.
    training_groups = {r['group'] for r in records if r['split'] == 'train'}
    validation_groups = {r['group'] for r in records if r['split'] == 'validation'}
    reassigned = []
    if len(validation_groups) == 1 and len(training_groups) >= 5 and 'adelaide' in training_groups:
        for r in records:
            if r['group'] == 'adelaide': r['split'] = 'validation'
        reassigned = ['adelaide']
        for r in reports:
            if r['group'] == 'adelaide': r['split'] = 'validation'
    counts = {s: len({r['group'] for r in records if r['split'] == s}) for s in ('train', 'validation')}
    ready = counts['train'] >= 4 and counts['validation'] >= 2
    manifest = {'version': 1, 'ready': ready, 'group_counts': counts, 'records': records,
                'notice': 'Recovered >=99% jointly usable 256px patches; flagged pixels excluded; synthetic clouds only; location-disjoint split fixed before training',
                'validity_policy': 'joint_clear_mask_v1',
                'recovery': {'source_archive_sha256': sha(archive), 'clear_policy': 'at least 99% joint unflagged pixels; others excluded from inputs/loss/metrics',
                             'backup_validation_groups': reassigned,
                             'split_reason': 'Adelaide replaces unavailable validation coverage only before training; no metric-based selection'}}
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (destination/'recovery_report.json').write_text(json.dumps(reports, indent=2))
    if ready:
        from .data import validate
        validate(destination)
    print('READY:', ready, counts, flush=True)
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('archive'); parser.add_argument('destination')
    args = parser.parse_args(); recover(args.archive, args.destination)
