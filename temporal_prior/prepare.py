"""CPU collection of distinct, independently observed clear dates. No training."""
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from rasterio.io import MemoryFile
from clear_sky.service import ClearClient, quality

# Deliberately separated locations; the validation locations never train.
AREAS = [
    ('pune', 18.52, 73.86, 'train'), ('jaipur', 26.91, 75.79, 'train'),
    ('sacramento', 38.58, -121.49, 'train'), ('seville', 37.39, -5.98, 'train'),
    ('adelaide', -34.93, 138.60, 'train'), ('nairobi', -1.29, 36.82, 'train'),
    ('nagpur', 21.15, 79.09, 'validation'), ('fresno', 36.74, -119.78, 'validation')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bbox_for(lat, lon):
    dy = 2560 / 111320
    dx = dy / math.cos(math.radians(lat))
    return [lon-dx, lat-dy, lon+dx, lat+dy]


def prepare(root, end=None, client=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    client = client or ClearClient()
    end = end or (datetime.now(timezone.utc).date()-timedelta(days=7)).isoformat()
    start = (datetime.fromisoformat(end)-timedelta(days=90)).date().isoformat()
    config = {'start': start, 'end': end, 'areas': AREAS, 'candidate_limit': 16,
              'clear_dates_per_area': 4, 'collection': 'sentinel-2-l2a', 'version': 1}
    config = json.loads(json.dumps(config))
    cfgpath = root/'config.json'
    if cfgpath.exists() and json.loads(cfgpath.read_text()) != config:
        raise ValueError('Preparation configuration changed; use a new output folder or original end date')
    cfgpath.write_text(json.dumps(config, indent=2))
    records = []; reports = []
    for name, lat, lon, split in AREAS:
        folder = root/name; folder.mkdir(exist_ok=True)
        bbox = bbox_for(lat, lon)
        catalog_file = folder/'catalog.json'
        if not catalog_file.exists():
            catalog_file.write_text(json.dumps(client.search(bbox, start, end)))
        catalog = json.loads(catalog_file.read_text())
        scenes = sorted(catalog['scenes'], key=lambda s: (s.get('cloud_cover') if s.get('cloud_cover') is not None else 101, s['datetime']))
        clear = []; checked = []
        for scene in scenes[:16]:
            stamp = scene['datetime']
            key = hashlib.sha256(stamp.encode()).hexdigest()[:16]
            path = folder/(key+'.tif')
            if not path.exists():
                raw = client.fetch(bbox, stamp)
                temp = path.with_suffix('.tmp'); temp.write_bytes(raw); temp.replace(path)
            raw = path.read_bytes(); rgb, risk, stats = quality(raw)
            checked.append({'datetime': stamp, **stats})
            # Require completely unflagged labels/reference for this controlled test.
            if stats['flagged_fraction'] != 0 or stats['valid_fraction'] != 1:
                continue
            with MemoryFile(raw) as mem:
                with mem.open() as ds:
                    grid = [str(ds.crs), list(ds.transform), ds.width, ds.height]
            if min(rgb.shape[-2:]) < 256 or not np.isfinite(rgb).all():
                continue
            if clear and grid != clear[0]['grid']:
                raise ValueError('Reference/target grid mismatch')
            pixels = hashlib.sha256(rgb.tobytes()).hexdigest()
            if any(c['pixels'] == pixels or c['datetime'][:10] == stamp[:10] for c in clear):
                continue
            clear.append({'path': path.relative_to(root).as_posix(), 'sha256': sha(path),
                          'datetime': stamp, 'pixels': pixels, 'grid': grid})
            if len(clear) == 4: break
        clear.sort(key=lambda c: c['datetime'])
        # Earlier independent reference only. Clear target is never the reference.
        for i in range(1, len(clear)):
            ref, target = clear[i-1], clear[i]
            records.append({'id': f'{name}-{i}', 'group': name, 'split': split, 'bbox': bbox,
                            'reference': ref, 'target': target})
        reports.append({'group': name, 'clear_dates': len(clear), 'checked': checked,
                        'catalog_truncated': catalog.get('truncated', False)})
        print(name, 'accepted clear dates:', len(clear), flush=True)
        (root/'preparation_report.json').write_text(json.dumps(reports, indent=2))
    counts = {s: len({r['group'] for r in records if r['split'] == s}) for s in ('train', 'validation')}
    ready = counts['train'] >= 4 and counts['validation'] >= 2
    manifest = {'version': 1, 'ready': ready, 'group_counts': counts, 'records': records,
                'notice': 'Real independent L2A dates; target clouds are simulated. No real thick-cloud claim.'}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Preparation ready:', ready, counts, flush=True)
    return manifest
