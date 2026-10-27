"""Experimental reference-assisted inference on the user's exact acquisition."""
import json
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.io import MemoryFile

from clear_sky.service import ClearClient
from cloud_severity.engine import load_model
from cloud_severity.satellite import check_area
from temporal_prior.real_review import read_scene
from temporal_prior.train import PARENT_SHA
from .data import digest, days
from .full_area import tiled, png
from .model import BoundedFusion

NOTICE = ('Experimental reconstruction using an earlier observation. Hidden selected-date '
          'details are unverified; reference-date land cover may appear in the estimate. '
          'Unsupported pixels remain original. Mixture weights are not confidence. '
          'Earlier proxy evaluation did not outperform direct reference copying.')


class TemporalService:
    def __init__(self, parent, checkpoint, output='temporal_web_results', client=None, multi_reference=False):
        self.multi_reference = bool(multi_reference)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.client = client or ClearClient()
        self.lock = threading.Lock()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if digest(parent) != PARENT_SHA:
            raise ValueError('Incorrect HSFormer parent checkpoint')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (state['config']['model'] != 'bounded_fusion_v1'
                or state['config']['parent_sha256'] != PARENT_SHA):
            raise ValueError('Incorrect bounded fusion checkpoint')
        self.provenance = {'hsformer_sha256': PARENT_SHA, 'fusion_sha256': digest(checkpoint)}
        torch.set_num_threads(2)
        self.backbone, _ = load_model(parent, self.device)
        self.fusion = BoundedFusion().to(self.device)
        self.fusion.load_state_dict(state['model'])
        self.fusion.eval()

    def status(self):
        return {'model_ready': True, 'satellite_ready': self.client.configured,
                'device': self.device, 'experimental': True}

    def restore(self, bbox, acquisition, progress=lambda message: None):
        with self.lock:
            return self._restore(bbox, acquisition, progress)

    def _restore(self, bbox, acquisition, progress):
        check_area(bbox)
        started = time.perf_counter()
        dt = datetime.fromisoformat(acquisition.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            raise ValueError('Acquisition requires a timezone')
        catalog = self.client.search(bbox, dt.date().isoformat(), dt.date().isoformat())
        if acquisition not in {s['datetime'] for s in catalog['scenes']}:
            raise ValueError('Selected acquisition is not available for this area')
        progress('Retrieving your selected acquisition')
        raw = self.client.fetch(bbox, acquisition)
        a = read_scene(raw)
        if not a['usable'].any():
            raise ValueError('No usable target pixels for this area')
        x = np.nan_to_num(a['rgb'], nan=0, posinf=1, neginf=0).clip(0, 1)
        mask = a['cloud'] & a['usable']
        best = None
        attempts = []
        candidates = []
        multi_report = None
        truncated = False
        if mask.any():
            progress('Searching earlier reference observations (up to 60 days)')
            previous = self.client.search(bbox, (dt-timedelta(days=60)).date().isoformat(),
                                          (dt-timedelta(days=1)).date().isoformat())
            truncated = previous.get('truncated', False)
            refs = [s for s in previous['scenes']
                    if s['datetime'][:10] < acquisition[:10] and 0 < days(acquisition, s['datetime']) <= 60]
            refs.sort(key=lambda s: (s.get('cloud_cover') if s.get('cloud_cover') is not None else 101,
                                     days(acquisition, s['datetime'])))
            for i, s in enumerate(refs[:8]):
                progress(f'Checking reference {i+1} of {min(8, len(refs))}')
                rr = self.client.fetch(bbox, s['datetime'])
                b = read_scene(rr)
                if a['grid'] != b['grid']:
                    attempts.append({'date': s['datetime'], 'state': 'grid_mismatch'})
                    continue
                coverage = float((mask & ~b['risk']).sum()/mask.sum())
                candidates.append({'scene': b, 'raw': rr, 'date': s['datetime'],
                                   'age_days': days(acquisition, s['datetime'])})
                attempts.append({'date': s['datetime'], 'cloud_reference_coverage': coverage})
                if best is None or coverage > best[0]:
                    best = (coverage, rr, b, s['datetime'])
        ident = uuid.uuid4().hex
        folder = self.output/ident
        folder.mkdir()
        (folder/'source_reflectance.tif').write_bytes(raw)
        png(x, folder/'original.png')
        restored = x.copy()
        edit = np.zeros_like(mask)
        weights = None
        reference_date = None
        state = 'no_cloud_flagged' if not mask.any() else 'no_usable_reference'
        if best is not None and best[0] > 0:
            _, rr, b, reference_date = best
            r = np.nan_to_num(b['rgb'], nan=0, posinf=1, neginf=0).clip(0, 1)
            valid = a['usable'] & ~b['risk']
            edit = mask & valid
            restored, _, weights = tiled(self.backbone, self.fusion, x, r, mask, valid,
                                         days(acquisition, reference_date), self.device,
                                         progress=lambda n, total: progress(f'Reconstructing tile {n}/{total}'))
            png(r, folder/'reference.png')
            (folder/'reference_source.tif').write_bytes(rr)
            if getattr(self, 'multi_reference', False):
                from .multi_reference import reconstruct
                anchor_index = next(i for i,c in enumerate(candidates) if c['date']==reference_date)
                a['rgb'] = x
                restored, composite, weights, source_index, checks = reconstruct(
                    a, candidates, anchor_index,
                    lambda tx, tr, tm, tv, age: tiled(self.backbone,self.fusion,tx,tr,tm,tv,age,self.device))
                original_edit = edit.copy()
                edit = source_index >= 0
                np.save(folder/'reference_source_index.npy', source_index)
                png(composite, folder/'multi_reference.png')
                used=[]
                for index in np.unique(source_index[edit]):
                    c=candidates[int(index)]; filename=f'reference_{int(index)}.tif'
                    (folder/filename).write_bytes(c['raw'])
                    used.append({'index':int(index),'date':c['date'],'age_days':c['age_days'],'file':filename,
                                 'selected_pixels':int((source_index==index).sum())})
                multi_report={'experimental':True,'references':used,'alignment_checks':checks,
                              'added_pixels':int((edit&~original_edit).sum()),
                              'baseline_alignment':'grid match only; not newly verified',
                              'reference_preview':'reference.png is baseline; multi_reference.png shows selected reference pixels only',
                              'method':'Separate frozen-model inference per reference; fill only previously unsupported flagged pixels. No learned weights changed.'}
            state = 'partial_reconstruction' if (mask & ~edit).any() else 'reconstruction_complete'
        png(restored, folder/'restored.png')
        for name, value in [('edit_mask', edit), ('unprocessed_cloud_mask', mask & ~edit)]:
            Image.fromarray(value.astype('uint8')*255).save(folder/(name+'.png'))
        with MemoryFile(raw) as mem:
            with mem.open() as src:
                profile = src.profile.copy()
        profile.update(count=3, dtype='float32', nodata=None, compress='deflate')
        with rasterio.open(folder/'restored.tif', 'w', **profile) as dst:
            dst.write(restored)
            dst.write_mask(a['usable'].astype('uint8')*255)
            dst.update_tags(description='Normalized RGB model estimate, not measured reflectance',
                            target_date=acquisition, reference_date=reference_date or 'none')
        report = {'id': ident, 'state': state, 'bbox': bbox, 'acquisition': acquisition,
                  'reference_date': reference_date,
                  'reference_age_days': days(acquisition, reference_date) if reference_date else None,
                  'model': 'HSFormer + bounded temporal fusion', 'collection': 'sentinel-2-l2a',
                  'width': x.shape[2], 'height': x.shape[1], 'seconds': round(time.perf_counter()-started, 2),
                  'cloud_flagged_fraction': float(mask.mean()), 'edited_fraction': float(edit.mean()),
                  'unprocessed_cloud_fraction': float((mask & ~edit).mean()),
                  'cloud_reference_coverage': float(edit.sum()/mask.sum()) if mask.any() else None,
                  'unchanged_outside_edit_mask': bool(np.array_equal(restored[:, ~edit], x[:, ~edit])),
                  'mean_mixture_weights': dict(zip(['hsformer', 'reference', 'learned_rgb'],
                      [float(w[edit].mean()) for w in weights])) if weights is not None else None,
                  'reference_checks': attempts, 'reference_catalog_truncated': truncated,
                  'multi_reference': multi_report,
                  'provenance': self.provenance, 'same_date_truth_available': False,
                  'display': 'Shared fixed gamma 2.2; input RGB clipped to [0,1]', 'notice': NOTICE}
        (folder/'metadata.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return report, folder
