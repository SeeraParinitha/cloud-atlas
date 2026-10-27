"""Reproducible RGB subsets from official WHUS2-CRv train/validation archives."""
import hashlib
import json
import random
import time
import zlib
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import PseudoConfig, pseudo_labels
from .engine import save_json
from .remote_validation import TrainingZip, ValidationZip
from .sentinel_validation import read_pair, InvalidPairError


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def grouped_members(archive, prefix):
    groups = defaultdict(list)
    for name in sorted(archive.members):
        if name.startswith(prefix+'/clearDNclips/10m/') and name.endswith('.tif'):
            if name.replace('/clearDNclips/', '/cloudDNclips/') not in archive.members:
                raise ValueError('Unpaired archive member: '+name)
            groups[name.split('/')[-2]].append(name)
    if not groups:
        raise ValueError('No expected 10 m paired patches in official archive')
    return groups


def choose_members(train_groups, val_groups, train_count, val_count, seed=123):
    # Hold out a fixed subset of validation scene folders before inspecting pixels.
    rng = random.Random(seed)
    groups = sorted(val_groups)
    rng.shuffle(groups)
    n = len(groups) if not set(train_groups) & set(val_groups) else min(8, max(1, len(groups)//4))
    held_out = set(groups[:n])
    training = {g:v for g,v in train_groups.items() if g not in held_out}
    validation = {g:v for g,v in val_groups.items() if g in held_out}

    def sample(pool, count):
        names = sorted(pool)
        rng.shuffle(names)
        pool = {g:list(pool[g]) for g in names}
        for items in pool.values():
            rng.shuffle(items)
        selected = []
        while len(selected) < count:
            before = len(selected)
            for g in names:
                if pool[g] and len(selected) < count:
                    selected.append(pool[g].pop())
            if len(selected) == before:
                raise ValueError('Insufficient patches after scene holdout; do not relax the split silently')
        return selected
    return sample(training, train_count), sample(validation, val_count), sorted(held_out)


def validate_manifest(root, protocol):
    seen = {}
    groups = defaultdict(set)
    for row in protocol['records']:
        groups[row['split']].add(row['group'])
        for key in ('cloudy', 'clear'):
            path = Path(root)/row[key]
            if digest(path) != row[key+'_sha256']:
                raise ValueError('Changed dataset bytes: '+str(path))
            previous = seen.get(row[key+'_pixels_sha256'])
            if previous is not None and previous != row['split']:
                raise ValueError('Exact RGB content leaks across training/validation')
            seen[row[key+'_pixels_sha256']] = row['split']
    if groups['train'] & groups['validation']:
        raise ValueError('Scene group leaks across training/validation')


def prepare(root, train_count=512, val_count=64, seed=123, deadline=float('inf'), archives=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    spec = dict(train_count=train_count, val_count=val_count, seed=seed, version=2)
    if min(train_count, val_count) < 1:
        raise ValueError('Nonempty training and validation samples required')
    manifest = root/'manifest.json'
    if manifest.exists():
        protocol = json.loads(manifest.read_text())
        if protocol['sampling_config'] != spec:
            raise ValueError('Dataset configuration changed; use a new experiment folder')
        validate_manifest(root, protocol)
        return protocol
    train_archive, val_archive = archives or (TrainingZip(), ValidationZip())
    train_groups, val_groups = grouped_members(train_archive, 'train'), grouped_members(val_archive, 'val')
    train, val, held_out = choose_members(train_groups, val_groups, train_count, val_count, seed)
    tasks = [('train', n, train_archive) for n in train] + [('validation', n, val_archive) for n in val]

    cancelled = threading.Event()
    failures = []

    def fetch_one(task):
        split, clear, archive = task
        row = dict(id=clear, group=clear.split('/')[-2], split=split)
        for key, name in [('clear', clear), ('cloudy', clear.replace('/clearDNclips/', '/cloudDNclips/'))]:
            if cancelled.is_set():
                raise RuntimeError('Download batch stopped after: '+failures[0])
            if time.monotonic() >= deadline:
                raise TimeoutError('Data download reached the session time budget')
            relative = f'{split}/{hashlib.sha256(name.encode()).hexdigest()[:24]}_{key}.tif'
            path = root/relative
            data = path.read_bytes() if path.exists() else b''
            info = archive.members[name]
            if len(data) != info['unpacked'] or zlib.crc32(data) != info['crc']:
                data = archive.read(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix('.part')
                temp.write_bytes(data); temp.replace(path)
            row[key], row[key+'_sha256'], row[key+'_member'] = relative, hashlib.sha256(data).hexdigest(), name
        try:
            cloudy, clear_tensor = read_pair(root, row)
            if min(cloudy.shape[-2:]) < 256:
                raise InvalidPairError('Patch is too small for native 256-pixel training crops')
        except InvalidPairError as exc:
            return {**row, 'rejection_reason': str(exc)}
        for key, value in [('cloudy', cloudy), ('clear', clear_tensor)]:
            row[key+'_pixels_sha256'] = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        return row

    def fetch(task):
        try:
            return fetch_one(task)
        except Exception as exc:
            failures.append(f'{type(exc).__name__}: {exc}')
            cancelled.set()
            raise

    records, rejected = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for index, row in enumerate(pool.map(fetch, tasks)):
            if 'rejection_reason' in row:
                rejected.append(row)
                print('Excluded pair:',row['id'],'—',row['rejection_reason'],flush=True)
                save_json(root/'rejected_pairs.json',rejected)
            else:
                records.append(row)
            if (index+1) % 16 == 0 or index+1 == len(tasks):
                print(f'Checked Sentinel-2 pairs {index+1}/{len(tasks)}; accepted {len(records)}, excluded {len(rejected)}', flush=True)
    # Replacements are deterministic, in the SAME scene and official split. They
    # depend only on the seed, member name and data validity, never model scores.
    used = set(train+val)
    original_rejections = list(rejected)
    for bad in original_rejections:
        split, group = bad['split'], bad['group']
        archive = train_archive if split == 'train' else val_archive
        groups = train_groups if split == 'train' else val_groups
        candidates = sorted((n for n in groups[group] if n not in used),
            key=lambda n: hashlib.sha256(f'{seed}:{n}'.encode()).hexdigest())
        for name in candidates:
            used.add(name)
            replacement = fetch((split,name,archive))
            if 'rejection_reason' in replacement:
                rejected.append(replacement)
                save_json(root/'rejected_pairs.json',rejected)
                continue
            replacement['replaces'] = bad['id']
            records.append(replacement)
            print('Accepted same-scene replacement for:',bad['id'],flush=True)
            break
        else:
            raise ValueError('No valid same-scene replacement available for '+bad['id'])
    if sum(r['split']=='train' for r in records)!=train_count or sum(r['split']=='validation' for r in records)!=val_count:
        raise ValueError('Prepared pair counts do not match requested experiment')
    protocol = dict(sampling_config=spec, dataset='WHUS2-CRv', records=records,
        rejected_pairs=rejected,quality_policy='Reject nonfinite, masked, all-zero RGB pixels or mismatched/small patches; deterministic same-scene/split replacement. This is conservative, not proof zero pixels are nodata.',
        held_out_scene_groups=held_out, train_source='https://zenodo.org/records/8035347',
        validation_source='https://zenodo.org/records/8035349', test_accessed=False,
        preprocessing=dict(bands=['B04','B03','B02'], rgb_indices_one_based=[3,2,1],
            dn_divisor=10000, reflectance_ceiling=1.0, note='Fixed before fine-tuning; no per-image stretching'),
        limitations=['RGB subset, not published full multispectral benchmark',
            'Scene-folder separation and exact RGB duplicate checks, not independent geographic/near-duplicate audit',
            'Official pairs lack georeferencing; subpixel registration not independently verified',
            'Different-date references may contain land changes; pseudo labels are not cloud ground truth'])
    validate_manifest(root, protocol)
    save_json(manifest, protocol)
    return protocol


class SentinelPairs(Dataset):
    def __init__(self, root, protocol, split, pseudo=None, augment=False):
        self.root = Path(root)
        self.rows = [r for r in protocol['records'] if r['split'] == split]
        self.pseudo = pseudo or PseudoConfig()
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        cloudy, clear = (a.clamp(0,1) for a in read_pair(self.root, row))
        if self.augment:
            h, w = cloudy.shape[-2:]
            if min(h,w) < 256:
                raise ValueError('Training patches must be at least 256 pixels')
            y, x = random.randrange(h-255), random.randrange(w-255)
            cloudy, clear = [a[:,y:y+256,x:x+256] for a in (cloudy,clear)]
            turns = random.randrange(4)
            cloudy, clear = [torch.rot90(a, turns, (-2,-1)) for a in (cloudy,clear)]
            if random.random() < .5:
                cloudy, clear = cloudy.flip(-1), clear.flip(-1)
            # Explicit identity examples teach both models to leave clear input alone.
            if random.random() < .1:
                cloudy = clear.clone()
        severity, mask = pseudo_labels(cloudy, clear, self.pseudo)
        return dict(cloudy=cloudy, clear=clear, severity=severity, mask=mask,
            stable_clear=(1-mask)*((cloudy-clear).abs().mean(0,keepdim=True)<=self.pseudo.noise_floor),
            mask_source='pseudo', id=row['id'])


def calibrate(root, protocol):
    """Training-only robust difference scale; not measured cloud opacity."""
    samples = []
    for row in protocol['records']:
        if row['split'] == 'train':
            x, y = read_pair(root, row)
            samples.append((x.clamp(0,1)-y.clamp(0,1)).abs().mean(0)[::8,::8].numpy().ravel())
    scale = max(.01, float(np.quantile(np.concatenate(samples), .95))-.005)
    return PseudoConfig(noise_floor=.005, scale=scale, threshold=.05, sigma=1.0)
