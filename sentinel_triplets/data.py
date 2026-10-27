import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def days(a, b):
    return (datetime.fromisoformat(a.replace('Z','+00:00'))-
            datetime.fromisoformat(b.replace('Z','+00:00'))).total_seconds()/86400


def counts(records):
    return {s: {'locations': len({r['group'] for r in records if r['split']==s}),
                'triplets': sum(r['split']==s for r in records)} for s in ('train','validation','test')}


def ready(records):
    n=counts(records)
    return (n['train']['locations']>=4 and n['train']['triplets']>=16 and
            all(n[s]['locations']>=2 and n[s]['triplets']>=4 for s in ('validation','test')))


def validate(root):
    root=Path(root).resolve(); protocol=json.loads((root/'manifest.json').read_text())
    if protocol.get('schema')!='real_sentinel_triplets_v1': raise ValueError('Not the real-triplet dataset')
    if not protocol.get('ready') or not ready(protocol['records']):
        raise ValueError('Preparation not ready: inspect preparation_report.json on CPU, do not start GPU training')
    groups={}; ids=set(); pixels={}
    for r in protocol['records']:
        if r['split'] not in ('train','validation','test'): raise ValueError('Invalid split')
        if r['id'] in ids: raise ValueError('Duplicate record')
        ids.add(r['id'])
        if groups.setdefault(r['group'],r['split'])!=r['split']: raise ValueError('Location leakage')
        stamps=[r[k+'_date'] for k in ('cloudy','reference','label')]
        if len({s[:10] for s in stamps})!=3: raise ValueError('Three independent dates required')
        if not 0<days(stamps[0],stamps[1])<=60 or abs(days(stamps[2],stamps[0]))>21:
            raise ValueError('Temporal gap outside fixed policy')
        p=(root/r['path']).resolve()
        if root not in p.parents or digest(p)!=r['sha256']: raise ValueError('Unsafe or changed data')
        with np.load(p,allow_pickle=False) as a:
            for k in ('cloudy','reference','label'):
                v=a[k]
                if v.shape!=(3,256,256) or not np.isfinite(v).all() or v.min()<0 or v.max()>1:
                    raise ValueError('Invalid RGB')
                fingerprint=hashlib.sha256(v.tobytes()).hexdigest()
                if pixels.setdefault(fingerprint,r['split'])!=r['split']: raise ValueError('Exact pixel leakage')
            for k in ('valid','eval_valid','loss_valid','mask','high_cloud'):
                if a[k].shape!=(1,256,256) or not np.isin(a[k],[0,1]).all(): raise ValueError('Invalid mask')
            if (a['loss_valid']>a['valid']).any() or (a['mask']>a['valid']).any(): raise ValueError('Mask containment')
            if (a['loss_valid']>a['eval_valid']).any() or (a['eval_valid']>a['valid']).any(): raise ValueError('Label validity containment')
            if (a['high_cloud']>a['mask']).any(): raise ValueError('Invalid high cloud mask')
            if (a['mask']*a['loss_valid']).mean()<.05: raise ValueError('Insufficient supervised cloud pixels')
    return protocol


def load(root, record):
    with np.load(Path(root)/record['path'],allow_pickle=False) as a:
        b={k:torch.tensor(a[k],dtype=torch.float32) for k in a.files}
    b['days']=torch.tensor(days(record['cloudy_date'],record['reference_date']),dtype=torch.float32)
    return b
