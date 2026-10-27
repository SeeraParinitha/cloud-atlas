"""Leakage screening, provenance and paired statistical comparisons."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .data import validate_records,read_mask

def audit_records(records,output):
    validate_records(records)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fingerprints=[];detail=[]
    for r in records:
        with Image.open(r['clear']) as im:
            arr=np.asarray(im.convert('L').resize((9,8)),dtype=np.float32)
            fingerprints.append((arr[:,1:]>arr[:,:-1]).reshape(-1))
        row={k:r.get(k) for k in ['id','dataset','group','split']}
        if r.get('mask'):row['mask_fraction']=float(read_mask(r['mask']).mean())
        detail.append(row)
    fingerprints=np.asarray(fingerprints);suspects=[]
    for i in range(len(records)):
        distances=(fingerprints[i+1:]!=fingerprints[i]).sum(1)
        for j in np.where(distances<=4)[0]+i+1:
            if records[i]['split']!=records[j]['split']:
                suspects.append(dict(id_a=records[i]['id'],id_b=records[j]['id'],hamming=int((fingerprints[i]!=fingerprints[j]).sum())))
    pd.DataFrame(suspects,columns=['id_a','id_b','hamming']).to_csv(output/'near_duplicate_candidates.csv',index=False)
    pd.DataFrame(detail).to_csv(output/'sample_audit.csv',index=False)
    return dict(samples=len(records),near_duplicate_candidates=len(suspects),
                note='Perceptual hashes are screening only. No candidate is automatically removed; geography cannot be inferred from image similarity.')

def paired_bootstrap(a,b,metric,replicates=2000,seed=2026):
    """Paired group bootstrap of image-macro differences; seed aggregation happens before this call."""
    a=pd.DataFrame(a);b=pd.DataFrame(b)
    m=a[['id','group',metric]].merge(b[['id',metric]],on='id',suffixes=('_a','_b'),validate='one_to_one')
    if len(m)!=len(a) or len(m)!=len(b):raise ValueError('Comparisons require identical sample IDs')
    x=pd.to_numeric(m[metric+'_a'],errors='coerce')-pd.to_numeric(m[metric+'_b'],errors='coerce')
    m=m.assign(delta=x);m=m[np.isfinite(m.delta)]
    groups=[g.delta.to_numpy() for _,g in m.groupby('group')]
    if len(groups)<2:return dict(delta=None,low=None,high=None,n=len(m),groups=len(groups))
    rng=np.random.default_rng(seed);values=[]
    for _ in range(replicates):
        draw=rng.integers(0,len(groups),len(groups));values.append(np.concatenate([groups[i] for i in draw]).mean())
    return dict(delta=float(m.delta.mean()),low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)),n=len(m),groups=len(groups))