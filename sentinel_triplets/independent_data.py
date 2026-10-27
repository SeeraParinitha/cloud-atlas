"""Prospective collection protocol for a frozen-model audit; never training data."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from rasterio.io import MemoryFile
from clear_sky.service import ClearClient
from temporal_prior.prepare import bbox_for
from .prepare import parse
from .data import days, digest

AREAS=[('kolkata',22.57,88.36),('dhaka',23.81,90.41),('bangkok',13.76,100.50),
       ('rotterdam',51.92,4.48),('houston',29.76,-95.37),('cape_town',-33.92,18.42)]
WINDOWS=[('early_2026','2026-01-01','2026-03-31'),('late_2026','2026-07-01','2026-09-28')]
FUSION_SHA='5c0a93924e71db114243cc1854713a91d45eb5f3629cfef6d690ab38cc4caf08'


def parse_labels(raw,stamp):
    scene=parse(raw,stamp)
    with MemoryFile(raw) as mem:
        with mem.open() as src:
            a=src.read();scene['scl']=np.nan_to_num(a[3],nan=0).astype('uint8')
            scene['cld']=np.nan_to_num(a[4],nan=100).astype('float32')
    return scene


def select(scenes,group,window,root):
    """Select without reference-label RGB agreement or model predictions."""
    records=[];used_windows=set();targets=set()
    for c in sorted(scenes,key=lambda s:s['date'],reverse=True):
        if len(targets)>=2:break
        refs=sorted((s for s in scenes if s['date'][:10]<c['date'][:10] and 0<days(c['date'],s['date'])<=60),key=lambda s:days(c['date'],s['date']))
        labels=sorted((s for s in scenes if s['date'][:10]!=c['date'][:10] and abs(days(s['date'],c['date']))<=21),key=lambda s:abs(days(s['date'],c['date'])))
        count=0
        for r,l in itertools.product(refs,labels):
            if r['date'][:10]==l['date'][:10] or c['grid']!=r['grid'] or c['grid']!=l['grid']:continue
            h,w=c['cloud'].shape
            for y in range(0,h-255,256):
                for x in range(0,w-255,256):
                    if (y,x) in used_windows:continue
                    sl=np.s_[y:y+256,x:x+256]
                    target_valid=c['usable'][sl];reference_valid=~r['risk'][sl]
                    evaluation_valid=target_valid&~l['risk'][sl]
                    mask=c['cloud'][sl]&target_valid
                    if target_valid.mean()<.98 or evaluation_valid.mean()<.98 or mask.mean()<.10:continue
                    coverage=float((mask&reference_valid).sum()/max(1,mask.sum()))
                    if coverage<.50:continue
                    values={k:np.nan_to_num(s['rgb'][:,sl[0],sl[1]],nan=0,posinf=1,neginf=0).clip(0,1)
                            for k,s in [('cloudy',c),('reference',r),('label',l)]}
                    values.update(target_valid=target_valid[None],reference_valid=reference_valid[None],
                                  eval_valid=evaluation_valid[None],mask=mask[None],
                                  high_cloud=(c['high_cloud'][sl]&mask)[None],
                                  shadow_provider=(c['scl'][sl]==3)[None],
                                  target_scl=c['scl'][sl],target_cloud_probability=c['cld'][sl])
                    ident=f'{group}-{window}-{len(records):02d}';p=Path(root)/'patches'/(ident+'.npz')
                    p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**values)
                    records.append({'id':ident,'group':group,'window':window,'split':'audit_only',
                                    'acquisition_group':group+'/'+c['date'][:10],
                                    'path':p.relative_to(root).as_posix(),'sha256':digest(p),
                                    'cloudy_date':c['date'],'reference_date':r['date'],'label_date':l['date'],
                                    'reference_coverage':coverage,'source_window':[y,x,256,256],
                                    'source_grid':c['grid'],'source_sha256':{k:s['source_sha256'] for k,s in [('cloudy',c),('reference',r),('label',l)]}})
                    used_windows.add((y,x));targets.add(c['date'][:10]);count+=1
                    if count>=2:break
                if count>=2:break
            if count>=2:break
    return records


def prepare(root,client=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);client=client or ClearClient()
    config={'schema':'independent_temporal_audit_v1','areas':AREAS,'windows':WINDOWS,
            'max_observations_per_area_window':12,'max_total_image_requests':144,
            'reference_coverage_minimum':.50,'fusion_checkpoint_sha256':FUSION_SHA,
            'selection_uses_reference_label_rgb_agreement':False,
            'policy':'Audit only. Fixed protocol before outputs. No training or checkpoint selection.'}
    path=root/'config.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(config)):
        raise ValueError('Different protocol: use a fresh folder')
    path.write_text(json.dumps(config,indent=2))
    records=[];reports=[]
    def save():
        groups={r['group'] for r in records}
        manifest={'schema':'independent_temporal_audit_v1','records':records,'locations':sorted(groups),
                  'ready_for_descriptive_audit':len(groups)>=4 and len(records)>=16,
                  'confirmed_change_labels_available':False,'config_sha256':digest(path),
                  'notice':'Different-date proxy labels. Provider classes are not verified thickness or shadow truth. No stable-pixel selection filter. Never use this set for training.'}
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
        (root/'collection_report.json').write_text(json.dumps(reports,indent=2))
        return manifest
    save()
    for group,lat,lon in AREAS:
        for window,start,end in WINDOWS:
            folder=root/'cache'/group/window;folder.mkdir(parents=True,exist_ok=True)
            try:
                catfile=folder/'catalog.json';bbox=bbox_for(lat,lon)
                if not catfile.exists():catfile.write_text(json.dumps(client.search(bbox,start,end)))
                catalog=json.loads(catfile.read_text());unique={}
                for s in sorted(catalog['scenes'],key=lambda s:s['datetime']):unique.setdefault(s['datetime'][:10],s)
                ordered=sorted(unique.values(),key=lambda s:(s.get('cloud_cover') if s.get('cloud_cover') is not None else 50,s['datetime']))
                chosen={s['datetime']:s for s in ordered[:8]+ordered[-4:]};scenes=[]
                for stamp in sorted(chosen):
                    key=hashlib.sha256(stamp.encode()).hexdigest()[:16];p=folder/(key+'.tif')
                    if not p.exists():
                        raw=client.fetch(bbox,stamp);parse_labels(raw,stamp)
                        temp=p.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(p)
                    scenes.append(parse_labels(p.read_bytes(),stamp))
                    print(group,window,'cached',len(scenes),'/',len(chosen),flush=True)
                found=select(scenes,group,window,root);records.extend(found)
                reports.append({'group':group,'window':window,'observations':len(scenes),'patches':len(found),
                                'catalog_truncated':catalog.get('truncated',False)})
                save()
            except Exception as exc:
                reports.append({'group':group,'window':window,'error_type':type(exc).__name__});save();raise
    return save()
