"""CPU-only, cached and bounded acquisition of three independent Sentinel dates."""
import hashlib
import itertools
import json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
from rasterio.io import MemoryFile
from clear_sky.service import ClearClient
from temporal_prior.prepare import bbox_for
from temporal_prior.real_review import read_scene
from .data import days,digest,counts,ready

AREAS=[('pune',18.52,73.86,'train'),('jaipur',26.91,75.79,'train'),
       ('sacramento',38.58,-121.49,'train'),('seville',37.39,-5.98,'train'),
       ('nairobi',-1.29,36.82,'train'),('phoenix',33.45,-112.07,'train'),
       ('sydney',-33.87,151.21,'train'),('brasilia',-15.79,-47.88,'train'),
       ('adelaide',-34.93,138.60,'validation'),('fresno',36.74,-119.78,'validation'),
       ('sao_paulo',-23.55,-46.63,'test'),('bologna',44.49,11.34,'test')]


def parse(raw,stamp):
    s=read_scene(raw)
    with MemoryFile(raw) as mem:
        with mem.open() as ds:
            a=ds.read(); s['high_cloud']=((a[3]==9)|(a[4]>=80))&s['cloud']
    return dict(s,date=stamp,source_sha256=hashlib.sha256(raw).hexdigest())


def select(scenes,group,split,root,limit=12):
    """Masks/dates choose data before any model is run. Label is never an input."""
    records=[];used=set()
    for c in sorted(scenes,key=lambda s:s['date'],reverse=True):
        references=sorted((s for s in scenes if s['date'][:10]!=c['date'][:10] and 0<days(c['date'],s['date'])<=60),
                          key=lambda s:days(c['date'],s['date']))
        labels=sorted((s for s in scenes if s['date'][:10]!=c['date'][:10] and abs(days(s['date'],c['date']))<=21),
                      key=lambda s:abs(days(s['date'],c['date'])))
        per_cloud=0
        for r,l in itertools.product(references,labels):
            if r['date'][:10]==l['date'][:10] or c['grid']!=r['grid'] or c['grid']!=l['grid']: continue
            h,w=c['cloud'].shape
            for y in range(0,h-255,128):
                for x in range(0,w-255,128):
                    key=(c['date'][:10],y//256,x//256)
                    if key in used: continue
                    sl=np.s_[y:y+256,x:x+256]
                    valid=c['usable'][sl]&~r['risk'][sl]
                    eval_valid=valid&~l['risk'][sl]
                    mask=c['cloud'][sl]&valid
                    if valid.mean()<.98 or eval_valid.mean()<.98 or mask.mean()<.10: continue
                    cloud_rgb=np.nan_to_num(c['rgb'][:,sl[0],sl[1]],nan=0,posinf=1,neginf=0).clip(0,1)
                    ref=np.nan_to_num(r['rgb'][:,sl[0],sl[1]],nan=0,posinf=1,neginf=0).clip(0,1)
                    label=np.nan_to_num(l['rgb'][:,sl[0],sl[1]],nan=0,posinf=1,neginf=0).clip(0,1)
                    # Heuristic temporal-change filter only affects supervision.
                    # Evaluation also reports all usable pixels without this filter.
                    stable=(np.abs(ref-label).mean(0)<=.08)
                    loss_valid=eval_valid&stable
                    if (loss_valid&mask).mean()<.05 or (loss_valid&mask).sum()<.5*mask.sum(): continue
                    if np.array_equal(ref,label): continue
                    ident=f'{group}-{len(records):03d}'; p=Path(root)/'patches'/(ident+'.npz')
                    p.parent.mkdir(parents=True,exist_ok=True)
                    values={'cloudy':cloud_rgb*valid,'reference':ref*valid,'label':label*eval_valid,
                            'valid':valid[None], 'eval_valid':eval_valid[None], 'loss_valid':loss_valid[None],
                            'mask':mask[None], 'high_cloud':(c['high_cloud'][sl]&mask)[None]}
                    np.savez_compressed(p,**values)
                    records.append({'id':ident,'group':group,'split':split,'path':p.relative_to(root).as_posix(),
                                    'sha256':digest(p),'cloudy_date':c['date'],'reference_date':r['date'],'label_date':l['date'],
                                    'source_grid':c['grid'],'source_window':[y,x,256,256],
                                    'source_sha256':{k:s['source_sha256'] for k,s in [('cloudy',c),('reference',r),('label',l)]},
                                    'provider_cloud_shadow_fraction':float(mask.mean()),
                                    'label_is_different_date_proxy':True})
                    used.add(key); per_cloud+=1
                    if len(records)>=limit: return records
                    if per_cloud>=2: break
                if per_cloud>=2: break
            if per_cloud>=2: break
    return records


def prepare(root,client=None,end=None):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    client=client or ClearClient(); cfgpath=root/'config.json'
    if cfgpath.exists():
        cfg=json.loads(cfgpath.read_text())
        if cfg['version']!=1: raise ValueError('Use a new preparation folder')
    else:
        end=end or (datetime.now(timezone.utc).date()-timedelta(days=1)).isoformat()
        cfg={'version':1,'start':(datetime.fromisoformat(end)-timedelta(days=90)).date().isoformat(),
             'end':end,'areas':AREAS,'candidate_limit':24,'collection':'sentinel-2-l2a',
             'max_reference_gap_days':60,'max_proxy_label_gap_days':21}
        cfgpath.write_text(json.dumps(cfg,indent=2))
    rows=[];report=[]
    def save():
        protocol={'schema':'real_sentinel_triplets_v1','ready':ready(rows),'counts':counts(rows),
                  'records':rows,'config_sha256':digest(cfgpath),'notice':'Real clouds; separate-date approximate labels, not same-date ground truth'}
        (root/'manifest.json').write_text(json.dumps(protocol,indent=2))
        (root/'preparation_report.json').write_text(json.dumps(report,indent=2))
        return protocol
    save()
    for group,lat,lon,split in cfg['areas']:
        folder=root/'cache'/group;folder.mkdir(parents=True,exist_ok=True)
        try:
            path=folder/'catalog.json';bbox=bbox_for(lat,lon)
            if not path.exists(): path.write_text(json.dumps(client.search(bbox,cfg['start'],cfg['end'])))
            catalog=json.loads(path.read_text())
            ordered=sorted(catalog['scenes'],key=lambda s:(s.get('cloud_cover') if s.get('cloud_cover') is not None else 50,s['datetime']))
            # Favor enough clear dates (16) and cloudy dates (8); pixels decide acceptance.
            chosen={s['datetime']:s for s in ordered[:16]+ordered[-8:]}
            scenes=[]
            for stamp in sorted(chosen):
                key=hashlib.sha256(stamp.encode()).hexdigest()[:16];p=folder/(key+'.tif')
                if not p.exists():
                    raw=client.fetch(bbox,stamp);parse(raw,stamp)
                    temp=p.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(p)
                scenes.append(parse(p.read_bytes(),stamp))
                print(group,'cached',len(scenes),'/',len(chosen),flush=True)
            found=select(scenes,group,split,root);rows.extend(found)
            report.append({'group':group,'split':split,'cached_dates':len(scenes),'triplets':len(found),
                           'catalog_truncated':catalog.get('truncated',False)})
        except Exception as exc:
            # Preserve prior progress, but never pretend an API/storage failure succeeded.
            report.append({'group':group,'split':split,'error_type':type(exc).__name__})
            save();raise
        save();print('Location complete:',report[-1],flush=True)
    protocol=save()
    if protocol['ready']:
        from .data import validate
        try:validate(root)
        except Exception:
            protocol['ready']=False
            protocol['validation_failed']=True
            (root/'manifest.json').write_text(json.dumps(protocol,indent=2))
            raise
    return protocol
