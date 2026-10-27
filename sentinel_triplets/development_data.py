"""Separate CPU development collection; old independent audit stays excluded."""
import hashlib
import json
from pathlib import Path
from clear_sky.service import ClearClient
from temporal_prior.prepare import bbox_for
from .data import digest
from .independent_data import select,parse_labels,FUSION_SHA,WINDOWS

AREAS=[('ahmedabad',23.02,72.57),('nagpur',21.15,79.09),('madrid',40.42,-3.70),
       ('dallas',32.78,-96.80),('melbourne',-37.81,144.96),('casablanca',33.57,-7.59),
       ('coimbatore',11.02,76.96),('perth',-31.95,115.86)]
SPLITS={name:('development_validation' if name in ('coimbatore','perth') else 'development_train')
        for name,lat,lon in AREAS}

def prepare(root,client=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);client=client or ClearClient()
    config={'schema':'temporal_development_v1','areas':AREAS,'windows':WINDOWS,'splits':SPLITS,
            'max_observations_per_area_window':12,'max_total_image_requests':192,
            'reference_coverage_minimum':.50,'fusion_checkpoint_sha256':FUSION_SHA,
            'selection_uses_reference_label_rgb_agreement':False,
            'policy':'Development only: location-separated training and validation. No independent-audit locations. No training in this collector.'}
    path=root/'config.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(config)):
        raise ValueError('Different protocol: use a fresh folder')
    path.write_text(json.dumps(config,indent=2))
    records=[];reports=[]
    def save():
        groups={r['group'] for r in records}
        manifest={'schema':'temporal_development_v1','records':records,'locations':sorted(groups),
                  'ready_for_training_design':all(sum(r['split']==split for r in records)>=minimum for split,minimum in [('development_train',16),('development_validation',4)]) and len({r['group'] for r in records if r['split']=='development_train'})>=4 and len({r['group'] for r in records if r['split']=='development_validation'})==2,
                  'confirmed_change_labels_available':False,'config_sha256':digest(path),
                  'notice':'Different-date proxy labels. Provider classes are not verified thickness or shadow truth. No stable-pixel selection filter. Only development_train may be used for optimization; keep development_validation out of optimization.'}
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
                found=select(scenes,group,window,root)
                for record in found:record['split']=SPLITS[group]
                records.extend(found)
                reports.append({'group':group,'window':window,'observations':len(scenes),'patches':len(found),
                                'catalog_truncated':catalog.get('truncated',False)})
                save()
            except Exception as exc:
                reports.append({'group':group,'window':window,'error_type':type(exc).__name__});save();raise
    return save()
