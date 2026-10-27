"""Frozen full-area qualitative review. No training and no automatic deployment."""
import hashlib,json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from rasterio.io import MemoryFile
from cloud_severity.engine import load_model
from clear_sky.service import ClearClient
from temporal_prior.prepare import bbox_for
from temporal_prior.real_review import read_scene
from temporal_prior.train import PARENT_SHA
from .data import digest,days
from .model import BoundedFusion


def starts(length,offset=0):
    if length<=256:return [0]
    return sorted({0,length-256,*range(192-offset,length-256,192)})


def tiled(backbone,fusion,target,reference,mask,valid,gap,device='cpu',offset=0,progress=None):
    """Hann-weight overlapping predictions; unsupported/unmasked output stays original."""
    if not 0<gap<=60:raise ValueError('Reference must precede target by at most 60 days')
    if offset not in (0,96):raise ValueError('Unsupported diagnostic grid offset')
    if target.shape!=reference.shape or target.shape[0]!=3 or mask.shape!=target.shape[1:] or valid.shape!=mask.shape:
        raise ValueError('Image/mask shapes differ')
    if not np.isfinite(target).all() or not np.isfinite(reference).all():raise ValueError('Nonfinite input')
    h,w=mask.shape;pad=((0,0),(0,max(0,256-h)),(0,max(0,256-w)))
    x=np.pad(target*valid,pad,mode='edge');r=np.pad(reference*valid,pad,mode='edge')
    v=np.pad(valid,pad[1:],mode='edge');m=np.pad(mask&valid,pad[1:],mode='edge')
    sums=np.zeros((3,)+v.shape,dtype='float64');single=np.zeros_like(sums);mix=np.zeros_like(sums)
    total=np.zeros(v.shape,dtype='float64')
    window=np.maximum(np.outer(np.hanning(256),np.hanning(256)),1e-3).astype('float32')
    captured={};handle=None
    if hasattr(fusion,'head'):
        handle=fusion.head.register_forward_hook(lambda mod,args,out:captured.update(weights=out[:,:3].softmax(1).detach()))
    def t(a):return torch.as_tensor(a,dtype=torch.float32,device=device)[None]
    count=0;tile_count=len(starts(v.shape[0],offset))*len(starts(v.shape[1],offset))
    try:
        with torch.no_grad():
            for y in starts(v.shape[0],offset):
                for xx in starts(v.shape[1],offset):
                    sl=np.s_[y:y+256,xx:xx+256]
                    tv=t(v[sl][None]);tm=t(m[sl][None])
                    tx=t(x[:,sl[0],sl[1]]);tr=t(r[:,sl[0],sl[1]])
                    base=backbone(tx)['restored']*tv
                    pred=fusion(tx,base,tr,tm,torch.tensor([gap],device=device),tv)
                    if not torch.isfinite(pred).all():raise ValueError('Nonfinite model prediction')
                    sums[:,sl[0],sl[1]]+=pred[0].cpu().numpy()*window
                    single[:,sl[0],sl[1]]+=base[0].cpu().numpy()*window
                    if 'weights' in captured:mix[:,sl[0],sl[1]]+=captured['weights'][0].cpu().numpy()*window
                    total[sl]+=window
                    count+=1
                    if progress is not None and (count==1 or count%4==0 or count==tile_count):progress(count,tile_count)
    finally:
        if handle is not None:handle.remove()
    if (total[:h,:w]<=0).any():raise ValueError('Tiling left uncovered pixels')
    estimate=(sums/total)[:,:h,:w].astype('float32')
    single=(single/total)[:,:h,:w].astype('float32')
    edit=mask&valid
    restored=np.where(edit[None],estimate,target)
    single=np.where(valid[None],single,target)
    weights=(mix/total)[:,:h,:w].astype('float32') if captured else None
    return restored,single,weights


def png(rgb,path):
    a=np.nan_to_num(rgb,nan=0).clip(0,1)**(1/2.2)
    Image.fromarray((a.transpose(1,2,0)*255).round().astype('uint8')).save(path)


def review_pair(backbone,fusion,target_raw,reference_raw,target_date,reference_date,folder,device='cpu'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    a,b=read_scene(target_raw),read_scene(reference_raw)
    if a['grid']!=b['grid']:raise ValueError('Source grids differ')
    gap=days(target_date,reference_date)
    valid=a['usable']&~b['risk'];mask=a['cloud']&a['usable']
    edit=mask&valid
    x=np.nan_to_num(a['rgb'],nan=0,posinf=1,neginf=0).clip(0,1)
    r=np.nan_to_num(b['rgb'],nan=0,posinf=1,neginf=0).clip(0,1)
    restored,single,weights=tiled(backbone,fusion,x,r,mask,valid,gap,device,
                                 progress=lambda done,total:print(f'Main grid: {done}/{total} tiles',flush=True))
    shifted,_,_=tiled(backbone,fusion,x,r,mask,valid,gap,device,offset=96,
                      progress=lambda done,total:print(f'Shifted grid: {done}/{total} tiles',flush=True))
    diff=np.abs(restored-shifted).mean(0)
    copy=np.where(edit[None],r,x)
    for name,image in [('original',x),('reference',r),('hsformer',single),('fusion',restored),('reference_copy',copy),('shifted_grid',shifted)]:png(image,folder/(name+'.png'))
    for name,image in [('edit_mask',edit),('unprocessed_cloud_mask',mask&~valid),('reference_risk_mask',b['risk'])]:
        Image.fromarray(image.astype('uint8')*255).save(folder/(name+'.png'))
    Image.fromarray((diff/.05).clip(0,1).__mul__(255).astype('uint8')).save(folder/'tile_shift_difference.png')
    if weights is not None:Image.fromarray((weights[1]*255).round().astype('uint8')).save(folder/'reference_mixture_weight.png')
    h,w=x.shape[1:];canvas=Image.new('RGB',(w*3,h+28),'white');draw=ImageDraw.Draw(canvas)
    for i,(name,label) in enumerate([('original','Selected cloudy observation'),('reference','Earlier reference'),('fusion','Learned bounded fusion estimate')]):
        canvas.paste(Image.open(folder/(name+'.png')),(i*w,28));draw.text((i*w+5,6),label,fill='black')
    canvas.save(folder/'comparison.png')
    (folder/'target_source.tif').write_bytes(target_raw);(folder/'reference_source.tif').write_bytes(reference_raw)
    with MemoryFile(target_raw) as mem:
        with mem.open() as src:profile=src.profile.copy()
    profile.update(count=3,dtype='float32',nodata=None)
    import rasterio
    with rasterio.open(folder/'fusion_rgb_estimate.tif','w',**profile) as dst:dst.write(restored)
    newblack=restored.max(0)<=1e-6
    report={'state':'qualitative_inference_complete','website_ready':False,'same_date_truth_available':False,
            'target_date':target_date,'reference_date':reference_date,'reference_age_days':gap,
            'provider_cloud_shadow_fraction':float(mask.mean()),'edited_fraction':float(edit.mean()),
            'cloud_reference_coverage':float(edit.sum()/max(1,mask.sum())),
            'unprocessed_cloud_fraction':float((mask&~valid).mean()),
            'new_black_edited_fraction':float((newblack&(x.max(0)>1e-6)&edit).sum()/max(1,edit.sum())),
            'tile_shift_mae_on_edited_pixels':float(diff[edit].mean()) if edit.any() else None,
            'tile_shift_p95_on_edited_pixels':float(np.quantile(diff[edit],.95)) if edit.any() else None,
            'unchanged_outside_edit_mask':bool(np.array_equal(restored[:,~edit],x[:,~edit])),
            'mean_mixture_weights_on_edited_pixels':dict(zip(['hsformer','reference','learned_rgb'],[float(v[edit].mean()) for v in weights])) if weights is not None and edit.any() else None,
            'notice':'No PSNR: no same-date clear truth. Mixture weights are not confidence. Tile-grid disagreement measures consistency, not accuracy. Missing-reference areas remain original. RGB TIFF is a normalized model estimate, not calibrated surface reflectance.'}
    (folder/'report.json').write_text(json.dumps(report,indent=2));return report


def run(parent,checkpoint,output,client=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);client=client or ClearClient()
    (output/'run_status.json').write_text(json.dumps({'state':'running','website_ready':False}))
    if digest(parent)!=PARENT_SHA:raise ValueError('Wrong HSFormer parent')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['config']['model']!='bounded_fusion_v1' or state['config']['parent_sha256']!=PARENT_SHA:
        raise ValueError('Use the completed real-triplet bounded fusion weights')
    torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
    backbone,_=load_model(parent,device);fusion=BoundedFusion().to(device)
    fusion.load_state_dict(state['model']);fusion.eval()
    cfgpath=output/'config.json'
    if cfgpath.exists():cfg=json.loads(cfgpath.read_text())
    else:
        end=datetime.now(timezone.utc).date()-timedelta(days=1)
        cfg={'end':end.isoformat(),'start':(end-timedelta(days=90)).isoformat(),
             'areas':[['lisbon',38.72,-9.14],['chennai',13.08,80.27]],
             'max_fetches_per_area':24,'checkpoint_sha256':digest(checkpoint)}
        cfgpath.write_text(json.dumps(cfg,indent=2))
    if cfg['checkpoint_sha256']!=digest(checkpoint):raise ValueError('Different checkpoint; start a new review folder')
    reports=[]
    for name,lat,lon in cfg['areas']:
        folder=output/name;cache=folder/'cache';cache.mkdir(parents=True,exist_ok=True)
        bbox=bbox_for(lat,lon);catfile=cache/'catalog.json'
        if not catfile.exists():catfile.write_text(json.dumps(client.search(bbox,cfg['start'],cfg['end'])))
        catalog=json.loads(catfile.read_text());scenes=catalog['scenes'];used=set();attempts=[]
        def fetch(s):
            stamp=s['datetime'];key=hashlib.sha256(stamp.encode()).hexdigest()[:16];p=cache/(key+'.tif')
            if stamp not in used and len(used)>=24:return None
            used.add(stamp)
            if not p.exists():
                raw=client.fetch(bbox,stamp);read_scene(raw)
                temp=p.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(p)
            return p.read_bytes()
        # Latest three distinct dates with metadata suggesting possible clouds.
        targets=[];seen=set()
        for s in sorted(scenes,key=lambda s:s['datetime'],reverse=True):
            day=s['datetime'][:10]
            if day in seen or (s.get('cloud_cover') is not None and s['cloud_cover']<10):continue
            seen.add(day);targets.append(s)
            if len(targets)==3:break
        chosen=None
        for target in targets:
            raw=fetch(target)
            if raw is None:break
            a=read_scene(raw);mask=a['cloud']&a['usable']
            if mask.mean()<.10:continue
            refs=[s for s in scenes if s['datetime'][:10]<target['datetime'][:10] and 0<days(target['datetime'],s['datetime'])<=60]
            refs.sort(key=lambda s:(s.get('cloud_cover') if s.get('cloud_cover') is not None else 101,days(target['datetime'],s['datetime'])))
            best=None
            for ref in refs[:8]:
                rr=fetch(ref)
                if rr is None:break
                b=read_scene(rr)
                if a['grid']!=b['grid']:continue
                coverage=float((mask&~b['risk']).sum()/max(1,mask.sum()))
                attempts.append({'target_date':target['datetime'],'reference_date':ref['datetime'],'cloud_reference_coverage':coverage})
                if best is None or coverage>best[0]:best=(coverage,rr,ref)
            if best and best[0]>=.95:
                chosen=(raw,best[1],target['datetime'],best[2]['datetime']);break
        (folder/'selection.json').write_text(json.dumps({'attempts':attempts,'catalog_truncated':catalog.get('truncated',False)},indent=2))
        if chosen:
            print('Running overlapping tiles:',name,flush=True)
            report=review_pair(backbone,fusion,*chosen,folder/'prediction',device)
        else:report={'state':'no_suitable_pair','notice':'Bounded search found no pair with >=95% reference coverage of flagged target pixels; no replacement image presented.'}
        reports.append({'area':name,**report});print(reports[-1],flush=True)
        (output/'summary.json').write_text(json.dumps(reports,indent=2))
    (output/'run_status.json').write_text(json.dumps({'state':'review_complete','website_ready':False,
                                                    'areas_processed':len(reports),
                                                    'successful_pairs':sum(r['state']=='qualitative_inference_complete' for r in reports)},indent=2))
    return reports
