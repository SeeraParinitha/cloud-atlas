"""Fixed-split, single-seed research command line. No test-driven selection."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .data import (PairedDataset, PseudoConfig, pair_records, split_records,
                   write_manifest, validate_records)
from .rice_masks import prepare_rice_masks
from .engine import train, evaluate, infer, save_json, checked_manifest
from .experiments import code_hash


def calibrate(manifest, size=256):
    torch.set_num_threads(4)
    ds = PairedDataset(manifest, 'train', size, False,
                       PseudoConfig(noise_floor=0, scale=1, threshold=0, sigma=1))
    rng = np.random.default_rng(42)
    chosen = []
    for name in sorted({r['dataset'] for r in ds.records}):
        indices = [i for i,r in enumerate(ds.records) if r['dataset'] == name]
        chosen.extend(rng.choice(indices, min(64, len(indices)), replace=False).tolist())
    items = [ds[i] for i in chosen]
    maps = [(x['severity']-.02).clamp_min(0) for x in items]
    samples = torch.cat([x.flatten()[::max(1,x.numel()//4096)] for x in maps])
    scale = max(float(torch.quantile(samples,.99)),.01)
    maps = [(x/scale).clamp(0,1) for x in maps]
    reliable = [(s,x['mask']) for s,x in zip(maps,items) if x['mask_source']=='dataset']
    grid = np.linspace(.05,.95,19)
    if reliable and sum(float(m.sum()) for _,m in reliable)>0:
        scores = []
        for t in grid:
            intersection=sum(float(((s>t)&(m>.5)).sum()) for s,m in reliable)
            union=sum(float(((s>t)|(m>.5)).sum()) for s,m in reliable)
            scores.append(intersection/union if union else 1.)
        threshold=float(grid[np.argmax(scores)])
        method='training mask IoU grid'
    else:
        # Otsu computed from training-only pixels.
        hist,edges=np.histogram(torch.cat([s.flatten()[::max(1,s.numel()//4096)] for s in maps]).numpy(),256,(0,1))
        p=hist/max(hist.sum(),1); centers=(edges[:-1]+edges[1:])/2
        w=np.cumsum(p); m=np.cumsum(p*centers); valid=(w*(1-w))>1e-12
        score=np.full(256,-np.inf); score[valid]=(m[-1]*w[valid]-m[valid])**2/(w[valid]*(1-w[valid]))
        threshold=float(centers[np.argmax(score)]) if valid.any() else .15
        method='training-only Otsu'
    pseudo=PseudoConfig(scale=scale,threshold=threshold)
    values=[]
    for x in PairedDataset(manifest,'train',size,False,pseudo):
        active=x['severity']>threshold
        values.append(float(x['severity'][active].mean()) if active.any() else 0.)
    bins=np.quantile(values,[1/3,2/3]).tolist()
    if bins[1]-bins[0]<1e-4: bins=[.33,.66]
    return dict(pseudo=asdict(pseudo), severity_bins=bins,
                calibration_ids=[x['id'] for x in items], threshold_method=method)


def main():
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--sources',required=True); p.add_argument('--manifest',default='data/rice.json')
    p=sub.add_parser('calibrate'); p.add_argument('--manifest',default='data/rice.json'); p.add_argument('--output',default='data/calibration.json'); p.add_argument('--size',type=int,default=256)
    p=sub.add_parser('train'); p.add_argument('--config',default='configs/research.json'); p.add_argument('--calibration',default='data/calibration.json'); p.add_argument('--variant',choices=['full','mask_blend','hsformer'],required=True)
    p=sub.add_parser('evaluate'); p.add_argument('--checkpoint',required=True); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True); p.add_argument('--split',choices=['validation','test'],default='validation'); p.add_argument('--external-protocol'); p.add_argument('--without-lpips',action='store_true')
    p=sub.add_parser('infer'); p.add_argument('--checkpoint',required=True); p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        path=Path(args.manifest)
        if path.exists(): raise ValueError('Manifest exists. Preserve the frozen split; use another path for a new experiment.')
        sources=json.loads(Path(args.sources).read_text()); records=[]
        for s in sources:
            records.extend(pair_records(s['cloudy'],s['clear'],s['name'],s.get('masks'),s.get('groups')))
        prepare_rice_masks(records,path.parent/'binary_masks')
        split_records(records,42); write_manifest(path,records,42)
        print('Manifest content hash:',checked_manifest(path))
    elif args.command=='calibrate':
        if Path(args.output).exists(): raise ValueError('Calibration exists; do not silently replace frozen settings')
        result=calibrate(args.manifest,args.size)
        result['manifest_sha256']=checked_manifest(args.manifest); result['size']=args.size
        save_json(args.output,result)
    elif args.command=='train':
        cfg=json.loads(Path(args.config).read_text()); cal=json.loads(Path(args.calibration).read_text())
        if cal['manifest_sha256']!=checked_manifest(cfg['manifest']) or cal['size']!=cfg['size']:
            raise ValueError('Calibration does not match manifest or resolution')
        cfg.update({k:cal[k] for k in ['pseudo','severity_bins']})
        cfg['model']['variant']=args.variant; cfg['output']=str(Path(cfg['output']).parent/args.variant)
        cfg['code_sha256']=code_hash()
        if args.variant=='hsformer':
            p=Path(__file__).resolve().parents[1]/'third_party/hsformer_runtime/provenance.json'
            cfg['hsformer_provenance']=json.loads(p.read_text())
        last=Path(cfg['output'])/'last.pt'
        result=train(cfg,resume=str(last) if last.exists() else None)
        print('Training complete:' if result else 'Session stopped; resume the same command:',result or cfg['output'])
    elif args.command=='evaluate':
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        status=Path(args.checkpoint).parent/'status.json'
        if not status.exists() or not json.loads(status.read_text()).get('completed'):
            raise ValueError('Finish the training schedule before evaluation')
        external=bool(args.external_protocol)
        if external:
            protocol=json.loads(Path(args.external_protocol).read_text())
            required=['dataset_version','official_test_split_evidence','bands','normalization','alignment_checks','spatial_resolution','transforms','manifest_sha256']
            if any(not protocol.get(k) for k in required): raise ValueError('External protocol is incomplete')
            if args.split!='test' or protocol['manifest_sha256']!=checked_manifest(args.manifest): raise ValueError('External test manifest mismatch')
            save_json(Path(args.output)/'external_protocol.json',protocol)
        evaluate(args.checkpoint,args.manifest,args.output,split=args.split,cross_dataset=external,with_lpips=not args.without_lpips)
    else:
        infer(args.checkpoint,args.input,args.output)

if __name__=='__main__': main()
