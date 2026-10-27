"""Authorize a bounded continuation without resetting optimizer/best/history."""
import hashlib,json
from pathlib import Path
import torch
from .data import digest


def restore_input(inputs,folder):
    """Discover by checkpoint contents; allow renamed archives/extracted datasets."""
    import zipfile
    inputs=Path(inputs);folder=Path(folder);candidates=[]
    def eligible(cfg):
        return cfg.get('model')=='soft_dense_fusion_v1' and cfg.get('initialization')=='balanced' and cfg.get('epochs') in (12,36)
    for path in inputs.rglob('*.zip'):
        print('Inspecting checkpoint archive:',path,flush=True)
        with zipfile.ZipFile(path) as archive:
            names=set(archive.namelist())
            for name in names:
                if not name.endswith('config.json'):continue
                prefix=name[:-len('config.json')]
                if prefix+'last.pt' not in names or prefix+'best.pt' not in names:continue
                cfg=json.loads(archive.read(name))
                if not eligible(cfg):continue
                h=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode())
                for checkpoint in ('last.pt','best.pt'):
                    with archive.open(prefix+checkpoint) as stream:
                        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
                candidates.append((h.hexdigest(),path,prefix))
    for path in inputs.rglob('config.json'):
        if not (path.parent/'last.pt').is_file() or not (path.parent/'best.pt').is_file():continue
        print('Inspecting extracted checkpoint:',path.parent,flush=True)
        cfg=json.loads(path.read_text())
        if not eligible(cfg):continue
        h=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode())
        for checkpoint in ('last.pt','best.pt'):
            with (path.parent/checkpoint).open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        candidates.append((h.hexdigest(),path.parent,None))
    unique={}
    for ident,path,prefix in candidates:unique.setdefault(ident,(path,prefix))
    if not unique:raise RuntimeError('No balanced V2 continuation checkpoint found. Add the dataset containing config.json, last.pt and best.pt from soft_fusion_pilot_v2_results.zip (ZIP or extracted).')
    if len(unique)>1:raise RuntimeError('Different V2 checkpoints are attached. Keep only the intended V2 or latest continuation dataset. Found: '+', '.join(str(p) for p,_ in unique.values()))
    source,prefix=next(iter(unique.values()));folder.mkdir(parents=True,exist_ok=True)
    print('Restoring checkpoint files from:',source,flush=True)
    def destination(relative):
        path=(folder/relative).resolve()
        if folder.resolve() not in path.parents:raise ValueError('Unsafe checkpoint archive path')
        path.parent.mkdir(parents=True,exist_ok=True);return path
    if prefix is not None:
        with zipfile.ZipFile(source) as archive:
            for entry in archive.infolist():
                if entry.is_dir() or not entry.filename.startswith(prefix):continue
                relative=entry.filename[len(prefix):]
                if relative.endswith('.tmp'):continue
                path=destination(relative)
                if not path.exists():path.write_bytes(archive.read(entry))
    else:
        import shutil
        for entry in source.rglob('*'):
            if not entry.is_file() or entry.suffix=='.tmp':continue
            path=destination(entry.relative_to(source))
            if not path.exists():shutil.copyfile(entry,path)
    print('Continuation checkpoint source:',source)
    return str(source)


def prepare_continuation(folder):
    folder=Path(folder)
    for name in ('config.json','last.pt','best.pt'):
        if not (folder/name).is_file():raise ValueError('Missing V2 continuation file: '+name)
    cfg=json.loads((folder/'config.json').read_text())
    expected={'model':'soft_dense_fusion_v1','initialization':'balanced','patience':4,
              'learning_rate':.0002,'seed':721,'batch':1,'accumulation':4}
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Not the unchanged V2 pilot protocol')
    code=''.join(Path(__file__).with_name(n).read_text(encoding='utf-8-sig') for n in ('soft_pilot.py','soft_model.py'))
    if cfg['code_sha256']!=hashlib.sha256(code.encode()).hexdigest():raise ValueError('Training code differs from V2; do not silently continue')
    if cfg['epochs'] not in (12,36):raise ValueError('Unsupported epoch limit')
    last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
    best=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    if not 12<=last['epoch']<=36 or not last.get('optimizer',{}).get('state'):
        raise ValueError('Expected epoch-12-or-later checkpoint with optimizer state')
    for k in ('model','initialization','manifest_sha256','parent_sha256','baseline_sha256','code_sha256'):
        if best['config'].get(k)!=cfg.get(k):raise ValueError('Mixed checkpoint provenance')
        if 'config' in last and last['config'].get(k)!=cfg.get(k):raise ValueError('Mixed last-checkpoint provenance')
    if len(last['history'])!=last['epoch'] or best['epoch']>last['epoch']:raise ValueError('Inconsistent checkpoint history')
    if set(last['model'])!=set(best['model']):raise ValueError('Different checkpoint architectures')
    if last['epoch']==best['epoch'] and any(not torch.equal(last['model'][k],best['model'][k]) for k in best['model']):raise ValueError('Same-epoch checkpoint weights differ')
    if abs(last['best']-min(h['selection_score'] for h in last['history']))>1e-7:raise ValueError('Best score and history differ')
    if cfg['epochs']==12:
        if last['epoch']!=12:raise ValueError('Expected completed V2 epoch 12')
        lineage={'original_config':cfg,'original_last_sha256':digest(folder/'last.pt'),
                 'original_best_sha256':digest(folder/'best.pt'),'continued_from_epoch':12,
                 'new_maximum_epoch':36,'additional_epochs_maximum':24,
                 'optimizer_reset':False,'best_score_reset':False,'patience_reset':False}
        (folder/'continuation_lineage.json').write_text(json.dumps(lineage,indent=2))
    elif not (folder/'continuation_lineage.json').exists():raise ValueError('Missing continuation provenance')
    cfg['epochs']=36;cfg['hours']=1
    (folder/'config.json').write_text(json.dumps(cfg,indent=2))
    return {'resume_epoch':last['epoch'],'maximum_epoch':36,'stale_epochs':last['stale'],
            'saved_best_epoch':best['epoch'],'maximum_additional_epochs':36-last['epoch']}
