"""Load only deployment assets from the user's trained output archive."""
import hashlib,json,zipfile
from pathlib import Path

def restore_archive(archive,destination='.'):
    root=Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        matches=[n for n in z.namelist() if n.endswith('deployment/artifact.json')]
        if len(matches)!=1:raise ValueError('Expected one trained deployment/artifact.json; attach the completed training ZIP')
        prefix=matches[0][:-len('deployment/artifact.json')]
        required=['deployment/model.pt','deployment/preprocessing.json','deployment/selection.json','deployment/artifact.json']
        for name in required:
            if prefix+name not in z.namelist():raise ValueError('Missing trained asset: '+name)
        expected=json.loads(z.read(prefix+'deployment/artifact.json'))['sha256']
        selected=[(n,n[len(prefix):]) for n in z.namelist() if n.startswith(prefix) and (n[len(prefix):] in required or n[len(prefix):].startswith('third_party/hsformer_runtime/')) and not n.endswith('/')]
        if sum(z.getinfo(n).file_size for n,_ in selected)>1024**3:raise ValueError('Deployment archive is unexpectedly large')
        for name,relative in selected:
            target=(root/relative).resolve()
            if root not in target.parents or '\\' in relative or (z.getinfo(name).external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe deployment archive path')
            data=z.read(name)
            if relative=='deployment/model.pt' and hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Model checksum mismatch')
            if target.exists() and target.read_bytes()!=data:raise ValueError('Different deployment already exists: '+str(target))
        for name,relative in selected:
            target=root/relative
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                temp=target.with_suffix(target.suffix+'.part');temp.write_bytes(z.read(name));temp.replace(target)
    return root/'deployment/model.pt'

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--destination',default='.')
    a=p.parse_args();print(restore_archive(a.archive,a.destination))
