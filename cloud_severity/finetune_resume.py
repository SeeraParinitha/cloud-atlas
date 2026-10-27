"""Restore only saved experiment data/state, never notebook code or secrets."""
from pathlib import Path
import zipfile
import shutil


def restore(archive, destination):
    root = Path(destination).resolve()
    allowed = ('finetune/', 'data/sentinel/', 'third_party/hsformer_runtime/')
    source = Path(archive)
    if source.is_dir():
        source = source.resolve()
        if not (source/'finetune/status.json').is_file():
            raise ValueError('Missing extracted fine-tuning status')
        files = [p for folder in allowed for p in (source/folder).rglob('*') if p.is_file()]
        if any(p.is_symlink() or source not in p.resolve().parents for p in files):
            raise ValueError('Unsafe extracted resume path')
        if sum(p.stat().st_size for p in files)>8*1024**3:
            raise ValueError('Resume directory unexpectedly large')
        for path in files:
            target = (root/path.relative_to(source)).resolve()
            if root not in target.parents:
                raise ValueError('Unsafe destination path')
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,target)
        return root/'finetune/status.json'
    with zipfile.ZipFile(archive) as z:
        selected = []
        for info in z.infolist():
            name = info.filename
            target = (root/name).resolve()
            if '\\' in name or root not in target.parents or (info.external_attr >> 16)&0o170000 == 0o120000:
                raise ValueError('Unsafe resume archive member')
            if not info.is_dir() and name.startswith(allowed):
                selected.append(info)
        if not any(i.filename=='finetune/status.json' for i in selected):
            raise ValueError('This is not a Sentinel-2 fine-tuning resume archive')
        if sum(i.file_size for i in selected)>8*1024**3 or any(i.file_size>1024**3 for i in selected):
            raise ValueError('Resume archive unexpectedly large')
        for info in selected:
            target = root/info.filename
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                temp = target.with_suffix('.part')
                temp.write_bytes(z.read(info)); temp.replace(target)
    return root/'finetune/status.json'
