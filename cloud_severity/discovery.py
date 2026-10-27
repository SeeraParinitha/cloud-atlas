"""Find the official RICE layout without guessing individual file paths."""
from pathlib import Path


def discover_rice(root):
    root=Path(root)
    found={}
    aliases={'rice1':'RICE1','ricei':'RICE1','rice2':'RICE2','riceii':'RICE2'}
    for folder in [root]+sorted(p for p in root.rglob('*') if p.is_dir()):
        key=''.join(c for c in folder.name.lower() if c.isalnum())
        if key not in aliases: continue
        children={p.name.lower():p for p in folder.iterdir() if p.is_dir()}
        cloud=next((children[k] for k in ['cloud','cloudy','cloudy_image'] if k in children),None)
        clear=next((children[k] for k in ['label','clear','cloudfree','cloud-free','ground_truth'] if k in children),None)
        mask=next((children[k] for k in ['mask','masks'] if k in children),None)
        if cloud is None or clear is None: continue
        name=aliases[key]
        item=dict(name=name,cloudy=str(cloud.resolve()),clear=str(clear.resolve()),masks=str(mask.resolve()) if mask else None)
        if name in found and found[name]!=item: raise ValueError(f'Multiple copies of {name} found; use an explicit source configuration')
        found[name]=item
    if set(found)!={'RICE1','RICE2'}: raise ValueError(f'Expected RICE1 and RICE2 under {root}; found {list(found)}')
    return [found[k] for k in sorted(found)]
