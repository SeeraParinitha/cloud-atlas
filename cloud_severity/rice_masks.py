"""Explicit RICE LandsatLook palette conversion; Table 1, arXiv:1901.00600."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image


def decode_rice_mask(path):
    with Image.open(path) as im:
        a=np.asarray(im.convert('RGB'))
    black=np.all(a==[0,0,0],axis=-1)
    cloud=np.all(a==[255,255,255],axis=-1)
    shadow=np.all(a==[0,0,255],axis=-1)
    cirrus=np.all(a==[255,255,0],axis=-1)
    undocumented=np.all(a==[255,0,0],axis=-1)|np.all(a==[0,255,255],axis=-1)
    if not (black|cloud|shadow|cirrus|undocumented).all():
        raise ValueError(f'Unknown RICE quality-mask colour: {path}')
    if undocumented.any():
        return None  # No verified class meaning: use existing pair-derived pseudo supervision.
    # Restoration support includes shadows; never classify blue shadows as clear.
    return ((cloud|shadow|cirrus)*255).astype('uint8')


def prepare_rice_masks(records, destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    count=0
    for row in records:
        if row['dataset']!='RICE2' or not row.get('mask'):continue
        source=Path(row['mask'])
        target=destination/(hashlib.sha256(row['id'].encode()).hexdigest()+'.png')
        decoded=decode_rice_mask(source)
        if decoded is None:
            row['original_mask']=str(source)
            row['mask']=None
            row['mask_definition']='Pseudo supervision: original RICE quality mask contains undocumented red/cyan labels'
            continue
        Image.fromarray(decoded).save(target)
        row['original_mask']=str(source)
        row['mask']=str(target.resolve())
        row['mask_definition']='RICE quality support: white cloud OR blue shadow OR yellow cirrus; black unflagged'
        count+=1
    return count