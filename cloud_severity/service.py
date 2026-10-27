"""Shared satellite restoration service; used by FastAPI and Kaggle demo."""
import json
import os
import threading
import time
import uuid
from pathlib import Path
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import torch
from PIL import Image
from .engine import load_model
from .satellite import SatelliteClient
from .tiling import restore_tiled


class RestorationService:
    def __init__(self, deployment='deployment', output='web_results', client=None):
        self.deployment=Path(deployment); self.output=Path(output); self.output.mkdir(parents=True,exist_ok=True)
        self.client=client or SatelliteClient(); self.lock=threading.Lock(); self.model=None
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def status(self):
        return {'model_ready':(self.deployment/'model.pt').exists(),'satellite_ready':self.client.configured,'device':str(self.device)}

    def restore(self,bbox,acquisition):
        with self.lock:
            if not (self.deployment/'model.pt').exists(): raise RuntimeError('Train and select the deployment model first')
            if self.model is None:
                self.model,self.state=load_model(self.deployment/'model.pt',self.device)
            started=time.perf_counter(); raw=self.client.fetch(bbox,acquisition)
            with MemoryFile(raw) as mem:
                with mem.open() as ds:
                    data=ds.read(); profile=ds.profile.copy()
            if data.shape[0]!=4: raise RuntimeError('Provider did not return RGB plus validity mask')
            valid=(data[3]>.5)&np.isfinite(data[:3]).all(0)
            if not valid.any(): raise ValueError('No valid satellite pixels for this area/acquisition')
            config=json.loads((self.deployment/'preprocessing.json').read_text())
            ceiling=float(config['satellite_reflectance_ceiling'])
            rgb=np.nan_to_num(data[:3]/ceiling,nan=0,posinf=1,neginf=0).clip(0,1)
            rgb[:,~valid]=0
            pred=restore_tiled(self.model,torch.from_numpy(rgb).float(),self.device)
            restored=pred['restored'].numpy(); restored[:,~valid]=0
            ident=uuid.uuid4().hex; folder=self.output/ident; folder.mkdir()
            (folder/'source_reflectance.tif').write_bytes(raw)
            def png(arr,path):
                Image.fromarray((np.moveaxis(arr,0,-1)*255).round().astype('uint8')).save(path)
            png(rgb,folder/'original.png'); png(restored,folder/'restored.png')
            profile.update(count=3,dtype='uint8',nodata=None,compress='deflate')
            with rasterio.open(folder/'restored.tif','w',**profile) as dst:
                dst.write((restored*255).round().astype('uint8')); dst.write_mask(valid.astype('uint8')*255)
                dst.update_tags(description='Model-estimated display RGB, not measured reflectance',acquisition=acquisition)
            maps=[]
            for key in ['mask','severity']:
                if key in pred:
                    im=pred[key][0].numpy(); im[~valid]=0
                    Image.fromarray((im*255).round().astype('uint8')).save(folder/f'{key}.png'); maps.append(key)
            meta={'id':ident,'bbox':bbox,'acquisition':acquisition,'model':self.state['config']['model']['variant'],
                  'seconds':round(time.perf_counter()-started,3),'valid_fraction':float(valid.mean()),
                  'width':rgb.shape[2],'height':rgb.shape[1],'maps':maps,'preprocessing':config,
                  'notice':'Restored pixels are estimates. Satellite-domain performance is not established by RICE validation. Thick-cloud ground details cannot be verified from this image.'}
            (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
            return meta,folder
