"""Deterministic validation-only selection and portable inference export."""
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd
import torch
from .engine import save_json

METRICS={'psnr':False,'ssim':False,'lpips':True,'stable_clear_change_mae':True}

def select_model(reports):
    rows=[]
    for name,report in reports.items():
        if report['protocol']!='in-domain validation': raise ValueError('Selection accepts validation reports only')
        rows.append({'variant':name,**{k:report['overall'].get(k) for k in METRICS},'inference_ms':report['latency_ms_median']})
    frame=pd.DataFrame(rows).set_index('variant')
    if frame.empty or frame[list(METRICS)].isna().any().any(): raise ValueError('Complete validation metrics required for selection')
    for k in METRICS:
        if not pd.to_numeric(frame[k],errors='coerce').map(lambda v: __import__('math').isfinite(v)).all():
            raise ValueError('Non-finite validation metric; inspect results before selection')
    ranks=pd.DataFrame({k:frame[k].rank(ascending=ascending,method='average') for k,ascending in METRICS.items()})
    frame['mean_rank']=ranks.mean(axis=1)
    frame=frame.sort_values(['mean_rank','psnr','inference_ms'],ascending=[True,False,True],kind='stable')
    return frame.index[0],frame.reset_index().to_dict('records')


def export_model(checkpoint,decision,destination='deployment'):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    # Optimizer state and original image paths are not needed by the public service.
    torch.save({'config':state['config'],'model_state':state['model_state'],'manifest_sha256':state['manifest_sha256']},destination/'model.pt')
    save_json(destination/'selection.json',decision)
    save_json(destination/'preprocessing.json',{'input':'RGB','range':[0,1],'tile':256,'overlap':64,
        'satellite_collection':'sentinel-2-l1c','bands':['B04','B03','B02'],'satellite_reflectance_ceiling':.4,
        'satellite_scaling':'clip(reflectance / 0.4, 0, 1); display-RGB adaptation, not independently calibrated on satellite data',
        'satellite_domain_validated':False,'output':'estimated display RGB, not physical reflectance'})
    h=hashlib.sha256((destination/'model.pt').read_bytes()).hexdigest()
    save_json(destination/'artifact.json',{'sha256':h,'variant':state['config']['model']['variant']})
