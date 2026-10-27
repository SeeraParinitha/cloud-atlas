"""Frozen synthetic regression evaluation; not an independent accuracy claim."""
import json
from pathlib import Path
import numpy as np
import torch
from cloud_severity.engine import load_model
from .soft_pilot import validate
from .data import days
from .model import BoundedFusion
from .residual_model import ResidualFusion
from .restoration_checks import guarded_output,truth_metrics,synthetic_probability
from temporal_prior.real_review import panel

def corrupt(truth,kind):
    h,w=truth.shape[-2:];y,x=np.mgrid[:h,:w];cloud=((x-w*.5)/(w*.27))**2+((y-h*.5)/(h*.32))**2<1
    if kind=='identity':return truth.copy(),np.zeros_like(cloud)
    a=truth.copy()
    if kind=='thin':a[:,cloud]=.55*a[:,cloud]+.45*.9
    elif kind=='opaque':a[:,cloud]=.9
    elif kind=='shadow':a[:,cloud]*=.45
    else:raise ValueError(kind)
    return a,cloud

def run(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    manifest=validate(root/'data');records=[r for r in manifest['records'] if r['split']=='development_validation']
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(2)
    backbone,_=load_model(root/'parent.pt',device);backbone.eval()
    old=BoundedFusion().to(device).eval();old.load_state_dict(torch.load(root/'baseline.pt',map_location=device,weights_only=True)['model'])
    net=ResidualFusion().to(device).eval();net.load_state_dict(torch.load(root/'residual.pt',map_location=device,weights_only=True)['model'])
    rows=[]
    def t(x):return torch.tensor(x,dtype=torch.float32,device=device)[None]
    for record in records:
        with np.load(root/'data'/record['path'],allow_pickle=False) as a:
            truth=a['label'].copy();reference=a['reference'].copy();valid=a['eval_valid'][0].astype(bool);rv=a['reference_valid'][0].astype(bool)
        # The synthetic observation is on the label acquisition date, not cloudy_date.
        gap=days(record['label_date'],record['reference_date'])
        if not 0<gap<=60:
            rows.append({'id':record['id'],'state':'skipped_reference_not_earlier_than_synthetic_truth_date'});continue
        for kind in ['identity','thin','opaque','shadow']:
            cloudy,known=corrupt(truth,kind);mask=known&valid
            # Match training: soft_pilot normalizes provider CLD by 100.
            probability=synthetic_probability(mask,kind)
            if not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
                raise ValueError('Model cloud probability must be normalized to [0,1]')
            with torch.no_grad():base=backbone(t(cloudy)*t(valid[None]))['restored']
            for support_case in ['available','central_reference_missing']:
                refvalid=rv.copy()
                if support_case=='central_reference_missing':refvalid[96:160,96:160]=False
                joint=refvalid&valid;edit=mask&joint
                with torch.no_grad():
                    raw,_=net(t(cloudy),base,t(reference),t(mask[None]),t(refvalid[None]),t(valid[None]),t(probability[None]),torch.tensor([gap],device=device))
                    oldbase=backbone(t(cloudy)*t(joint[None]))['restored']*t(joint[None])
                    previous=old(t(cloudy)*t(joint[None]),oldbase,t(reference)*t(joint[None]),t(edit[None]),torch.tensor([gap],device=device),t(joint[None]))[0].cpu().numpy()
                raw=raw[0].cpu().numpy();guard,masks=guarded_output(cloudy,raw,mask,valid,refvalid)
                outputs={'input':cloudy,'old_fusion':np.where(edit[None],previous,cloudy),'reference_copy':np.where(edit[None],reference,cloudy),'raw_residual':raw,'guarded_residual':guard}
                for method,pred in outputs.items():
                    metrics=truth_metrics(pred,truth,valid,mask,'synthetic_exact')
                    rows.append({'id':record['id'],'group':record['group'],'acquisition_group':record['label_date'],'kind':kind,'support':support_case,'method':method,'metrics':metrics,'unresolved_fraction':float(masks['unresolved'].mean()),'clear_input_unchanged':bool(np.array_equal(pred[:,valid&~mask],cloudy[:,valid&~mask])),'unsupported_input_unchanged':bool(np.array_equal(pred[:,masks['unresolved']],cloudy[:,masks['unresolved']]))})
                name=record['id']+'_'+kind+'_'+support_case
                panel([truth,cloudy,reference,outputs['old_fusion'],raw,guard],['Exact synthetic truth','Corrupted','Earlier reference','Old fusion','Raw residual','Guarded residual'],out/(name+'.png'))
                (out/'metrics.json').write_text(json.dumps(rows,indent=2));print(name,flush=True)
    (out/'status.json').write_text(json.dumps({'state':'evaluation_complete','records':len(records),'evaluated_rows':sum('method' in r for r in rows),'skipped_records':[r for r in rows if 'state' in r],'training':False,'website_updated':False,'notice':'Development regression test, ideal known corruption mask. Synthetic truth is exact only on valid pixels. Earlier references retain real temporal differences. Not real-cloud accuracy or a detector test.'},indent=2))
    return rows
