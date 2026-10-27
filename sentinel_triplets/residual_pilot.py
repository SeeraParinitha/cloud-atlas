"""Small development-only pilot; no test/audit ingestion and no web deployment."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cloud_severity.engine import load_model
from temporal_prior.train import PARENT_SHA,atomic_save
from temporal_prior.real_review import panel
from .data import digest,days
from .independent_data import FUSION_SHA
from .development_data import SPLITS
from .model import BoundedFusion
from .soft_model import SoftFusion
from .residual_model import ResidualFusion


def validate(root):
    root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text())
    if m.get('schema')!='temporal_development_v1' or not m.get('ready_for_training_design'):
        raise ValueError('Use the ready temporal_development_data ZIP, not independent evaluation data')
    groups={};seen=set()
    for r in m['records']:
        if r['id'] in seen or SPLITS.get(r['group'])!=r['split']:raise ValueError('Incorrect development split')
        seen.add(r['id']);groups[r['group']]=r['split']
        stamps=[r[k] for k in ['cloudy_date','reference_date','label_date']]
        if len({s[:10] for s in stamps})!=3 or not 0<days(stamps[0],stamps[1])<=60 or abs(days(stamps[2],stamps[0]))>21:raise ValueError('Invalid dates')
        p=(root/r['path']).resolve()
        if root not in p.parents or digest(p)!=r['sha256']:raise ValueError('Unsafe or changed data')
        with np.load(p,allow_pickle=False) as a:
            for k in ['cloudy','reference','label']:
                if a[k].shape!=(3,256,256) or not np.isfinite(a[k]).all() or a[k].min()<0 or a[k].max()>1:raise ValueError('Invalid RGB')
            for k in ['target_valid','reference_valid','eval_valid','mask','high_cloud','shadow_provider']:
                if a[k].shape!=(1,256,256) or not np.isin(a[k],[0,1]).all():raise ValueError('Invalid mask')
            if (a['eval_valid']&~a['target_valid']).any():raise ValueError('Invalid evaluation support')
    if len([g for g,s in groups.items() if s=='development_train'])<4 or len([g for g,s in groups.items() if s=='development_validation'])!=2:raise ValueError('Insufficient locations')
    return m


def mean_error(pred,target,mask):
    return ((pred-target).abs()*mask).sum()/(3*mask.sum()).clamp_min(1)


def summarize(rows):
    result={}
    for method in sorted({r['method'] for r in rows}):
        result[method]={}
        for key in ['all_mae','cloud_mae','unflagged_proxy_mae','unflagged_input_change','shadow_mae','psnr']:
            locations=[]
            for group in sorted({r['group'] for r in rows}):
                acquisitions=[]
                for acquisition in sorted({r['acquisition_group'] for r in rows if r['group']==group}):
                    vals=[r[key] for r in rows if r['method']==method and r['group']==group and r['acquisition_group']==acquisition and r[key] is not None]
                    if vals:acquisitions.append(float(np.mean(vals)))
                if acquisitions:locations.append(float(np.mean(acquisitions)))
            result[method][key]=float(np.mean(locations)) if locations else None
    return result


def synthesize(label,valid,seed,kind):
    """Controlled synthetic supervision only; do not call it real thick-cloud truth."""
    generator=torch.Generator(device=label.device).manual_seed(seed)
    noise=torch.rand((1,1,4,4),generator=generator,device=label.device)
    field=F.interpolate(noise,size=label.shape[-2:],mode='bilinear',align_corners=False)
    if kind=='identity':return label.clone()
    if kind=='haze':return (label*(1-(.15+.4*field)) +(.15+.4*field)*.85)*valid
    if kind=='shadow':return label*(.4+.4*field)*valid
    raise ValueError('Unknown synthetic case')


def create_model(device,initialization):
    if initialization!='zero_residual':raise ValueError('Use zero_residual')
    return ResidualFusion().to(device)


def run(data,parent,old_checkpoint,output,epochs=36,hours=1,initialization="zero_residual",soft_checkpoint=None):
    if initialization!="zero_residual":raise ValueError("Unknown initialization")
    data=Path(data);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    m=validate(data)
    if digest(parent)!=PARENT_SHA or digest(old_checkpoint)!=FUSION_SHA:raise ValueError('Incorrect baseline weights')
    if not torch.cuda.is_available():raise RuntimeError('Enable T4 GPU for this pilot')
    device='cuda';torch.set_num_threads(2);torch.manual_seed(721);np.random.seed(721);started=time.monotonic()
    code=Path(__file__).read_bytes()+Path(__file__).with_name('residual_model.py').read_bytes()
    cfg={'model':'signed_residual_pilot_v1','result_key':'soft_fusion means residual candidate in this output','loss':'valid_proxy_mae_only','soft_checkpoint_sha256':digest(soft_checkpoint),'initialization':initialization,'manifest_sha256':digest(data/'manifest.json'),'parent_sha256':PARENT_SHA,
         'baseline_sha256':FUSION_SHA,'code_sha256':hashlib.sha256(code).hexdigest(),'epochs':epochs,'patience':4,
         'hours':hours,'learning_rate':.0002,'seed':721,'batch':1,'accumulation':4,
         'selection':'location-balanced validation all-pixel proxy MAE; cloud/shadow metrics reported separately'}
    cfgpath=out/'config.json'
    if cfgpath.exists() and json.loads(cfgpath.read_text())!=cfg:raise ValueError('Changed experiment; use a fresh output folder')
    cfgpath.write_text(json.dumps(cfg,indent=2))
    def status(state,**extra):
        value={'state':state,'website_updated':False,'elapsed_seconds':time.monotonic()-started,**extra}
        (out/'status.json').write_text(json.dumps(value,indent=2));return value
    def expired():return time.monotonic()-started>max(0,hours*3600-180)
    status('caching')
    backbone,_=load_model(parent,device);backbone.eval()
    for p in backbone.parameters():p.requires_grad_(False)
    old=BoundedFusion().to(device).eval();old.load_state_dict(torch.load(old_checkpoint,map_location=device,weights_only=True)['model'])
    records=m['records'];cache=out/'cache';cache.mkdir(exist_ok=True)
    for r in records:
        p=cache/(r['id']+'.pt')
        if not p.exists():
            with np.load(data/r['path'],allow_pickle=False) as a:b={k:torch.tensor(a[k],dtype=torch.float32)[None].to(device) for k in a.files}
            b['probability']=(b['target_cloud_probability'][:,None]/100).clamp(0,1)
            b['days']=torch.tensor([days(r['cloudy_date'],r['reference_date'])],device=device)
            with torch.no_grad():
                b['base']=backbone(b['cloudy']*b['target_valid'])['restored']
                joint=b['target_valid']*b['reference_valid'];edit=b['mask']*joint
                previous_base=backbone(b['cloudy']*joint)['restored']*joint
                previous=old(b['cloudy']*joint,previous_base,b['reference']*joint,edit,b['days'],joint)
                b['old_fusion']=torch.where(edit.bool(),previous,b['cloudy'])
            atomic_save({k:v.cpu() for k,v in b.items()},p)
        if expired():return status('paused',stage='caching')
    del old
    def get(r):return {k:v.to(device) for k,v in torch.load(cache/(r['id']+'.pt'),weights_only=True).items()}
    def forward(net,b):return net(b['cloudy'],b['base'],b['reference'],b['mask'],b['reference_valid'],b['target_valid'],b['probability'],b['days'])[0]
    soft=SoftFusion().to(device).eval();soft.load_state_dict(torch.load(soft_checkpoint,map_location=device,weights_only=True)['model'])
    net=create_model(device,initialization)
    optim=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=1e-4)
    train=[r for r in records if r['split']=='development_train'];val=[r for r in records if r['split']=='development_validation']
    def evaluate(export=False):
        net.eval();rows=[]
        with torch.no_grad():
            for r in val:
                b=get(r);prediction=forward(net,b);ev=b['eval_valid'];mask=b['mask'];joint=mask*b['reference_valid']*b['target_valid']
                outputs={'cloudy':b['cloudy'],'hsformer':b['base'],'old_fusion':b['old_fusion'],
                         'reference_copy':torch.where(joint.bool(),b['reference'],b['cloudy']),'soft_epoch36':forward(soft,b),'zero_residual':torch.where(b['target_valid'].bool(),torch.where(b['reference_valid'].bool(),b['reference'],b['base']),b['cloudy']),'soft_fusion':prediction}
                for name,pred in outputs.items():
                    if not torch.isfinite(pred).all():raise ValueError('Nonfinite prediction')
                    def metric(region,target=b['label']):return float(mean_error(pred,target,region)) if region.sum()>0 else None
                    mse=((pred-b['label']).square()*ev).sum()/(3*ev.sum()).clamp_min(1)
                    rows.append({'id':r['id'],'group':r['group'],'acquisition_group':r['acquisition_group'],'method':name,
                                 'all_mae':metric(ev),'cloud_mae':metric(ev*mask),'unflagged_proxy_mae':metric(ev*(1-mask)),
                                 'unflagged_input_change':metric(b['target_valid']*(1-mask),b['cloudy']),
                                 'shadow_mae':metric(ev*b['shadow_provider']),'psnr':float(-10*torch.log10(mse.clamp_min(1e-12)))})
                if export:panel([b[k][0].cpu().numpy() for k in ['cloudy','reference']]+[b['old_fusion'][0].cpu().numpy(),prediction[0].cpu().numpy(),b['label'][0].cpu().numpy()],['Cloudy','Earlier reference','Old fusion','Residual pilot','Different-date proxy'],out/(r['id']+'.png'))
        if export:(out/'validation_per_sample.json').write_text(json.dumps(rows,indent=2))
        return summarize(rows)
    initial_metrics=evaluate()
    first=0;best=float('inf');stale=0;history=[]
    if (out/'last.pt').exists():
        resume=torch.load(out/'last.pt',map_location=device,weights_only=True);net.load_state_dict(resume['model']);optim.load_state_dict(resume['optimizer'])
        first,best,stale,history=resume['epoch'],resume['best'],resume['stale'],resume['history']
    for epoch in range(first,epochs):
        if stale>=4:break
        if expired():return status('paused',stage='before_epoch',completed_epochs=epoch)
        net.train();optim.zero_grad(set_to_none=True);rng=np.random.default_rng(721+epoch)
        order=rng.permutation(len(train));n=len(order)
        for i,index in enumerate(order):
            b=get(train[index]);real_pred=forward(net,b)
            loss=mean_error(real_pred,b['label'],b['eval_valid'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            (loss/min(4,n-(i//4)*4)).backward()
            if (i+1)%4==0 or i+1==n:torch.nn.utils.clip_grad_norm_(net.parameters(),1);optim.step();optim.zero_grad(set_to_none=True)
            if expired():return status('paused',stage='epoch_interrupted',completed_epochs=epoch)
        metrics=evaluate();s=metrics['soft_fusion'];score=s['all_mae']
        history.append({'epoch':epoch+1,'validation':metrics,'selection_score':score})
        if score<best:
            best=score;stale=0;atomic_save({'model':net.state_dict(),'epoch':epoch+1,'config':cfg},out/'best.pt')
        else:stale+=1
        atomic_save({'model':net.state_dict(),'optimizer':optim.state_dict(),'epoch':epoch+1,'best':best,'stale':stale,'history':history},out/'last.pt')
        print('Epoch',epoch+1,'validation score',score,flush=True);status('training',completed_epochs=epoch+1)
    (out/'initial_baselines.json').write_text(json.dumps(initial_metrics,indent=2))
    chosen=torch.load(out/'best.pt',map_location=device,weights_only=True);net.load_state_dict(chosen['model']);final=evaluate(True)
    # Fixed synthetic validation probes; they are not real-cloud evidence.
    probes=[]
    net.eval()
    with torch.no_grad():
        for i,r in enumerate(val):
            for kind in ['identity','haze','shadow']:
                b=get(r);target=b['label'];b['target_valid']=b['eval_valid'];b['cloudy']=synthesize(target,b['target_valid'],1721+i,kind);b['mask'].zero_();b['probability'].zero_()
                b['base']=backbone(b['cloudy']*b['target_valid'])['restored'];pred=forward(net,b)
                probes.append({'id':r['id'],'kind':kind,'input_mae':float(mean_error(b['cloudy'],target,b['eval_valid'])),
                               'model_mae':float(mean_error(pred,target,b['eval_valid']))})
    (out/'synthetic_validation.json').write_text(json.dumps(probes,indent=2));(out/'history.json').write_text(json.dumps(history,indent=2));(out/'comparison.json').write_text(json.dumps(final,indent=2))
    return status('pilot_complete',selected_epoch=chosen['epoch'],validation=final,
                  beats_copy_on_both_proxy_metrics=all(final['soft_fusion'][k]<final['reference_copy'][k] for k in ['all_mae','cloud_mae']),
                  notice='Development pilot only. Can alter unflagged pixels. Proxy targets differ in date. Synthetic probes are not real ground-change validation. No website export or independent-test tuning.')
