"""Explicit support handling and truth-based metrics for offline experiments."""
import numpy as np
from scipy.ndimage import uniform_filter,minimum_filter

def synthetic_probability(mask,kind):
    """Model-space CLD: 0..1, not the provider's 0..100 representation."""
    if kind not in ['identity','thin','opaque','shadow']:raise ValueError('Unknown corruption')
    mask=np.asarray(mask)
    if not np.isin(mask,[0,1]).all():raise ValueError('Expected binary mask')
    return mask.astype('float32')*(0 if kind in ['identity','shadow'] else 1)

def guarded_output(original,estimate,flagged,target_valid,reference_valid):
    original=np.asarray(original);estimate=np.asarray(estimate)
    if original.shape!=estimate.shape or original.ndim!=3 or original.shape[0]!=3:raise ValueError('RGB dimensions differ')
    masks=[np.asarray(m) for m in (flagged,target_valid,reference_valid)]
    if any(m.shape!=original.shape[1:] or not np.isin(m,[0,1]).all() for m in masks):raise ValueError('Invalid mask shape or values')
    if not np.isfinite(original).all() or not np.isfinite(estimate).all():raise ValueError('Nonfinite RGB')
    flagged,valid,support=[m.astype(bool) for m in masks]
    edit=flagged&valid&support
    return np.where(edit[None],estimate.clip(0,1),original),{'edit':edit,'unresolved':flagged&valid&~support,'preserved':~edit}

def temporal_reference(references,target_grid,target_valid):
    """Nearest usable earlier pixel, never averaging cloudy samples.

    Inputs must already have checked alignment; equal grids alone do not prove it.
    Returned image is a multi-date reference, NOT selected-date reconstruction.
    """
    valid=np.asarray(target_valid,dtype=bool);rgb=np.zeros((3,)+valid.shape,np.float32)
    chosen=np.full(valid.shape,-1,np.int32);age=np.full(valid.shape,np.nan,np.float32)
    for index,r in sorted(enumerate(references),key=lambda pair:pair[1]['age_days']):
        if not 0<r['age_days']<=60:raise ValueError('Reference must be earlier and within 60 days')
        if r['grid']!=target_grid:raise ValueError('Reference grid differs')
        if r.get('alignment_checked') is not True:raise ValueError('Alignment has not been checked')
        x=np.asarray(r['rgb']);clear=np.asarray(r['clear'])
        if x.shape!=rgb.shape or clear.shape!=valid.shape or not np.isfinite(x).all() or not np.isin(clear,[0,1]).all():raise ValueError('Invalid reference')
        take=valid&clear.astype(bool)&(chosen<0)
        rgb[:,take]=x[:,take];chosen[take]=index;age[take]=r['age_days']
    return rgb,chosen>=0,chosen,age

def truth_metrics(pred,truth,valid,cloud,truth_kind):
    if truth_kind not in ['synthetic_exact','verified_same_date']:raise ValueError('Do not label different-date proxy as exact truth')
    pred=np.asarray(pred,dtype=float);truth=np.asarray(truth,dtype=float)
    if pred.shape!=truth.shape or pred.ndim!=3 or pred.shape[0]!=3:raise ValueError('RGB dimensions differ')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all() or min(pred.min(),truth.min())<0 or max(pred.max(),truth.max())>1:raise ValueError('Expected finite normalized RGB')
    valid=np.asarray(valid,dtype=bool);cloud=np.asarray(cloud,dtype=bool)
    if valid.shape!=pred.shape[1:] or cloud.shape!=valid.shape:raise ValueError('Mask dimensions differ')
    def region(mask):
        if not mask.any():return {'mae':None,'psnr':None,'pixels':0}
        error=pred[:,mask]-truth[:,mask];mse=float(np.mean(error**2))
        return {'mae':float(np.abs(error).mean()),'psnr':float(-10*np.log10(mse)) if mse else 'infinity','pixels':int(mask.sum())}
    # Local SSIM, 7x7 uniform windows; omit windows touching invalid pixels.
    ux=uniform_filter(pred,size=(1,7,7));uy=uniform_filter(truth,size=(1,7,7))
    vx=np.maximum(0,uniform_filter(pred*pred,size=(1,7,7))-ux*ux)
    vy=np.maximum(0,uniform_filter(truth*truth,size=(1,7,7))-uy*uy)
    cov=uniform_filter(pred*truth,size=(1,7,7))-ux*uy
    score=((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    support=minimum_filter(valid.astype('uint8'),size=7,mode='constant',cval=0)>0
    return {'truth_kind':truth_kind,'all':region(valid),'cloud':region(valid&cloud),'clear':region(valid&~cloud),'ssim':float(score[:,support].mean()) if support.any() else None,'ssim_definition':'7x7 uniform windows, population covariance, data range 1; valid full windows only'}
