"""Conservative experimental gap filling with separately evaluated references."""
import numpy as np
from scipy.ndimage import binary_erosion,sobel

def alignment_check(a,b,clear):
    support=binary_erosion(clear,iterations=4);support[:4]=False;support[-4:]=False;support[:,:4]=False;support[:,-4:]=False
    if support.sum()<512:return {'accepted':False,'reason':'insufficient_clear_overlap','pixels':int(support.sum())}
    def edge(x):
        gray=x.mean(0);return np.hypot(sobel(gray,0),sobel(gray,1))
    x=edge(a)[support];y=edge(b);scores=[]
    if x.std()<1e-6:return {'accepted':False,'reason':'low_texture'}
    for dy in range(-3,4):
        for dx in range(-3,4):
            q=np.roll(y,(dy,dx),(0,1))[support]
            scores.append((float(np.corrcoef(x,q)[0,1]) if q.std()>1e-6 else -1,dy,dx))
    scores.sort(reverse=True);best=scores[0];margin=best[0]-scores[1][0]
    return {'accepted':best[1:]==(0,0) and best[0]>=.5 and margin>=.05,'reason':'integer_translation_screen','shift_dy_dx':list(best[1:]),'correlation':best[0],'margin':margin,'pixels':int(support.sum()),'notice':'Heuristic screen, not verified subpixel registration. No warping applied.'}

def plan(target,candidates,anchor_index):
    """Retain baseline anchor; add only screened references in previously unsupported pixels."""
    anchor=candidates[anchor_index];mask=target['cloud']&target['usable']
    source=np.full(mask.shape,-1,np.int16);checks=[]
    initial=mask&~anchor['scene']['risk'];source[initial]=anchor_index
    for index,c in sorted(enumerate(candidates),key=lambda pair:pair[1]['age_days']):
        if index==anchor_index:continue
        b=c['scene']
        if not 0<c['age_days']<=60 or b['grid']!=target['grid']:checks.append({'index':index,'accepted':False,'reason':'date_or_grid'});continue
        new=mask&(source<0)&~b['risk']
        if not new.any():continue
        overlap=~target['risk']&~b['risk']
        check=alignment_check(target['rgb'],b['rgb'],overlap);basis='target'
        if check.get('reason') in ['insufficient_clear_overlap','low_texture']:
            check=alignment_check(anchor['scene']['rgb'],b['rgb'],~anchor['scene']['risk']&~b['risk']);basis='baseline_reference'
        check.update(index=index,basis=basis,potential_added_pixels=int(new.sum()))
        if check['accepted']:source[new]=index
        checks.append(check)
    return source,checks

def reconstruct(target,candidates,anchor_index,infer):
    source,checks=plan(target,candidates,anchor_index);x=target['rgb'];result=x.copy();reference=np.zeros_like(x);mix=np.zeros_like(x)
    for index in np.unique(source[source>=0]):
        c=candidates[int(index)];b=c['scene'];valid=target['usable']&~b['risk']
        pred,_,weights=infer(x,b['rgb'],target['cloud']&target['usable'],valid,c['age_days'])
        if pred.shape!=x.shape or not np.isfinite(pred).all():raise ValueError('Invalid model output')
        use=source==index;result[:,use]=pred[:,use];reference[:,use]=b['rgb'][:,use]
        if weights is not None:mix[:,use]=weights[:,use]
    return result,reference,mix,source,checks
