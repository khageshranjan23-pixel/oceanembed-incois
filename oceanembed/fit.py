"""Fit all statistics on 2016-2021 only, streaming spatial/time arrays."""
import argparse,json
from pathlib import Path
import numpy as np
from .data import Cube

def main():
    p=argparse.ArgumentParser();p.add_argument('--cube',required=True);p.add_argument('--out',required=True);p.add_argument('--modes',type=int,default=8);a=p.parse_args()
    if not 1<=a.modes<=15:raise ValueError('modes must be 1..15')
    c=Cube(a.cube);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ids=np.where((c.years>=2016)&(c.years<=2021))[0]
    if not len(ids):raise ValueError('No training years')
    total=np.zeros(7);square=np.zeros(7);count=np.zeros(7)
    h,w=c.x.shape[-2:];clim=np.zeros((5,15,h,w),dtype='f4');train_count=np.zeros((15,h,w),dtype='i4')
    # Each per-depth block is manageable (~200 MB on full 6-year grid).
    design=c.calendar[ids].astype(float)
    for k in range(15):
        vals=np.asarray(c.y[ids,k],dtype=float).reshape(len(ids),-1);valid=np.isfinite(vals)
        train_count[k]=valid.sum(0).reshape(h,w)
        for start in range(0,h*w,512):
            v=vals[:,start:start+512];m=valid[:,start:start+512]
            lhs=np.einsum('ti,tj,tn->nij',design,design,m)+np.eye(5)[None]*1e-3
            rhs=np.einsum('ti,tn->ni',design,np.where(m,v,0))
            coef=np.linalg.solve(lhs,rhs[...,None])[...,0].T
            clim[:,k].reshape(5,-1)[:,start:start+512]=coef
    for i in ids:
        x=np.asarray(c.x[i],float);valid=np.isfinite(x);v=np.where(valid,x,0)
        total+=v.sum((1,2));square+=(v*v).sum((1,2));count+=valid.sum((1,2))
    if np.any(count==0):raise ValueError('A training channel has no valid values')
    mean=total/count;std=np.sqrt(np.maximum(square/count-mean**2,1e-6))
    rng=np.random.default_rng(17);profiles=[]
    for i in ids[::max(1,len(ids)//250)]:
        y=np.array(c.y[i])-np.einsum('k,kzhw->zhw',c.calendar[i],clim)
        flat=y.reshape(15,-1).T;good=flat[np.isfinite(flat).all(1)]
        if len(good):profiles.append(good[rng.choice(len(good),min(200,len(good)),replace=False)])
    if not profiles:raise ValueError('No complete deep-water columns for EOF')
    z=np.concatenate(profiles);pcmean=z.mean(0);_,_,v=np.linalg.svd(z-pcmean,full_matrices=False)
    np.savez(out/'stats.npz',clim=clim,train_count=train_count,xmean=mean.astype('f4'),xstd=std.astype('f4'),eof=v[:a.modes].astype('f4'),pc_mean=pcmean.astype('f4'))
    (out/'fit_manifest.json').write_text(json.dumps({'years':[2016,2021],'synthetic':c.meta['synthetic'],'eof_profiles':len(z),'modes':a.modes},indent=2))
    print(out/'stats.npz')
if __name__=='__main__':main()
