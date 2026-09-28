"""Training-only pointwise ridge baseline using surface, coordinates and season."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from .data import Cube
from .science import DEPTHS

def features(c,i,s):
    x=c.input(i,s,1)[0]
    return x.reshape(32,-1).T

def main():
    p=argparse.ArgumentParser();p.add_argument('--cube',required=True);p.add_argument('--stats',required=True);p.add_argument('--out',required=True);p.add_argument('--alpha',type=float,default=10);p.add_argument('--points-per-day',type=int,default=100);p.add_argument('--start',default='2022-01-01');p.add_argument('--end',default='2022-12-31');a=p.parse_args()
    c=Cube(a.cube);s=dict(np.load(a.stats));rng=np.random.default_rng(17);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);xx=[];yy=[]
    for i in c.indices(2016,2021,1):
        x=features(c,i,s);y=(c.y[i]-c.climatology(i,s)).reshape(15,-1).T
        valid=np.where(np.isfinite(y).any(1))[0]
        if not len(valid):continue
        ix=rng.choice(valid,min(len(valid),a.points_per_day),replace=False);xx.append(x[ix]);yy.append(y[ix])
    x=np.concatenate(xx);y=np.concatenate(yy);models=[]
    for z in range(15):
        ok=np.isfinite(y[:,z]);m=Ridge(alpha=a.alpha).fit(x[ok],y[ok,z]);models.append(m)
    np.savez(out/'ridge.npz',coef=np.array([m.coef_ for m in models]),intercept=np.array([m.intercept_ for m in models]))
    (out/'manifest.json').write_text(json.dumps({**c.meta,'runs':[{'name':'ridge','alpha':a.alpha}],'drop':None,'base_only':True}))
    for i,d in enumerate(c.dates):
        if not np.datetime64(a.start)<=d<=np.datetime64(a.end):continue
        clim=c.climatology(i,s);f=features(c,i,s);pred=np.array([m.predict(f).reshape(len(c.lat),len(c.lon)) for m in models])+clim;pred=np.where(c.wet,pred,np.nan)
        np.savez_compressed(out/f'{d}.npz',temperature=pred,scale=np.full_like(pred,np.nan),climatology=clim,lat=c.lat,lon=c.lon,depth=DEPTHS)
if __name__=='__main__':main()
