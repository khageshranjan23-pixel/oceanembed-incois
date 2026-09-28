"""Fit the observer on training Argo; select only on the 2022 selection partition."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .science import DEPTHS,collocate,stencil
from .data import Cube
from .train import load_model
from .model import Observer

def aux_fields(c,i,base):
    h,w=base.shape[-2:];lat,lon=np.meshgrid((c.lat-17.5)/12.5,(c.lon-75)/30,indexing='ij')
    out=[]
    for z,d in enumerate(DEPTHS):
        out.append(np.stack([np.full((h,w),d/1000),base[z]/30,lat,lon,np.full((h,w),c.calendar[i,1]),np.full((h,w),c.calendar[i,2]),np.nan_to_num(c.support[i]).mean(0)],-1))
    return np.array(out,dtype='f4')

def apply_observer(obs,feature,aux,device):
    h,w=feature.shape[-2:];f=feature.transpose(1,2,0);delta=[];scale=[]
    with torch.no_grad():
        for z in range(15):
            d,b=obs(torch.from_numpy(f.reshape(-1,f.shape[-1]).copy()).to(device),torch.from_numpy(aux[z].reshape(-1,7)).to(device))
            delta.append(d.cpu().numpy().reshape(h,w));scale.append(b.cpu().numpy().reshape(h,w))
    return np.array(delta),np.array(scale)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cube',required=True);p.add_argument('--run',required=True);p.add_argument('--argo',required=True);p.add_argument('--epochs',type=int,default=100);p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(17);run=Path(a.run);c=Cube(a.cube);model,stats,conf=load_model(run,a.device)
    table=pd.read_csv(a.argo,dtype={'wmo':str});table=table[table.split.isin(['train','selection'])].copy()
    years=pd.to_datetime(table.date).dt.year
    if not (((table.split=='train')&years.between(2016,2021))|((table.split=='selection')&(years==2022))).all():pass
    lookup={str(d):i for i,d in enumerate(c.dates)};allowed=set(c.indices(2016,2022,conf['history'],need_target=False));records=[]
    with torch.no_grad():
        for date,group in table.groupby('date'):
            i=lookup.get(date)
            if i is None or i not in allowed:continue
            an,_,features=model(torch.from_numpy(c.input(i,stats,conf['history'])[None]).to(a.device))
            base=an[0].cpu().numpy()+c.climatology(i,stats);base=np.where(c.wet,base,np.nan)
            feat=features[0].cpu().numpy();aux=aux_fields(c,i,base)
            for row in group.itertuples():
                z=np.where(DEPTHS==row.depth)[0]
                if not len(z):continue
                z=z[0]
                try:
                    b=float(collocate(base[z],c.lat,c.lon,row.lat,row.lon))
                    (jj,ii),ww=stencil(c.lat,c.lon,row.lat,row.lon)
                    f=np.stack([feat[:,jj,ii],feat[:,jj,ii+1],feat[:,jj+1,ii],feat[:,jj+1,ii+1]])
                    u=np.stack([aux[z,jj,ii],aux[z,jj,ii+1],aux[z,jj+1,ii],aux[z,jj+1,ii+1]])
                except ValueError:continue
                if np.isfinite(b) and np.isfinite(f).all() and np.isfinite(u).all():records.append((f,u,row.theta0-b,row.split,row.profile_id,ww))
    if not records:raise ValueError('No collocated observer examples')
    f=torch.tensor(np.stack([r[0] for r in records]),device=a.device);u=torch.tensor(np.stack([r[1] for r in records]),device=a.device);y=torch.tensor([r[2] for r in records],dtype=torch.float32,device=a.device)
    bilinear=torch.tensor(np.stack([r[5] for r in records]),dtype=torch.float32,device=a.device)
    split=np.array([r[3] for r in records]);profiles=np.array([r[4] for r in records]);tr=np.where(split=='train')[0];va=np.where(split=='selection')[0]
    if not len(tr) or not len(va):raise ValueError('Need both training and selection profiles')
    counts=pd.Series(profiles[tr]).value_counts();weights=torch.tensor([1/counts[p] for p in profiles[tr]],device=a.device,dtype=torch.float32);weights/=weights.mean()
    obs=Observer(conf['width']).to(a.device);opt=torch.optim.AdamW(obs.parameters(),lr=3e-4,weight_decay=1e-3);best=float('inf')
    for epoch in range(a.epochs):
        obs.train();perm=np.random.permutation(len(tr))
        for start in range(0,len(tr),512):
            ix=perm[start:start+512];ind=tr[ix];delta,b=obs(f[ind],u[ind]);delta=(delta*bilinear[ind]).sum(1);b=(b*bilinear[ind]).sum(1);loss=((abs(y[ind]-delta)/b+torch.log(2*b)+.01*delta**2)*weights[ix]).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(obs.parameters(),1);opt.step()
        obs.eval()
        with torch.no_grad():delta,_=obs(f[va],u[va]);delta=(delta*bilinear[va]).sum(1);score=float(torch.sqrt(((y[va]-delta)**2).mean()))
        if score<best:best=score;torch.save(obs.state_dict(),run/'observer.pt')
        if epoch%10==0:print('Observer',epoch,'selection residual RMSE',score,flush=True)
    (run/'observer.json').write_text(json.dumps({'training_rows':len(tr),'selection_rows':len(va),'selection_rmse':best,'note':'Evaluate base and corrected predictions on identical profiles before retaining adapter.'},indent=2))
if __name__=='__main__':main()
