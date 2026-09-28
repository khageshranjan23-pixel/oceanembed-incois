import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .science import seasonal_design

class Cube:
    def __init__(self,path):
        self.path=Path(path)
        for n in ['x','y','support','age','wet','dates','lat','lon']:setattr(self,n,np.load(self.path/f'{n}.npy',mmap_mode='r'))
        self.meta=json.loads((self.path/'metadata.json').read_text())
        self.calendar=seasonal_design(self.dates);self.years=pd.DatetimeIndex(self.dates).year.to_numpy()
        if self.x.shape[1]!=7 or self.y.shape[1]!=15:raise ValueError('Unexpected channels/depths')
        if len(np.unique(self.dates))!=len(self.dates) or np.any(np.diff(self.dates).astype(int)<=0):raise ValueError('Dates must be unique/increasing')
    def indices(self,start,end,history=7,need_target=True):
        out=[]
        for i in np.where((self.years>=start)&(self.years<=end))[0]:
            if i<history-1:continue
            if (self.dates[i]-self.dates[i-history+1]).astype(int)!=history-1:continue
            if need_target and not np.isfinite(self.y[i]).any():continue
            out.append(int(i))
        return out
    def climatology(self,i,stats):return np.einsum('k,kzhw->zhw',self.calendar[i],stats['clim']).astype('f4')
    def input(self,i,stats,history,drop=None):
        xx=np.array(self.x[i-history+1:i+1]);valid=np.isfinite(xx)
        support=np.array(self.support[i-history+1:i+1]);age=np.array(self.age[i-history+1:i+1])
        if drop is not None:valid[:,drop]=False;support[:,drop]=0
        norm=(xx-stats['xmean'][None,:,None,None])/stats['xstd'][None,:,None,None]
        norm=np.where(valid,norm,0)
        # Fixed clipping is an encoding choice, not a permission to accept indefinitely stale inputs.
        age=np.clip(np.nan_to_num(age,nan=168)/168,0,1)
        h,w=xx.shape[-2:];lat,lon=np.meshgrid((self.lat-17.5)/12.5,(self.lon-75)/30,indexing='ij')
        extra=np.stack([lat,lon,np.full((h,w),self.calendar[i,1]),np.full((h,w),self.calendar[i,2])]).astype('f4')
        extra=np.broadcast_to(extra,(history,4,h,w))
        return np.concatenate([norm,valid.astype('f4'),support,age,extra],1).astype('f4')

class Windows(Dataset):
    def __init__(self,cube,stats,indices,history=7,patch=None,augment=False):
        self.c,self.s,self.ids,self.history,self.patch,self.augment=cube,stats,indices,history,patch,augment
    def __len__(self):return len(self.ids)
    def __getitem__(self,k):
        i=self.ids[k];x=self.c.input(i,self.s,self.history);clim=self.c.climatology(i,self.s)
        y=np.array(self.c.y[i])-clim;mask=np.isfinite(y)&self.c.wet
        if self.patch:
            h,w=y.shape[-2:];p=min(self.patch,h,w)
            # Retry to avoid entirely land-centred batches.
            for _ in range(20):
                r=np.random.randint(h-p+1);c=np.random.randint(w-p+1)
                if mask[:,r:r+p,c:c+p].any():break
            x=x[...,r:r+p,c:c+p];y=y[...,r:r+p,c:c+p];mask=mask[...,r:r+p,c:c+p]
        if self.augment and np.random.random()<.25:
            ch=np.random.randint(7);x[:,ch]=0;x[:,7+ch]=0;x[:,14+ch]=0
        return torch.from_numpy(x),torch.from_numpy(np.nan_to_num(y)),torch.from_numpy(mask)
