import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from .data import Cube
from .train import load_model
from .model import Observer
from .observer import aux_fields,apply_observer
from .science import DEPTHS,CHANNELS

def main():
    p=argparse.ArgumentParser();p.add_argument('--cube',required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--start',required=True);p.add_argument('--end',required=True);p.add_argument('--device',default='cpu');p.add_argument('--drop',choices=CHANNELS);p.add_argument('--base-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);c=Cube(a.cube);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);models=[]
    for run in a.runs:
        model,s,conf=load_model(run,a.device);obs=None
        if not a.base_only:
            if not (Path(run)/'observer.pt').exists():raise ValueError('Observer absent; use --base-only, which exports no calibrated intervals')
            obs=Observer(conf['width']).to(a.device);obs.load_state_dict(torch.load(Path(run)/'observer.pt',map_location=a.device,weights_only=True));obs.eval()
        models.append((model,s,conf,obs))
    allowed=[set(c.indices(1900,2200,m[2]['history'],False)) for m in models]
    signature={'runs':[{'name':str(Path(r)), 'checkpoint_sha256':hashlib.sha256((Path(r)/'best.pt').read_bytes()).hexdigest(),'stats_sha256':hashlib.sha256((Path(r)/'stats.npz').read_bytes()).hexdigest(),'model_config_sha256':hashlib.sha256((Path(r)/'model.json').read_bytes()).hexdigest(),'observer_sha256':None if a.base_only else hashlib.sha256((Path(r)/'observer.pt').read_bytes()).hexdigest()} for r in a.runs],'drop':a.drop,'base_only':a.base_only}
    (out/'manifest.json').write_text(json.dumps({**c.meta,**signature,'intervals_calibrated':False,'temperature':'potential temperature referenced to 0 dbar','zero_depth':'nominal near-surface teacher level'},indent=2))
    for i,date in enumerate(c.dates):
        if not(np.datetime64(a.start)<=date<=np.datetime64(a.end)) or not all(i in ids for ids in allowed):continue
        means=[];variances=[];clims=[]
        with torch.no_grad():
            for model,s,conf,obs in models:
                an,b,feat=model(torch.from_numpy(c.input(i,s,conf['history'],None if a.drop is None else CHANNELS.index(a.drop))[None]).to(a.device))
                clim=c.climatology(i,s);base=an[0].cpu().numpy()+clim
                if obs is not None:
                    delta,scale=apply_observer(obs,feat[0].cpu().numpy(),aux_fields(c,i,base),a.device);mu=base+delta
                    variance=2*scale**2
                else:mu=base;variance=np.full_like(base,np.nan)
                means.append(mu);variances.append(variance);clims.append(clim)
        mean=np.mean(means,0);variance=np.mean(variances,0)+np.var(means,axis=0)
        valid=c.wet & np.logical_and.reduce([m[1]['train_count']>0 for m in models])
        mean=np.where(valid,mean,np.nan);std=np.where(valid,np.sqrt(variance),np.nan)
        file=out/f'{date}.npz';tmp=out/f'{date}.tmp.npz'
        input_valid=np.isfinite(c.x[i]).copy()
        if a.drop is not None:input_valid[CHANNELS.index(a.drop)]=False
        np.savez_compressed(tmp,input_valid=input_valid,input_support=np.asarray(c.support[i]),input_age_hours=np.asarray(c.age[i]),temperature=mean.astype('f4'),scale=std.astype('f4'),climatology=np.mean(clims,0).astype('f4'),lat=c.lat,lon=c.lon,depth=DEPTHS)
        tmp.replace(file);print('Predicted',date,flush=True)
if __name__=='__main__':main()
