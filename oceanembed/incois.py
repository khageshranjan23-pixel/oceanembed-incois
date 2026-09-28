"""Monthly 1-degree comparison; refuses an unverified temperature definition."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from .science import remap_rect,metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--reference',required=True);p.add_argument('--out',required=True);p.add_argument('--temperature-kind',choices=['potential','in_situ'],required=True);p.add_argument('--definition-source',required=True);p.add_argument('--min-days',type=int,default=25);a=p.parse_args()
    d=xr.open_dataset(a.reference).rename({'ZAX':'depth'})
    d=d.sel(latitude=slice(5,30),longitude=slice(45,105),depth=slice(5,1000))
    rows=[]
    for time in d.time.values:
        month=pd.Timestamp(time).strftime('%Y-%m');files=sorted(Path(a.predictions).glob(month+'-*.npz'))
        if len(files)<a.min_days:continue
        fields=[]
        for file in files:
            with np.load(file) as f:
                field=xr.DataArray(f['temperature'],dims=['depth','latitude','longitude'],coords={'depth':f['depth'],'latitude':f['lat'],'longitude':f['lon']}).interp(depth=d.depth).values
                mapped,_=remap_rect(field,f['lat'],f['lon'],d.latitude.values,d.longitude.values,.8);fields.append(mapped)
        stack=np.array(fields);count=np.isfinite(stack).sum(0);mean=np.divide(np.nansum(stack,axis=0),count,out=np.full(stack.shape[1:],np.nan),where=count>=a.min_days)
        obs=d.TEMP.sel(time=time).transpose('depth','latitude','longitude').values
        if a.temperature_kind=='in_situ':
            if 'SAL' not in d:raise ValueError('Salinity needed for temperature conversion')
            sal=d.SAL.sel(time=time).transpose('depth','latitude','longitude').values
            zz,lat,lon=np.meshgrid(d.depth.values,d.latitude.values,d.longitude.values,indexing='ij')
            pres=gsw.p_from_z(-zz,lat);sa=gsw.SA_from_SP(sal,pres,lon,lat);obs=gsw.pt0_from_t(sa,obs,pres)
        for iz,z in enumerate(d.depth.values):
            m=metrics(obs[iz],mean[iz]);rows.append({'month':month,'depth':float(z),**m})
    d.close()
    if not rows:raise ValueError('No supported monthly comparisons')
    dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(dest/'monthly_depth_metrics.csv',index=False)
    (dest/'metadata.json').write_text(json.dumps({'temperature_kind':a.temperature_kind,'definition_source':a.definition_source,'note':'Scale-matched analysed-product comparison; no 0m validation; minimum-day threshold applies per cell.'},indent=2))
if __name__=='__main__':main()
