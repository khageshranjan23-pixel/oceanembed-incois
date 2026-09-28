"""Daily harmonized NetCDF -> memory-mappable cube. No targets needed after 2022."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from .science import CHANNELS,DEPTHS

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--start',required=True);p.add_argument('--end',required=True);a=p.parse_args()
    src,out=Path(a.input),Path(a.output);out.mkdir(parents=True,exist_ok=True)
    dates=pd.date_range(a.start,a.end);lat=np.arange(5,30.001,.25);lon=np.arange(45,105.001,.25)
    n,h,w=len(dates),len(lat),len(lon)
    x=np.lib.format.open_memmap(out/'x.npy',mode='w+',dtype='f4',shape=(n,7,h,w));x[:]=np.nan
    support=np.lib.format.open_memmap(out/'support.npy',mode='w+',dtype='f4',shape=x.shape);support[:]=0
    age=np.lib.format.open_memmap(out/'age.npy',mode='w+',dtype='f4',shape=x.shape);age[:]=0
    y=np.lib.format.open_memmap(out/'y.npy',mode='w+',dtype='f4',shape=(n,15,h,w));y[:]=np.nan
    with xr.open_dataset(src/'bathymetry.nc') as b:
        np.testing.assert_allclose(b.latitude,lat);np.testing.assert_allclose(b.longitude,lon)
        bottom=b.bottom_depth.values
    # Conservative cell-level depth mask, plus target finite mask per day.
    wet=np.isfinite(bottom)[None]&(bottom[None]>DEPTHS[:,None,None])
    np.save(out/'wet.npy',wet)
    for i,d in enumerate(dates):
        for k,name in enumerate(CHANNELS+['teacher']):
            f=src/name/f'{d:%Y-%m-%d}.nc'
            if not f.exists(): continue
            if name=='teacher' and d.year>2022: raise ValueError('Do not pack blind-period teacher targets')
            with xr.open_dataset(f) as ds:
                np.testing.assert_allclose(ds.latitude,lat);np.testing.assert_allclose(ds.longitude,lon)
                if name=='teacher':
                    np.testing.assert_allclose(ds.depth,DEPTHS);y[i]=np.where(wet,ds.value.values,np.nan)
                else:
                    x[i,k]=np.where(wet[0],ds.value.values,np.nan)
                    support[i,k]=np.where(wet[0],ds.support.values,0)
        if i%100==0: print('Packed',d.date(),flush=True)
    for z in [x,y,support,age]: z.flush()
    np.save(out/'dates.npy',dates.values.astype('datetime64[D]'));np.save(out/'lat.npy',lat);np.save(out/'lon.npy',lon)
    (out/'metadata.json').write_text(json.dumps({'synthetic':False,'mode':'retrospective','channels':CHANNELS,'age_note':'No stale carry-forward applied; zero means same represented day, not known publication latency.'},indent=2))
if __name__=='__main__':main()
