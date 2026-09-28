"""Normalize reviewed regular-grid source files to daily target-grid channels.
python -m oceanembed.prepare --spec configs/ingest.json
"""
import argparse,json,glob
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from .science import remap_rect,DEPTHS

def read_spec(spec):
    if not spec.get('reviewed'): raise ValueError('Review names, units, QC and geometry; then set reviewed=true')
    ds=xr.open_mfdataset(sorted(glob.glob(spec['files'])),combine='by_coords') if spec.get('multi_file') else xr.open_dataset(spec['files'])
    if 'swap_dims' in spec: ds=ds.swap_dims(spec['swap_dims'])
    ds=ds.rename(spec.get('rename',{}))
    for c in ['latitude','longitude']:
        if ds[c].ndim!=1: raise ValueError('Only 1-D rectilinear coordinates supported here')
    ds=ds.assign_coords(longitude=((ds.longitude+180)%360)-180).sortby('latitude').sortby('longitude')
    if len(np.unique(ds.longitude))!=len(ds.longitude): raise ValueError('Duplicate longitude')
    da=ds[spec['variable']]
    for dim,index in spec.get('select_index',{}).items(): da=da.isel({dim:index},drop=True)
    source_units=str(da.attrs.get('units',''))
    if source_units!=spec['expected_units']: raise ValueError(f'Units mismatch: {source_units!r}')
    # Arithmetic preserves only values. Product QC must be explicitly supplied.
    for q in spec.get('qc',[]):
        good=ds[q['variable']].isin(q['allowed'])
        for dim,index in spec.get('select_index',{}).items():
            if dim in good.dims: good=good.isel({dim:index},drop=True)
        da=da.where(good)
    da=da*spec.get('scale',1)+spec.get('offset',0)
    if 'valid_range' in spec: da=da.where((da>=spec['valid_range'][0])&(da<=spec['valid_range'][1]))
    return ds,da

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.spec).read_text())
    lat=np.arange(5,30.001,.25);lon=np.arange(45,105.001,.25)
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    for name,spec in cfg['sources'].items():
        ds,da=read_spec(spec)
        if name=='bathymetry':
            da=da.transpose('latitude','longitude')
            # Require positive-down ocean depth, land=NaN. Mixed coast cells rejected at support threshold.
            dep,sup=remap_rect(da.where(da>0).values,ds.latitude.values,ds.longitude.values,lat,lon,spec.get('min_support',0.95))
            xr.Dataset({'bottom_depth':(('latitude','longitude'),dep),'support':(('latitude','longitude'),sup)},coords={'latitude':lat,'longitude':lon}).to_netcdf(out/'bathymetry.nc')
            ds.close();continue
        if 'time' not in da.dims: raise ValueError(f'{name}: no time dimension')
        if len(np.unique(da.time))!=len(da.time): raise ValueError('Duplicate timestamps')
        # Aggregate only supplied samples; minimum per-pixel count is mandatory.
        n=da.resample(time='1D').count(); daily=da.resample(time='1D').mean(skipna=True).where(n>=spec['min_samples_per_day'])
        for t in daily.time.values:
            day=daily.sel(time=t)
            if name=='teacher': day=day.transpose('depth','latitude','longitude')
            else: day=day.transpose('latitude','longitude')
            v,support=remap_rect(day.values,ds.latitude.values,ds.longitude.values,lat,lon,spec.get('min_support',.8))
            coords={'latitude':lat,'longitude':lon}
            if name=='teacher':
                native=np.asarray(day.depth.values)
                if not np.all(np.diff(native)>0) or native[-1]<1000: raise ValueError('Need increasing GLORYS depths bracketing 1000 m')
                va=xr.DataArray(v,dims=('depth','latitude','longitude'),coords={**coords,'depth':native})
                zz=DEPTHS.copy();zz[0]=native[0]
                v=va.interp(depth=zz,method='linear').values
                coords['depth']=DEPTHS;dims=('depth','latitude','longitude')
                # Vertical finite support checked in pack stage.
                result=xr.Dataset({'value':(dims,v)},coords=coords)
            else:
                result=xr.Dataset({'value':(('latitude','longitude'),v),'support':(('latitude','longitude'),support)},coords=coords)
            result.attrs.update({'source':name,'provenance':json.dumps(spec),'mode':'retrospective','synthetic':0})
            dest=out/name;dest.mkdir(exist_ok=True)
            t_val = t.values if hasattr(t, "values") else t
            if isinstance(t_val, np.datetime64) or getattr(t_val, "dtype", None) == "<M8[ns]": ts = pd.to_datetime(t_val)
            else: ts = pd.to_datetime(str(t_val))
            result.to_netcdf(dest/f'{ts:%Y-%m-%d}.nc')
        ds.close()
if __name__=='__main__': main()
