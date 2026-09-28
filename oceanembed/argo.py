"""Strict delayed-mode GDAC profile conversion. Does not download profile files."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from .science import DEPTHS,interpolate_profile

def text(v):
    a=np.asarray(v).reshape(-1)
    return ''.join(i.decode() if isinstance(i,bytes) else str(i) for i in a).strip()

def partition(wmo,year):
    # Stable WMO assignment within 2022. Future-time test may include earlier floats.
    if 2016<=year<=2021:return 'train'
    if year==2022:return 'selection' if int(hashlib.sha256(wmo.encode()).hexdigest()[:8],16)%2==0 else 'calibration'
    if 2023<=year<=2025:return 'test'
    return 'unused'

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--out',required=True);p.add_argument('--temp-error',type=float,default=.05);p.add_argument('--sal-error',type=float,default=.05);p.add_argument('--pres-error',type=float,default=20);a=p.parse_args()
    rows=[];seen=set();audit={'accepted_profiles':0,'rejected_profiles':0,'duplicate_profiles':0}
    for file in sorted(Path(a.directory).rglob('*.nc')):
        with xr.open_dataset(file) as ds:
            if 'N_PROF' not in ds.dims:continue
            for i in range(ds.sizes['N_PROF']):
                d=ds.isel(N_PROF=i);wmo=text(d.PLATFORM_NUMBER.values)
                key=f'{wmo}_{text(d.CYCLE_NUMBER.values)}_{text(d.DIRECTION.values)}'
                if key in seen:audit['duplicate_profiles']+=1;continue
                # R/A rejected: primary benchmark uses D core profiles only.
                if text(d.DATA_MODE.values)!='D' or text(d.POSITION_QC.values)!='1' or text(d.JULD_QC.values)!='1':audit['rejected_profiles']+=1;continue
                t=pd.Timestamp(d.JULD.values);lat=float(d.LATITUDE);lon=float(d.LONGITUDE);lon=(lon+180)%360-180
                if pd.isna(t) or not(5<=lat<=30 and 45<=lon<=105):continue
                split=partition(wmo,t.year)
                if split=='unused':continue
                values=[];good=None
                for name,limit in [('PRES',a.pres_error),('TEMP',a.temp_error),('PSAL',a.sal_error)]:
                    var=name+'_ADJUSTED';qc=var+'_QC';err=var+'_ERROR'
                    if any(v not in d for v in [var,qc,err]):good=None;break
                    v=np.asarray(d[var].values,float);q=np.asarray(d[qc].values).astype('U');e=np.asarray(d[err].values,float)
                    m=np.isfinite(v)&(q=='1')&np.isfinite(e)&(e>=0)&(e<=limit)
                    good=m if good is None else good&m;values.append(v)
                if good is None or good.sum()<2:audit['rejected_profiles']+=1;continue
                pres,temp,sal=[v[good] for v in values]
                if np.any(pres<0) or np.any(sal<0):audit['rejected_profiles']+=1;continue
                sa=gsw.SA_from_SP(sal,pres,lon,lat);theta=gsw.pt0_from_t(sa,temp,pres);depth=-gsw.z_from_p(pres,lat)
                out=interpolate_profile(depth,theta);out[0]=np.nan
                if not np.isfinite(out).any():audit['rejected_profiles']+=1;continue
                seen.add(key);audit['accepted_profiles']+=1
                for z,v in zip(DEPTHS,out):
                    if np.isfinite(v):rows.append(dict(profile_id=key,wmo=wmo,date=f'{t:%Y-%m-%d}',timestamp=t.isoformat(),lat=lat,lon=lon,depth=float(z),theta0=float(v),split=split,source=file.name))
    if not rows:raise ValueError('No valid profiles; inspect QC/data mode and region')
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(path,index=False)
    path.with_suffix('.audit.json').write_text(json.dumps({**audit,'config':vars(a),'note':'Error/gap thresholds are engineering defaults, not universal Argo QC rules.'},indent=2))
if __name__=='__main__':main()
