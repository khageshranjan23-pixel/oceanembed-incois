"""Small clearly labelled plumbing fixture; never scientific evidence."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .science import DEPTHS,CHANNELS

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/smoke');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(7)
    dates=np.concatenate([pd.date_range(f'{yr}-06-01',periods=90).values for yr in [2020,2022,2023]])
    lat=np.linspace(5,30,16);lon=np.linspace(45,105,24);n=len(dates);h=len(lat);w=len(lon)
    xx,yy=np.meshgrid(lon,lat);ocean=np.ones((h,w),bool)
    # Coarse context mask for synthetic fixtures only, not scientific bathymetry.
    land_file=Path(__file__).resolve().parents[1]/'web'/'land.json'
    if land_file.exists():
        for poly in json.loads(land_file.read_text()):
            inside=np.zeros_like(ocean);previous=poly[-1]
            for current in poly:
                x1,y1=previous;x2,y2=current
                if y1!=y2:inside ^= ((y1>yy)!=(y2>yy)) & (xx < (x2-x1)*(yy-y1)/(y2-y1)+x1)
                previous=current
            ocean &= ~inside
    else:ocean[:3,:5]=False
    wet=np.broadcast_to(ocean,(15,h,w)).copy()
    x=rng.normal(0,1,(n,7,h,w)).astype('f4')
    x[:,0]+=28; x[:,1]+=34
    y=28-18*(1-np.exp(-DEPTHS[:,None,None]/220))+0.04*(yy-15)[None]
    y=np.broadcast_to(y,(n,15,h,w)).copy()+.35*(x[:,0,None]-28)
    y=y.astype('f4');y[:,:,~ocean]=np.nan;x[:,:,~ocean]=np.nan
    support=np.isfinite(x).astype('f4'); age=np.zeros_like(x)
    for name,v in dict(x=x,y=y,support=support,age=age,wet=wet,lat=lat,lon=lon,dates=dates.astype('datetime64[D]')).items():np.save(out/f'{name}.npy',v)
    (out/'metadata.json').write_text(json.dumps({'synthetic':True,'mode':'synthetic','channels':CHANNELS}))
    rows=[]
    for j,d in enumerate(pd.DatetimeIndex(dates)):
        if d.day<8:continue
        wmo=f'{d.year}{j%6:03d}'
        part='train' if d.year<2022 else ('selection' if j%6<3 else 'calibration') if d.year==2022 else 'test'
        for z,depth in enumerate(DEPTHS):
            if depth==0:continue
            rows.append(dict(profile_id=f'{wmo}_{d:%Y%m%d}',wmo=wmo,date=f'{d:%Y-%m-%d}',lat=float(lat[6]),lon=float(lon[17]),depth=float(depth),theta0=float(y[j,z,6,17]),split=part))
    pd.DataFrame(rows).to_csv(out/'argo.csv',index=False)
    print('SYNTHETIC fixture created; no observational skill can be inferred.')
if __name__=='__main__':main()
