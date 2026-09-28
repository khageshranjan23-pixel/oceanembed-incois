import json,sys
import numpy as np
import xarray as xr
from oceanembed.science import DEPTHS,CHANNELS
from oceanembed import prepare,pack,argo

def test_real_file_ingestion_contract(tmp_path,monkeypatch):
    lat=np.arange(5,30.001,.25);lon=np.arange(45,105.001,.25);h,w=len(lat),len(lon)
    native=np.r_[.5,DEPTHS[1:],1100.]
    d=xr.Dataset(coords={'time':[np.datetime64('2020-06-01')],'latitude':lat,'longitude':lon,'depth':native})
    for name in CHANNELS:d[name]=xr.DataArray(np.ones((1,h,w),dtype='f4')*10,dims=('time','latitude','longitude'),attrs={'units':'test_units'})
    vals=np.broadcast_to((28-native/100)[None,:,None,None],(1,len(native),h,w)).copy()
    d['thetao']=xr.DataArray(vals,dims=('time','depth','latitude','longitude'),attrs={'units':'degree_Celsius'})
    d['bottom']=xr.DataArray(np.full((h,w),1200.),dims=('latitude','longitude'),attrs={'units':'m'})
    f=tmp_path/'raw.nc';d.to_netcdf(f);dest=tmp_path/'harmonized';cfg={'output':str(dest),'sources':{}}
    for name in CHANNELS+['teacher','bathymetry']:
        cfg['sources'][name]={'reviewed':True,'files':str(f),'variable':name if name in CHANNELS else 'thetao' if name=='teacher' else 'bottom','expected_units':'test_units' if name in CHANNELS else 'degree_Celsius' if name=='teacher' else 'm','min_samples_per_day':1}
    spec=tmp_path/'spec.json';spec.write_text(json.dumps(cfg));monkeypatch.setattr(sys,'argv',['prepare','--spec',str(spec)]);prepare.main()
    out=tmp_path/'cube';monkeypatch.setattr(sys,'argv',['pack','--input',str(dest),'--output',str(out),'--start','2020-06-01','--end','2020-06-01']);pack.main()
    y=np.load(out/'y.npy');assert y.shape==(1,15,101,241)
    np.testing.assert_allclose(y[0,0],27.995,atol=1e-5);np.testing.assert_allclose(y[0,-1],18,atol=1e-5)

def test_raw_argo_profile_qc(tmp_path,monkeypatch):
    p=np.array([1,5,10,20,30,50,75,100,150,200,300,500,700,1000,1100.]);n=len(p)
    d=xr.Dataset({'PLATFORM_NUMBER':('N_PROF',[b'1234567']),'CYCLE_NUMBER':('N_PROF',[1]),'DIRECTION':('N_PROF',[b'A']),'DATA_MODE':('N_PROF',[b'D']),'POSITION_QC':('N_PROF',[b'1']),'JULD_QC':('N_PROF',[b'1']),'JULD':('N_PROF',[np.datetime64('2020-06-01')]),'LATITUDE':('N_PROF',[15.]),'LONGITUDE':('N_PROF',[80.])})
    for name,value,err in [('PRES',p,1.),('TEMP',28-p/100,.01),('PSAL',np.full(n,35.),.01)]:
        d[name+'_ADJUSTED']=(('N_PROF','N_LEVELS'),value[None]);d[name+'_ADJUSTED_QC']=(('N_PROF','N_LEVELS'),np.full((1,n),b'1'));d[name+'_ADJUSTED_ERROR']=(('N_PROF','N_LEVELS'),np.full((1,n),err))
    raw=tmp_path/'raw';raw.mkdir();d.to_netcdf(raw/'D1234567_001.nc');out=tmp_path/'argo.csv'
    monkeypatch.setattr(sys,'argv',['argo','--directory',str(raw),'--out',str(out)]);argo.main()
    import pandas as pd
    t=pd.read_csv(out);assert len(t)>3;assert t.split.eq('train').all();assert not (t.depth==0).any();assert t.theta0.notna().all()
