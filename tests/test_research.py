import importlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from oceanembed.research import signature

@pytest.fixture
def client(tmp_path,monkeypatch):
    from oceanembed import api
    monkeypatch.setenv('OCEANEMBED_RESULTS',str(tmp_path))
    manifest={'synthetic':True,'mode':'synthetic','runs':[{'name':'fixture'}],'drop':None,'base_only':False}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    lat=np.array([5.,6.,7.]);lon=np.array([45.,46.,47.]);depth=np.array([0.,100.,1000.]);temperature=np.ones((3,3,3))*20;temperature[:,0,0]=np.nan
    for date,shift in [('2023-01-01',0),('2023-01-03',2)]:np.savez(tmp_path/f'{date}.npz',temperature=temperature+shift,climatology=np.ones_like(temperature)*19,scale=np.ones_like(temperature)*.5,lat=lat,lon=lon,depth=depth,input_valid=np.ones((7,3,3),bool))
    (tmp_path/'calibration.json').write_text(json.dumps({'q':{'100.0':2.,'1000.0':2.},'signature':signature(manifest)}))
    pd.DataFrame([{'profile_id':'P1','wmo':'123','date':'2023-01-01','lat':6.5,'lon':46.5,'depth':100,'theta0':21,'temperature':20,'climatology':19,'lower':19,'upper':21}]).to_csv(tmp_path/'pairs.csv',index=False)
    importlib.reload(api);return TestClient(api.app)

def test_map_difference_and_width(client):
    d=client.get('/api/research/map',params={'date':'2023-01-03','reference_date':'2023-01-01','depth':100}).json();assert d['values'][1][1]==2
    d=client.get('/api/research/map',params={'date':'2023-01-01','variable':'interval_width','depth':100}).json();assert d['values'][1][1]==2
    d=client.get('/api/research/map',params={'date':'2023-01-01','variable':'interval_width','depth':0}).json();assert d['values'][1][1] is None

def test_time_gap(client):
    r=client.get('/api/research/time-depth',params={'start':'2023-01-01','end':'2023-01-03','lat':6.5,'lon':46.5});assert r.status_code==200
    d=r.json();assert d['dates'][1]=='2023-01-02';assert d['values'][1][1] is None

def test_profile_validation_exports(client):
    r=client.get('/api/research/profile',params={'date':'2023-01-01','lat':6.5,'lon':46.5}).json();assert r['lower']==[None,19.,19.]
    v=client.get('/api/research/validation').json();assert v['depths'][0]['bias']==-1.;assert v['depths'][0]['coverage']==1
    assert client.get('/api/research/profile.csv',params={'date':'2023-01-01','lat':6.5,'lon':46.5}).status_code==200
    r=client.get('/api/research/download.nc',params={'date':'2023-01-01'});assert r.status_code==200;assert r.content[:3]==b'CDF'
    assert client.get('/api/research/transect',params={'date':'2023-01-01','lat1':6.1,'lon1':46.1,'lat2':6.8,'lon2':46.8}).status_code==200

def test_invalid_input_and_missing_pairs(client):
    assert client.get('/api/research/map',params={'date':'../../etc/passwd'}).status_code==400
    assert client.get('/api/research/map',params={'date':'2023-01-01','depth':99}).status_code==400
    assert client.get('/api/research/profile',params={'date':'2023-01-01','lat':99,'lon':46}).status_code==400
    assert client.get('/api/research/transect',params={'date':'2023-01-01','points':99999}).status_code==422
