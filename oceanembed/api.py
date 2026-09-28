"""CPU-only serving: reads precomputed results; never imports model/PyTorch."""
import os,json,re
from functools import lru_cache
from pathlib import Path
import numpy as np
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from .science import collocate

ROOT=Path(os.environ.get('OCEANEMBED_RESULTS','outputs/smoke'))
WEB=Path(__file__).resolve().parents[1]/'web'
app=FastAPI(title='OceanEmbed research prototype')

def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.integer):return int(v)
    return v

@lru_cache(maxsize=4)
def read_day(date):
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',date):raise HTTPException(400,'Use YYYY-MM-DD')
    file=ROOT/f'{date}.npz'
    if not file.exists():raise HTTPException(404,'Date not available')
    with np.load(file,allow_pickle=False) as d:return {k:d[k] for k in d.files}

@app.get('/')
def index():return FileResponse(WEB/'index.html')

@app.get('/health')
def health():return {'status':'ok','result_directory_exists':ROOT.exists()}

@app.get('/api/meta')
def meta():
    f=ROOT/'manifest.json'
    if not f.exists():raise HTTPException(503,'Export predictions first')
    return json.loads(f.read_text())

@app.get('/api/dates')
def dates():return [p.stem for p in sorted(ROOT.glob('????-??-??.npz'))]

@app.get('/api/map')
def map_field(date:str,depth:float=100):
    d=read_day(date);z=np.where(d['depth']==depth)[0]
    if not len(z):raise HTTPException(400,'Unsupported depth')
    return clean({'lat':d['lat'],'lon':d['lon'],'temperature':d['temperature'][z[0]],'depths':d['depth']})

@app.get('/api/profile')
def profile(date:str,lat:float,lon:float):
    d=read_day(date)
    try:v=collocate(d['temperature'],d['lat'],d['lon'],lat,lon)
    except ValueError:raise HTTPException(400,'Location outside grid')
    return clean({'depth':d['depth'],'temperature':v,'note':'Strict four-cell interpolation; missing at unsupported coastal/depth locations.'})

@app.get('/api/metrics')
def get_metrics():
    file=ROOT/'metrics.json'
    if not file.exists():return {'status':'not_evaluated','message':'No evaluation published for this export.'}
    return json.loads(file.read_text())

from fastapi.staticfiles import StaticFiles
from .research import install
app.mount("/assets",StaticFiles(directory=WEB),name="assets")
install(app,ROOT,read_day,clean)
