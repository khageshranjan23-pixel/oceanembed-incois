"""From the project root: python run_smoke.py. Requires test dependencies."""
import subprocess,sys,shutil,json,os
from pathlib import Path

def run(*args):subprocess.run([sys.executable,*args],check=True)
run('-m','pytest','-q','tests')
run('-m','oceanembed.synthetic','--output','data/smoke')
run('-m','oceanembed.fit','--cube','data/smoke','--out','runs/smoke','--modes','6')
run('-m','oceanembed.train','--cube','data/smoke','--run','runs/smoke','--epochs','2','--width','8','--batch','8','--patch','16','--device','cpu')
run('-m','oceanembed.observer','--cube','data/smoke','--run','runs/smoke','--argo','data/smoke/argo.csv','--epochs','3')
run('-m','oceanembed.predict','--cube','data/smoke','--runs','runs/smoke','--out','outputs/smoke','--start','2022-06-07','--end','2023-08-29')
run('-m','oceanembed.evaluate','--predictions','outputs/smoke','--argo','data/smoke/argo.csv','--split','calibration','--out','outputs/smoke-calibration')
run('-m','oceanembed.evaluate','--predictions','outputs/smoke','--argo','data/smoke/argo.csv','--split','test','--unlock-test','--calibration','outputs/smoke-calibration/calibration.json','--out','outputs/smoke-evaluation')
shutil.copyfile('outputs/smoke-evaluation/metrics.json','outputs/smoke/metrics.json')
shutil.copyfile('outputs/smoke-evaluation/pairs.csv','outputs/smoke/pairs.csv')
shutil.copyfile('outputs/smoke-calibration/calibration.json','outputs/smoke/calibration.json')
os.environ['OCEANEMBED_RESULTS']='outputs/smoke'
from fastapi.testclient import TestClient
from oceanembed.api import app
client=TestClient(app)
assert client.get('/').status_code==200
assert client.get('/api/meta').json()['synthetic'] is True
dates=client.get('/api/dates').json();assert dates
assert client.get('/api/map',params={'date':dates[0],'depth':100}).status_code==200
assert client.get('/api/profile',params={'date':dates[0],'lat':15,'lon':75}).status_code==200
assert client.get('/api/map',params={'date':'../../etc/passwd'}).status_code==400
assert client.get('/api/metrics').json()['synthetic'] is True
Path('SMOKE_RESULT.json').write_text(json.dumps({'status':'passed','synthetic_only':True,'checks':['unit tests','climatology/EOF','teacher training','observer training','prediction','calibration','test evaluation','API responses']},indent=2))
print('\nSMOKE TEST PASSED. Start: python -m uvicorn oceanembed.api:app --reload')
print('Open http://127.0.0.1:8000 ; synthetic fixture is NOT an accuracy benchmark.')
