"""Package checked scientific outputs for read-only web serving."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .science import collocate
from .research import signature

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--calibration');p.add_argument('--incois');p.add_argument('--out',required=True);p.add_argument('--start',required=True);p.add_argument('--end',required=True);a=p.parse_args()
    src,ev,out=Path(a.predictions),Path(a.evaluation),Path(a.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose an empty/new bundle folder to avoid mixing model versions')
    manifest=json.loads((src/'manifest.json').read_text());report=json.loads((ev/'metrics.json').read_text());pairs=pd.read_csv(ev/'pairs.csv',dtype={'wmo':str,'profile_id':str})
    if bool(manifest['synthetic'])!=bool(report['synthetic']):raise ValueError('Synthetic/real mismatch')
    if pairs.duplicated(['profile_id','depth']).any():raise ValueError('Duplicate evaluation pairs')
    if 'signature' in report and report['signature']!=signature(manifest):raise ValueError('Evaluation model mismatch')
    checked=0
    for date,g in pairs.groupby('date'):
        path=src/f'{date}.npz'
        if not path.exists():raise ValueError(f'Missing source prediction for evaluation date {date}')
        with np.load(path) as d:
            for r in g.itertuples():
                zz=np.where(d['depth']==r.depth)[0]
                if not len(zz):raise ValueError('Unsupported evaluation depth')
                pred=float(collocate(d['temperature'][zz[0]],d['lat'],d['lon'],r.lat,r.lon))
                if not np.isclose(pred,r.temperature,atol=1e-4,rtol=1e-5):raise ValueError('Published pair predictions differ from source arrays')
                checked+=1
    cal=None
    if a.calibration:
        cal=json.loads(Path(a.calibration).read_text())
        if cal['signature']!=signature(manifest):raise ValueError('Calibration model mismatch')
    files=[f for f in sorted(src.glob('????-??-??.npz')) if a.start<=f.stem<=a.end]
    if not files:raise ValueError('No prediction files selected')
    out.mkdir(parents=True,exist_ok=True)
    for f in files:shutil.copyfile(f,out/f.name)
    for name in ['pairs.csv','metrics.json']:shutil.copyfile(ev/name,out/name)
    if cal is not None:(out/'calibration.json').write_text(json.dumps(cal,indent=2))
    if a.incois:shutil.copytree(a.incois,out/'incois')
    manifest.update({'evaluation_period':[str(pairs.date.min()),str(pairs.date.max())],'export_period':[files[0].stem,files[-1].stem],'evaluation_split':report['split'],'verified_pair_predictions':checked,'intervals_calibrated':cal is not None})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Web bundle: {len(files)} days, {checked} checked evaluation pairs. Synthetic={manifest["synthetic"]}')
if __name__=='__main__':main()
