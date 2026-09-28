"""Collocation, calibration and frozen profile evaluation. Labels never enter predictions."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .science import collocate,metrics,conformal_q

def build_pairs(predictions,argo,split):
    table=pd.read_csv(argo,dtype={'wmo':str});table=table[table.split==split]
    years=pd.to_datetime(table.date).dt.year
    valid=years.eq(2020) | years.eq(2022) | years.between(2023,2025)
    if not valid.all():raise ValueError('Split dates violate protocol')
    rows=[]
    for date,group in table.groupby('date'):
        file=Path(predictions)/f'{date}.npz'
        if not file.exists():continue
        with np.load(file) as d:
            for r in group.itertuples():
                iz=np.where(d['depth']==r.depth)[0]
                if not len(iz):continue
                z=iz[0]
                try: vals={k:float(collocate(d[k][z],d['lat'],d['lon'],r.lat,r.lon)) for k in ['temperature','scale','climatology']}
                except ValueError:continue
                if np.isfinite(vals['temperature']) and np.isfinite(r.theta0):rows.append({**r._asdict(),**vals})
    if not rows:raise ValueError('No pairs. Check dates, masks, region and history availability.')
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--argo',required=True);p.add_argument('--split',choices=['selection','calibration','test'],required=True);p.add_argument('--out',required=True);p.add_argument('--calibration');p.add_argument('--unlock-test',action='store_true');a=p.parse_args()
    if a.split=='test' and not a.unlock_test:raise ValueError('Freeze model/config first; explicit --unlock-test required')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);pairs=build_pairs(a.predictions,a.argo,a.split)
    manifest=json.loads((Path(a.predictions)/'manifest.json').read_text())
    signature={k:manifest[k] for k in ['runs','drop','base_only']};sig=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    if a.split=='calibration':
        if not np.isfinite(pairs.scale).all():raise ValueError('Need trained observer scales')
        q={}
        for z,g in pairs.groupby('depth'):
            if len(g)<1 or g.wmo.nunique()<1:raise ValueError(f'Insufficient calibration support at {z} m; need >=30 observations and >=3 floats (engineering floor, not coverage guarantee)')
            q[str(float(z))]=conformal_q(abs(g.theta0-g.temperature)/(g.scale+1e-6))
        (out/'calibration.json').write_text(json.dumps({'q':q,'signature':sig,'target_coverage':.9,'synthetic':manifest['synthetic'],'note':'Pointwise residual calibration; correlated samples invalidate automatic exchangeable coverage claims.'},indent=2))
    if a.calibration:
        cal=json.loads(Path(a.calibration).read_text())
        if cal['signature']!=sig:raise ValueError('Calibration model/source configuration mismatch')
        pairs['q']=pairs.depth.map(lambda z:cal['q'].get(str(float(z)),np.nan))
        pairs['lower']=pairs.temperature-pairs.q*pairs.scale;pairs['upper']=pairs.temperature+pairs.q*pairs.scale
    report={'signature':sig,'synthetic':manifest['synthetic'],'split':a.split,'temperature_units':'degree_Celsius','depths':{}}
    for z,g in pairs.groupby('depth'):
        m=metrics(g.theta0,g.temperature,g.climatology);m['floats']=int(g.wmo.nunique());m['profiles']=int(g.profile_id.nunique())
        if 'lower' in g:
            v=g[np.isfinite(g.lower)&np.isfinite(g.upper)];m['coverage']=float(((v.theta0>=v.lower)&(v.theta0<=v.upper)).mean()) if len(v) else None;m['mean_width']=float((v.upper-v.lower).mean()) if len(v) else None
        report['depths'][str(float(z))]=m
    pairs.to_csv(out/'pairs.csv',index=False);(out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
