"""Paired WMO-block bootstrap on common collocated observations."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--a',required=True);p.add_argument('--b',required=True);p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,default=1000);a=p.parse_args()
    x=pd.read_csv(a.a,dtype={'wmo':str});y=pd.read_csv(a.b,dtype={'wmo':str});keys=['profile_id','depth']
    if x.duplicated(keys).any() or y.duplicated(keys).any():raise ValueError('Duplicate profile/depth pairs')
    d=x.merge(y,on=keys,suffixes=('_a','_b'),validate='one_to_one');rng=np.random.default_rng(17);report={}
    for z,g in d.groupby('depth'):
        if not np.allclose(g.theta0_a,g.theta0_b):raise ValueError('Reference labels differ')
        ea=(g.temperature_a-g.theta0_a)**2;eb=(g.temperature_b-g.theta0_b)**2
        s=pd.DataFrame({'wmo':g.wmo_a,'a':ea,'b':eb,'n':1}).groupby('wmo').sum();v=s.to_numpy();deltas=[]
        for _ in range(a.repeats):
            sampled=v[rng.integers(len(v),size=len(v))].sum(0);deltas.append(np.sqrt(sampled[0]/sampled[2])-np.sqrt(sampled[1]/sampled[2]))
        report[str(z)]={'n':len(g),'floats':len(v),'rmse_a_minus_b':float(np.sqrt(ea.mean())-np.sqrt(eb.mean())),'bootstrap_95_percent':np.quantile(deltas,[.025,.975]).tolist()}
    Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
