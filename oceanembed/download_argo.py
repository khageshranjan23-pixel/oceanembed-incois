"""Subset official GDAC core-profile index, then fetch selected profile files."""
import argparse,gzip,io,json,urllib.request,time
from pathlib import Path,PurePosixPath
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--start',required=True);p.add_argument('--end',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int,default=100);a=p.parse_args()
    root='https://data-argo.ifremer.fr/';out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(root+'ar_index_global_prof.txt.gz',timeout=120) as r:raw=r.read()
    (out/'ar_index_global_prof.txt.gz').write_bytes(raw)
    df=pd.read_csv(io.BytesIO(gzip.decompress(raw)),comment='#',dtype={'date':str})
    dates=pd.to_datetime(df.date,format='%Y%m%d%H%M%S',errors='coerce')
    m=df.latitude.between(5,30)&df.longitude.between(45,105)&dates.between(pd.Timestamp(a.start),pd.Timestamp(a.end)+pd.Timedelta(days=1)-pd.Timedelta(seconds=1))
    rows=df.loc[m].sort_values('file')
    print('Matching profiles:',len(rows),'Downloading at most:',a.limit)
    rows=rows.head(a.limit);rows.to_csv(out/'selected_index.csv',index=False)
    for rel in rows.file:
        path=PurePosixPath(rel)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Invalid index path')
        dest=out/path;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():continue
        for attempt in range(3):
            try:
                with urllib.request.urlopen(root+'dac/'+rel,timeout=120) as r:data=r.read()
                tmp=dest.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(dest);break
            except Exception:
                if attempt==2:raise
                time.sleep(2**attempt)
    (out/'request.json').write_text(json.dumps(vars(a),indent=2))
if __name__=='__main__':main()
