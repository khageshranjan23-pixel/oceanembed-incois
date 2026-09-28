"""Provider download helpers: exact product choices stay in reviewed JSON."""
import argparse,json,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    if not cfg.get('reviewed'):raise ValueError('Copy current provider dataset IDs and review request; set reviewed=true')
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    if cfg['provider']=='copernicus':
        import copernicusmarine
        # Credentials are read by the official toolbox, never embedded in this file.
        copernicusmarine.subset(**cfg['request'],output_directory=str(out))
    elif cfg['provider']=='earthdata':
        import earthaccess
        earthaccess.login(persist=True)
        hits=earthaccess.search_data(short_name=cfg['short_name'],bounding_box=tuple(cfg['bbox']),temporal=tuple(cfg['temporal']))
        if not hits:raise ValueError('No matching granules')
        print('Matching granules:',len(hits))
        if len(hits)>cfg.get('maximum_granules',100):raise ValueError('Narrow request or explicitly increase maximum_granules')
        earthaccess.download(hits,str(out))
    else:raise ValueError('Unknown provider')
    manifest={'request':cfg,'files':[]}
    for file in sorted(out.glob('*.nc')):
        digest=hashlib.sha256()
        with file.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
        manifest['files'].append({'name':file.name,'bytes':file.stat().st_size,'sha256':digest.hexdigest()})
    (out/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
