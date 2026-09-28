"""Research read API. No training, provider credentials or GPU in request path."""
import csv,hashlib,io,json,re
from datetime import date as Date
from pathlib import Path
import numpy as np
from fastapi import HTTPException,Query
from fastapi.responses import Response
from .science import collocate,metrics

def signature(m):
    return hashlib.sha256(json.dumps({k:m[k] for k in ['runs','drop','base_only']},sort_keys=True).encode()).hexdigest()

def install(app,root,read_day,clean):
    def metadata():return json.loads((root/'manifest.json').read_text())
    def calibration():
        f=root/'calibration.json'
        if not f.exists():return None
        c=json.loads(f.read_text())
        if c['signature']!=signature(metadata()):raise HTTPException(409,'Calibration does not match this model configuration')
        return c
    def available(start=None,end=None,limit=366):
        dates=[p.stem for p in sorted(root.glob('????-??-??.npz'))]
        if start: dates=[d for d in dates if d>=start]
        if end: dates=[d for d in dates if d<=end]
        if len(dates)>limit:raise HTTPException(400,f'Select at most {limit} exported days')
        if not dates:raise HTTPException(404,'No exported dates in selection')
        return dates
    def field(d,name):
        if name=='temperature':return d['temperature']
        if name=='anomaly':return np.where(np.isfinite(d['temperature']),d['temperature']-d['climatology'],np.nan)
        if name=='interval_width':
            c=calibration()
            if c is None:raise HTTPException(409,'No matching calibration has been supplied')
            q=np.array([c['q'].get(str(float(z)),np.nan) for z in d['depth']])
            return np.where(np.isfinite(d['temperature']),2*q[:,None,None]*d['scale'],np.nan)
        if name=='input_fraction':
            if 'input_valid' not in d:raise HTTPException(409,'Input quality arrays not present; regenerate predictions')
            a=d['input_valid'].mean(0)
            a=np.where(np.isfinite(d['temperature'][0]),a,np.nan)
            return np.broadcast_to(a,d['temperature'].shape)
        raise HTTPException(400,'Unknown field')
    def depth_index(d,z):
        ids=np.where(d['depth']==z)[0]
        if not len(ids):raise HTTPException(400,'Choose a native output depth')
        return int(ids[0])
    def point(d,lat,lon,name='temperature'):
        try:return collocate(field(d,name),d['lat'],d['lon'],lat,lon)
        except ValueError:raise HTTPException(400,'Point outside export domain')
    def profile_data(date,lat,lon):
        d=read_day(date);temp=point(d,lat,lon);clim=collocate(d['climatology'],d['lat'],d['lon'],lat,lon)
        lo=np.full_like(temp,np.nan);hi=lo.copy();cal=calibration()
        if cal is not None:
            scale=collocate(d['scale'],d['lat'],d['lon'],lat,lon)
            q=np.array([cal['q'].get(str(float(z)),np.nan) for z in d['depth']]);lo=temp-q*scale;hi=temp+q*scale
        return {'date':date,'lat':lat,'lon':lon,'depth':d['depth'],'temperature':temp,'climatology':np.where(np.isfinite(temp),clim,np.nan),'lower':lo,'upper':hi,'calibrated':cal is not None,'note':'Pointwise calibrated interval where supported. No guarantee for each cell or entire profile; spatial/temporal shift can change coverage.'}
    def pairs():
        f=root/'pairs.csv'
        if not f.exists():raise HTTPException(404,'No collocated evaluation pairs published')
        import pandas as pd
        p=pd.read_csv(f,dtype={'wmo':str,'profile_id':str})
        required={'profile_id','wmo','date','lat','lon','depth','theta0','temperature','climatology'}
        if not required.issubset(p.columns):raise HTTPException(409,'Invalid evaluation pair schema')
        if p.duplicated(['profile_id','depth']).any():raise HTTPException(409,'Duplicate profile-depth pairs')
        return p
    def filtered(start,end,west,east,south,north):
        if west>=east or south>=north:raise HTTPException(400,'Invalid bounding box')
        p=pairs()
        return p[p.date.between(start,end)&p.lon.between(west,east)&p.lat.between(south,north)].copy()
    def csv_response(rows,filename):
        buff=io.StringIO();writer=csv.writer(buff);writer.writerows(rows)
        return Response(buff.getvalue(),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="{filename}"'})
    @app.get('/api/research/capabilities')
    def capabilities():
        dates=available(limit=100000);d=read_day(dates[0]);m=metadata();c=calibration()
        return {'dates':dates,'depths':clean(d['depth']),'synthetic':bool(m.get('synthetic',True)),'mode':m.get('mode','unspecified'),'calibrated':c is not None,'has_pairs':(root/'pairs.csv').exists(),'has_quality':'input_valid' in d,'has_incois':(root/'incois'/'monthly_depth_metrics.csv').exists(),'geometry':{'lat':clean(d['lat']),'lon':clean(d['lon'])},'manifest':m}
    @app.get('/api/research/map')
    def map_data(date:str,depth:float=100,variable:str='temperature',reference_date:str|None=None):
        d=read_day(date);z=depth_index(d,depth);v=field(d,variable)[z]
        if reference_date:
            if variable not in ['temperature','anomaly']:raise HTTPException(400,'Date differences apply only to temperature/anomaly')
            ref=read_day(reference_date)
            if any(not np.array_equal(d[k],ref[k]) for k in ['lat','lon','depth']):raise HTTPException(409,'Grid mismatch')
            v=v-field(ref,variable)[z]
        valid=np.isfinite(v)
        return clean({'date':date,'reference_date':reference_date,'depth':depth,'variable':variable,'lat':d['lat'],'lon':d['lon'],'values':v,'valid_cells':valid.sum(),'total_cells':v.size,'units':'fraction' if variable=='input_fraction' else 'degree_Celsius','note':'Missing includes land, below-bottom, invalid and unsupported data; rendering does not interpolate missing cells.'})
    @app.get('/api/research/profile')
    def profile(date:str,lat:float=15,lon:float=80):return clean(profile_data(date,lat,lon))
    @app.get('/api/research/profile.csv')
    def profile_csv(date:str,lat:float=15,lon:float=80):
        p=profile_data(date,lat,lon);rows=[['date','latitude','longitude','depth_m','potential_temperature_C','climatology_C','interval_lower_C','interval_upper_C']]
        for i,z in enumerate(p['depth']):rows.append([date,lat,lon,float(z)]+[float(p[k][i]) if np.isfinite(p[k][i]) else '' for k in ['temperature','climatology','lower','upper']])
        return csv_response(rows,f'profile-{date}.csv')
    @app.get('/api/research/time-depth')
    def time_depth(start:str,end:str,lat:float=15,lon:float=80,variable:str='temperature'):
        dates=available(start,end,366);out=[];depth=None
        for date in dates:
            d=read_day(date)
            if depth is not None and not np.array_equal(depth,d['depth']):raise HTTPException(409,'Depth grid changed')
            depth=d['depth'];out.append(point(d,lat,lon,variable))
        # Insert explicit missing days so plotting cannot bridge missing export dates.
        full=np.arange(np.datetime64(dates[0]),np.datetime64(dates[-1])+np.timedelta64(1,'D'),dtype='datetime64[D]')
        if len(full)>366:raise HTTPException(400,'Time span must be <=366 calendar days')
        lookup=dict(zip(dates,out));values=np.array([lookup.get(str(dt),np.full(len(depth),np.nan)) for dt in full])
        return clean({'dates':full.astype(str).tolist(),'depth':depth,'values':values.T,'lat':lat,'lon':lon,'variable':variable})
    @app.get('/api/research/transect')
    def transect(date:str,lat1:float=10,lon1:float=65,lat2:float=20,lon2:float=90,points:int=Query(100,ge=2,le=250),variable:str='temperature'):
        if variable not in ['temperature','anomaly','interval_width']:raise HTTPException(400,'Unsupported section variable')
        def xyz(lat,lon):
            lat,lon=np.deg2rad([lat,lon]);return np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
        a,b=xyz(lat1,lon1),xyz(lat2,lon2);angle=np.arccos(np.clip(a@b,-1,1));f=np.linspace(0,1,points)
        if angle<1e-10:raise HTTPException(400,'Choose distinct endpoints')
        if angle>np.pi-1e-5:raise HTTPException(400,'Antipodal transect unsupported')
        v=(np.sin((1-f)*angle)[:,None]*a+np.sin(f*angle)[:,None]*b)/np.sin(angle)
        lats=np.rad2deg(np.arcsin(v[:,2]));lons=np.rad2deg(np.arctan2(v[:,1],v[:,0]));d=read_day(date)
        vals=np.array([point(d,y,x,variable) for y,x in zip(lats,lons)]).T
        return clean({'distance_km':f*angle*6371.0088,'depth':d['depth'],'values':vals,'latitude':lats,'longitude':lons,'note':'Great-circle distance on mean-radius sphere. Native depth levels; horizontal bilinear sampling. Gaps retained; this adds no resolved information.'})
    @app.get('/api/research/validation')
    def validation(start:str='1900-01-01',end:str='2200-01-01',west:float=45,east:float=105,south:float=5,north:float=30):
        p=filtered(start,end,west,east,south,north);depths=[]
        for z,g in p.groupby('depth'):
            m=metrics(g.theta0,g.temperature,g.climatology);m.update({'depth':float(z),'floats':int(g.wmo.nunique()),'profiles':int(g.profile_id.nunique())})
            m['coverage']=None;m['mean_width']=None
            if {'lower','upper'}.issubset(g.columns):
                q=g[np.isfinite(g.lower)&np.isfinite(g.upper)&np.isfinite(g.theta0)]
                if len(q):m['coverage']=float(((q.theta0>=q.lower)&(q.theta0<=q.upper)).mean());m['mean_width']=float((q.upper-q.lower).mean())
            depths.append(m)
        # Deterministic cap for plotting only. Metrics above use all matching pairs.
        plot=p.iloc[np.linspace(0,len(p)-1,min(len(p),3000),dtype=int)] if len(p) else p
        return clean({'depths':depths,'pairs_count':len(p),'profiles':int(p.profile_id.nunique()),'floats':int(p.wmo.nunique()),'plot_sample_count':len(plot),'observed':plot.theta0.tolist(),'predicted':plot.temperature.tolist(),'residual':(plot.temperature-plot.theta0).tolist(),'sample_depth':plot.depth.tolist(),'note':'Metrics use all matched pairs. Scatter is capped at 3000 deterministic rows; absolute-temperature correlation can reflect season/geography. Filtered results are exploratory, not a new blind benchmark.'})
    @app.get('/api/research/argo')
    def argo_list(start:str='1900-01-01',end:str='2200-01-01'):
        p=pairs();p=p[p.date.between(start,end)];g=p.groupby('profile_id',sort=True).first().reset_index()
        return clean(g[['profile_id','wmo','date','lat','lon']].head(500).to_dict('records'))
    @app.get('/api/research/argo-profile')
    def argo_profile(profile_id:str):
        p=pairs();g=p[p.profile_id==profile_id].sort_values('depth')
        if not len(g):raise HTTPException(404,'Profile unavailable')
        return clean({'rows':g.to_dict('records'),'metrics':metrics(g.theta0,g.temperature,g.climatology)})
    @app.get('/api/research/incois')
    def incois():
        import pandas as pd
        f=root/'incois'/'monthly_depth_metrics.csv'
        if not f.exists():raise HTTPException(404,'INCOIS benchmark has not been published')
        return clean({'rows':pd.read_csv(f).to_dict('records'),'note':'Monthly analysed-grid comparison. May share underlying floats with raw Argo; not an independent second observing system.'})
    @app.get('/api/research/download.nc')
    def netcdf(date:str):
        import xarray as xr
        d=read_day(date);m=metadata();dims=('time','depth','latitude','longitude')
        ds=xr.Dataset({'thetao':(dims,d['temperature'][None],{'standard_name':'sea_water_potential_temperature','units':'degree_Celsius','units_metadata':'temperature: on_scale','reference_pressure':'0 dbar'}),'temperature_anomaly':(dims,field(d,'anomaly')[None],{'units':'degree_Celsius','units_metadata':'temperature: difference'})},coords={'time':[np.datetime64(date)],'depth':d['depth'],'latitude':d['lat'],'longitude':d['lon']},attrs={'Conventions':'CF-1.11','title':'OceanEmbed research reconstruction','synthetic':str(m.get('synthetic',True)),'source_manifest':json.dumps(m),'zero_depth_convention':m.get('zero_depth','nominal shallowest teacher level'),'history':'Exported by OceanEmbed research API; no CF compliance checker run'})
        ds.depth.attrs={'standard_name':'depth','units':'m','positive':'down','axis':'Z'};ds.latitude.attrs={'standard_name':'latitude','units':'degrees_north'};ds.longitude.attrs={'standard_name':'longitude','units':'degrees_east'}
        if calibration() is not None:ds['pointwise_interval_width']=(dims,field(d,'interval_width')[None],{'units':'degree_Celsius','long_name':'Calibrated pointwise interval width; empirical coverage must be assessed'})
        content=ds.to_netcdf(engine='scipy');ds.close()
        return Response(bytes(content),media_type='application/x-netcdf',headers={'Content-Disposition':f'attachment; filename="oceanembed-{date}.nc"'})
