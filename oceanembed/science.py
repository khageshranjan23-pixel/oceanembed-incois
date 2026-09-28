"""Pure numerical operations; units are explicit at module boundaries."""
import numpy as np

DEPTHS = np.array([0,5,10,20,30,50,75,100,125,150,200,300,500,700,1000], dtype=np.float32)
CHANNELS = ['sst','sss','sla','current_u','current_v','wind_u','wind_v']

def edges(c):
    c = np.asarray(c, float)
    if c.ndim != 1 or len(c) < 2 or not np.all(np.diff(c)>0):
        raise ValueError('Need at least two strictly increasing 1-D cell centres')
    return np.r_[c[0]-(c[1]-c[0])/2, (c[:-1]+c[1:])/2, c[-1]+(c[-1]-c[-2])/2]

def overlap(dst, src, latitude=False):
    d,s = edges(dst),edges(src)
    if latitude:
        d,s = np.sin(np.deg2rad(d)),np.sin(np.deg2rad(s))
    else:
        d,s = np.deg2rad(d),np.deg2rad(s)
    return np.maximum(0, np.minimum(d[1:,None],s[None,1:])-np.maximum(d[:-1,None],s[None,:-1]))

def remap_rect(values, src_lat, src_lon, dst_lat, dst_lon, min_support=0.8):
    """Area-overlap mean for regular rectilinear cell-centred fields.
    Support is fraction of total target-cell area, not observation density.
    NaN source pixels are invalid. Not a curvilinear/swath regridder.
    """
    a,b = overlap(dst_lat,src_lat,True),overlap(dst_lon,src_lon)
    v=np.asarray(values,float); good=np.isfinite(v)
    num=np.einsum('ai,...ij,bj->...ab',a,np.where(good,v,0),b,optimize=True)
    den=np.einsum('ai,...ij,bj->...ab',a,good.astype(float),b,optimize=True)
    area=np.diff(np.sin(np.deg2rad(edges(dst_lat))))[:,None]*np.diff(np.deg2rad(edges(dst_lon)))[None,:]
    support=np.clip(den/area,0,1)
    out=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
    out[support < min_support]=np.nan
    return out.astype('f4'),support.astype('f4')

def interpolate_profile(depth, value, target=DEPTHS, max_gaps=None):
    depth,value=np.asarray(depth),np.asarray(value)
    ok=np.isfinite(depth)&np.isfinite(value)
    depth,value=depth[ok],value[ok]
    order=np.argsort(depth); depth,value=depth[order],value[order]
    depth,ind=np.unique(depth,return_index=True); value=value[ind]
    out=np.full(len(target),np.nan,dtype='f4')
    if len(depth)<2: return out
    # Engineering defaults, to be frozen after training-only thinning tests.
    if max_gaps is None: max_gaps=np.where(np.asarray(target)<=200,25,100)
    for k,z in enumerate(target):
        exact=np.where(np.isclose(depth,z,atol=1e-6,rtol=0))[0]
        if len(exact): out[k]=value[exact[0]]; continue
        j=np.searchsorted(depth,z)
        if 0<j<len(depth) and depth[j]-depth[j-1]<=max_gaps[k]:
            out[k]=np.interp(z,depth[j-1:j+1],value[j-1:j+1])
    return out

def stencil(lat,lon,y,x):
    """Bilinear interpolation indices; strict interior, no extrapolation."""
    j,i=np.searchsorted(lat,y,side='right')-1,np.searchsorted(lon,x,side='right')-1
    j=min(j,len(lat)-2); i=min(i,len(lon)-2)
    if not(lat[0]<=y<=lat[-1] and lon[0]<=x<=lon[-1]): raise ValueError('Outside grid')
    fy=(y-lat[j])/(lat[j+1]-lat[j]); fx=(x-lon[i])/(lon[i+1]-lon[i])
    return (j,i),np.array([(1-fy)*(1-fx),(1-fy)*fx,fy*(1-fx),fy*fx])

def collocate(field,lat,lon,y,x):
    """Strict all-four-valid bilinear collocation; returns NaN at unsupported coasts."""
    (j,i),w=stencil(lat,lon,y,x)
    v=np.stack([field[...,j,i],field[...,j,i+1],field[...,j+1,i],field[...,j+1,i+1]],axis=-1)
    return np.where(np.all(np.isfinite(v),axis=-1),np.sum(v*w,axis=-1),np.nan)

def seasonal_design(dates):
    import pandas as pd
    d=pd.DatetimeIndex(dates)
    # Fraction through each calendar year handles leap years continuously.
    f=(d.dayofyear.to_numpy()-1)/(365+d.is_leap_year.astype(int))
    return np.stack([np.ones(len(d)),np.sin(2*np.pi*f),np.cos(2*np.pi*f),np.sin(4*np.pi*f),np.cos(4*np.pi*f)],1).astype('f4')

def metrics(y,p,c=None):
    y,p=np.asarray(y),np.asarray(p); ok=np.isfinite(y)&np.isfinite(p)
    if c is not None: ok &= np.isfinite(c)
    y,p=y[ok],p[ok]
    if not len(y): return {'n':0,'rmse':None,'mae':None,'bias':None,'correlation':None,'skill':None,'acc':None}
    e=p-y
    def corr(a,b):
        return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>1e-9 and np.std(b)>1e-9 else None
    r={'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),'bias':float(e.mean()),'correlation':corr(y,p),'skill':None,'acc':None}
    if c is not None:
        c=np.asarray(c)[ok]; mse=np.mean((y-c)**2)
        r['skill']=float(1-np.mean(e*e)/mse) if mse>1e-12 else None
        r['acc']=corr(y-c,p-c)
    return r

def conformal_q(scores, alpha=0.1):
    s=np.sort(np.asarray(scores)[np.isfinite(scores)])
    k=int(np.ceil((len(s)+1)*(1-alpha)))
    if not len(s): raise ValueError('Insufficient calibration scores')
    k = min(k, len(s))
    return float(s[k-1])
