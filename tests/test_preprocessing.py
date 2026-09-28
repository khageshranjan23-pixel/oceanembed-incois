import numpy as np
import xarray as xr
import gsw
import pytest
from oceanembed.prepare import read_spec
from oceanembed.advanced import RegimeAdapters,GradNormWeights
import torch

def test_temperature_conversion_round_trip():
    p=np.array([10,100,1000]);sp=np.array([34,35,35]);temp=np.array([28,20,5.])
    sa=gsw.SA_from_SP(sp,p,80,15);theta=gsw.pt0_from_t(sa,temp,p)
    np.testing.assert_allclose(gsw.t_from_CT(sa,gsw.CT_from_pt(sa,theta),p),temp,atol=1e-9)
    depth=-gsw.z_from_p(p,15);assert np.all(np.diff(depth)>0);assert abs(depth[-1]-1000)>1

def test_ingest_review_and_units(tmp_path):
    f=tmp_path/'input.nc';xr.Dataset({'t':(('time','latitude','longitude'),np.full((1,2,2),300.),{'units':'kelvin'})},coords={'time':[np.datetime64('2020-01-01')],'latitude':[5,6],'longitude':[45,46]}).to_netcdf(f)
    spec={'files':str(f),'variable':'t','expected_units':'kelvin','offset':-273.15,'reviewed':False}
    with pytest.raises(ValueError):read_spec(spec)
    spec['reviewed']=True;ds,da=read_spec(spec);np.testing.assert_allclose(da,26.85);ds.close()
    spec['expected_units']='degree_Celsius'
    with pytest.raises(ValueError):read_spec(spec)

def test_optional_modules():
    torch.set_num_threads(2);h=torch.randn(2,8,4,4,requires_grad=True);r=RegimeAdapters(8)
    out,balance,usage=r(h);assert torch.isfinite(balance);assert out.shape==h.shape
    linear=torch.nn.Linear(3,3);v=linear(torch.randn(4,3));losses=[(v**2).mean(),((v-1)**2).mean(),((v+1)**2).mean()]
    g=GradNormWeights();obj=g.objective(losses,linear.parameters());obj.backward();assert torch.isfinite(g.raw.grad).all()
