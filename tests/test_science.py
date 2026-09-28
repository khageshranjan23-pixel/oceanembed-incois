import numpy as np
import pytest
from oceanembed.science import remap_rect,interpolate_profile,collocate,metrics,conformal_q,seasonal_design

def test_constant_remapping():
    lat=np.arange(4,7,.1);lon=np.arange(44,47,.1)
    out,s=remap_rect(np.full((len(lat),len(lon)),17.),lat,lon,np.array([5.,5.25,5.5]),np.array([45.,45.25,45.5]))
    np.testing.assert_allclose(out,17);np.testing.assert_allclose(s,1,atol=1e-6)
def test_land_does_not_turn_into_zero():
    a=np.ones((3,3));a[:]=np.nan
    out,_=remap_rect(a,[0,1,2],[0,1,2],[0,1,2],[0,1,2]);assert np.isnan(out).all()
def test_no_vertical_extrapolation():
    v=interpolate_profile([5,10,100],[28,27,15],target=[0,7,50,101],max_gaps=[25]*4)
    assert np.isnan(v[[0,2,3]]).all();assert v[1]==pytest.approx(27.6)
def test_collocation_linear_and_coastal():
    a=np.array([[0.,1],[1,2]])
    assert collocate(a,[0,1],[0,1],.25,.5)==pytest.approx(.75)
    a[0,0]=np.nan;assert np.isnan(collocate(a,[0,1],[0,1],.25,.5))
def test_metrics_and_conformal():
    m=metrics([1,2,3],[2,3,4],[0,0,0]);assert m['rmse']==1;assert m['bias']==1
    assert conformal_q(np.arange(1,10))==9
    with pytest.raises(ValueError):conformal_q([1])
def test_calendar_leap():
    v=seasonal_design(['2020-01-01','2021-01-01']);np.testing.assert_array_equal(v[0],v[1])
