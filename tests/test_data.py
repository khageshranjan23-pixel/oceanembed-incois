import json
import numpy as np
from oceanembed.data import Cube

def test_history_must_be_consecutive(tmp_path):
    dates=np.array(['2020-01-01','2020-01-02','2020-01-05','2020-01-06','2020-01-07'],dtype='datetime64[D]');shape=(5,7,2,2)
    for name,v in {'x':np.zeros(shape),'support':np.ones(shape),'age':np.zeros(shape),'y':np.ones((5,15,2,2)),'wet':np.ones((15,2,2),bool),'lat':np.array([5,6]),'lon':np.array([45,46]),'dates':dates}.items():np.save(tmp_path/f'{name}.npy',v)
    (tmp_path/'metadata.json').write_text(json.dumps({'synthetic':True}));c=Cube(tmp_path);assert c.indices(2020,2020,3)==[4]
