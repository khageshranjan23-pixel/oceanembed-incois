import numpy as np
import torch
from oceanembed.model import OceanEmbed,teacher_loss
from oceanembed.science import DEPTHS

def test_shapes_backward_and_nan_mask():
    torch.set_num_threads(2)
    m=OceanEmbed(np.eye(15,dtype='f4')[:6],np.zeros(15,dtype='f4'),width=8,history=3,spatial=True)
    x=torch.randn(2,3,32,17,25);pred,b,_=m(x)
    assert pred.shape==(2,15,17,25);assert (b>0).all()
    target=torch.randn_like(pred);target[:,:,0]=float('nan');mask=torch.isfinite(target)
    loss=teacher_loss(pred,target,mask,torch.tensor(DEPTHS));assert torch.isfinite(loss)
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
