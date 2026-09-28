"""Optional modules, disabled by default. See docs/EXTENSIONS.md."""
import torch
from torch import nn
import torch.nn.functional as F
class RegimeAdapters(nn.Module):
    def __init__(self,channels,experts=4,rank=16):
        super().__init__();self.router=nn.Conv2d(channels,experts,1)
        self.experts=nn.ModuleList([nn.Sequential(nn.Conv2d(channels,rank,1),nn.GELU(),nn.Conv2d(rank,channels,1)) for _ in range(experts)])
        for e in self.experts:nn.init.zeros_(e[-1].weight);nn.init.zeros_(e[-1].bias)
    def forward(self,h):
        p=self.router(h).softmax(1);v,idx=p.topk(2,dim=1);gate=torch.zeros_like(p).scatter_(1,idx,v);gate=gate/gate.sum(1,keepdim=True).clamp_min(1e-8)
        out=h+sum(e(h)*gate[:,i:i+1] for i,e in enumerate(self.experts))
        usage=(gate>0).float().mean((0,2,3))/2
        balance=len(self.experts)*(usage.detach()*p.mean((0,2,3))).sum()
        return out,balance,usage
class GradNormWeights(nn.Module):
    """Nonnegative objectives only. Separate model/weight optimizers."""
    def __init__(self,n=3,alpha=1.0):
        super().__init__();self.raw=nn.Parameter(torch.zeros(n));self.alpha=alpha;self.register_buffer('initial',torch.zeros(n))
    def weights(self):return len(self.raw)*F.softmax(self.raw,dim=0)
    def objective(self,losses,shared_parameters):
        losses=torch.stack(losses)
        if torch.any(losses.detach()<0):raise ValueError('Continuous NLL cannot be used here')
        if torch.all(self.initial==0):self.initial.copy_(losses.detach().clamp_min(1e-6))
        params=list(shared_parameters);norms=[]
        for loss in losses:
            grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            terms=[(g.detach()**2).sum() for g in grads if g is not None]
            if not terms:raise ValueError('No shared parameter gradients')
            norms.append(torch.sqrt(sum(terms)+1e-12))
        norms=torch.stack(norms)*self.weights();rate=(losses.detach()/self.initial.clamp_min(1e-6)).clamp_min(1e-6);rate=rate/rate.mean()
        return (norms-norms.detach().mean()*rate**self.alpha).abs().sum()
