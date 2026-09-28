import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self,inc,out):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(inc,out,3,padding=1),nn.GroupNorm(4,out),nn.GELU(),nn.Conv2d(out,out,3,padding=1),nn.GroupNorm(4,out),nn.GELU())
    def forward(self,x):return self.net(x)

class OceanEmbed(nn.Module):
    """Per-day CNN, temporal pooling attention, optional bottleneck spatial attention,
    EOF+residual decoder and Laplace scale. GroupNorm supports small batches.
    Input channels: 7 values + 7 masks + 7 supports + 7 ages + lat/lon/sin/cos.
    """
    def __init__(self,eof,pc_mean,width=16,history=7,spatial=False):
        super().__init__();self.history=history;self.spatial=spatial
        self.e1=Block(32,width);self.e2=Block(width,width*2);self.e3=Block(width*2,width*4)
        dim=width*4
        self.time_pos=nn.Parameter(torch.zeros(history,dim));nn.init.normal_(self.time_pos,std=.02)
        self.temporal=nn.MultiheadAttention(dim,4,batch_first=True)
        self.tnorm=nn.LayerNorm(dim)
        if spatial:
            self.position=nn.Linear(2,dim)
            self.space=nn.MultiheadAttention(dim,4,batch_first=True)
            self.snorm=nn.LayerNorm(dim)
        self.d2=Block(dim+width*2,width*2);self.d1=Block(width*3,width)
        self.coeff=nn.Conv2d(width,eof.shape[0],1);self.residual=nn.Conv2d(width,15,1)
        self.scale=nn.Conv2d(width,15,1);self.ssl=nn.Conv2d(width,7,1)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
        self.register_buffer('eof',torch.as_tensor(eof,dtype=torch.float32))
        self.register_buffer('pc_mean',torch.as_tensor(pc_mean,dtype=torch.float32))
    def features(self,x):
        b,l,c,h,w=x.shape
        if l!=self.history:raise ValueError('History length differs from checkpoint')
        v=x.reshape(b*l,c,h,w);s1=self.e1(v);s2=self.e2(F.avg_pool2d(s1,2));s3=self.e3(F.avg_pool2d(s2,2))
        _,d,hh,ww=s3.shape
        seq=s3.reshape(b,l,d,hh,ww).permute(0,3,4,1,2).reshape(b*hh*ww,l,d)+self.time_pos
        fused,_=self.temporal(seq[:,-1:],seq,seq,need_weights=False)
        fused=self.tnorm(fused[:,0]+seq[:,-1]).reshape(b,hh,ww,d)
        if self.spatial:
            yy,xx=torch.meshgrid(torch.linspace(-1,1,hh,device=x.device),torch.linspace(-1,1,ww,device=x.device),indexing='ij')
            tokens=fused.reshape(b,hh*ww,d)+self.position(torch.stack([yy,xx],-1).reshape(1,hh*ww,2))
            att,_=self.space(tokens,tokens,tokens,need_weights=False)
            fused=self.snorm(tokens+att).reshape(b,hh,ww,d)
        f=fused.permute(0,3,1,2)
        s2=s2.reshape(b,l,-1,*s2.shape[-2:])[:,-1];s1=s1.reshape(b,l,-1,h,w)[:,-1]
        f=self.d2(torch.cat([F.interpolate(f,size=s2.shape[-2:],mode='bilinear',align_corners=False),s2],1))
        return self.d1(torch.cat([F.interpolate(f,size=(h,w),mode='bilinear',align_corners=False),s1],1))
    def forward(self,x):
        f=self.features(x)
        anomaly=torch.einsum('bkhw,kz->bzhw',self.coeff(f),self.eof)+self.pc_mean[None,:,None,None]+self.residual(f)
        return anomaly,F.softplus(self.scale(f))+.02,f

class Observer(nn.Module):
    """Shared across depths, uses only surface-derived features at inference."""
    def __init__(self,width):
        super().__init__();self.net=nn.Sequential(nn.Linear(width+7,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,feature,aux):
        o=self.net(torch.cat([feature,aux],-1))
        return o[...,0],F.softplus(o[...,1])+.02

def teacher_loss(pred,target,mask,depths,scale=None):
    # Never compute NaN residuals and hope multiplying by zero repairs them.
    target=torch.where(mask,target,torch.zeros_like(target))
    e=pred-target
    def masked_mean(v,m):return v[m].mean() if m.any() else pred.sum()*0
    profile=masked_mean(F.huber_loss(pred,target,reduction='none',delta=1),mask)
    if scale is not None: profile=masked_mean(abs(e)/scale+torch.log(2*scale),mask)
    dz=(depths[1:]-depths[:-1])[None,:,None,None]
    gp=(pred[:,1:]-pred[:,:-1])/dz;gt=(target[:,1:]-target[:,:-1])/dz
    gradient=masked_mean(F.huber_loss(gp,gt,reduction='none',delta=.05),mask[:,1:]&mask[:,:-1])
    # Integrated anomaly / 300m yields degrees C, easing numerical scaling.
    select=depths<=300
    ip=torch.trapezoid(pred[:,select],depths[select],dim=1)/300
    it=torch.trapezoid(target[:,select],depths[select],dim=1)/300
    column=masked_mean(F.huber_loss(ip,it,reduction='none'),mask[:,select].all(1))
    return profile+.1*gradient+.1*column
