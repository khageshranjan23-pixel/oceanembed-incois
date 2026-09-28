import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import Cube,Windows
from .model import OceanEmbed,teacher_loss
from .science import DEPTHS

def load_model(run,device='cpu'):
    run=Path(run);conf=json.loads((run/'model.json').read_text());stats=dict(np.load(run/'stats.npz'))
    model=OceanEmbed(stats['eof'],stats['pc_mean'],**conf).to(device)
    model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True));model.eval()
    return model,stats,conf

def main():
    p=argparse.ArgumentParser();p.add_argument('--cube',required=True);p.add_argument('--run',required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--width',type=int,default=16);p.add_argument('--history',type=int,default=7);p.add_argument('--batch',type=int,default=2);p.add_argument('--patch',type=int,default=64);p.add_argument('--seed',type=int,default=17);p.add_argument('--spatial',action='store_true');p.add_argument('--ssl',action='store_true');p.add_argument('--pretrained');p.add_argument('--nll',action='store_true');p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4)
    run=Path(a.run);run.mkdir(parents=True,exist_ok=True);stats=dict(np.load(run/'stats.npz'));c=Cube(a.cube)
    conf=dict(width=a.width,history=a.history,spatial=a.spatial);(run/'model.json').write_text(json.dumps(conf))
    (run/'train_config.json').write_text(json.dumps(vars(a),indent=2));(run/'data_metadata.json').write_text(json.dumps(c.meta))
    model=OceanEmbed(stats['eof'],stats['pc_mean'],**conf).to(a.device)
    if a.pretrained:model.load_state_dict(torch.load(a.pretrained,map_location=a.device,weights_only=True))
    train=Windows(c,stats,c.indices(2020,2020,a.history),a.history,a.patch,augment=not a.ssl)
    val=Windows(c,stats,c.indices(2020,2020,a.history),a.history,None)
    if not len(train) or not len(val):raise ValueError('Need training and 2022 validation windows')
    dl=DataLoader(train,batch_size=a.batch,shuffle=True,num_workers=0);vl=DataLoader(val,batch_size=1)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    best=float('inf');depth=torch.tensor(DEPTHS,device=a.device);history=[]
    for epoch in range(a.epochs):
        model.train();losses=[]
        for x,y,m in dl:
            x,y,m=x.to(a.device),y.to(a.device),m.to(a.device);opt.zero_grad()
            if a.ssl:
                original=x[:,-1,:7].clone();valid=x[:,-1,7:14].bool()
                # Spatial block mask shared over time, so no exact pixel leaks through adjacent input days.
                b,l,_,h,w=x.shape
                hide=torch.nn.functional.interpolate((torch.rand(b,7,max(1,h//8),max(1,w//8),device=a.device)<.4).float(),size=(h,w),mode='nearest').bool()
                for offset in [0,7,14]:x[:,:,offset:offset+7]=x[:,:,offset:offset+7].masked_fill(hide[:,None],0)
                recon=model.ssl(model.features(x));mask=hide&valid
                if not mask.any():continue
                loss=torch.nn.functional.huber_loss(recon[mask],original[mask])
            else:
                pred,scale,_=model(x);loss=teacher_loss(pred,y,m,depth,scale if a.nll else None)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
        model.eval();ss=0.;nn=0
        if not a.ssl:
            with torch.no_grad():
                for x,y,m in vl:
                    pred,_,_=model(x.to(a.device));e=(pred.cpu()-y)[m];ss+=float((e*e).sum());nn+=e.numel()
            score=(ss/max(nn,1))**.5
        else:score=float(np.mean(losses)) # SSL loss only selects pretraining duration; final models evaluated downstream.
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'teacher_validation_rmse' if not a.ssl else 'ssl_training_loss':score})
        print(history[-1],flush=True)
        if score<best:
            best=score;torch.save(model.state_dict(),run/('ssl.pt' if a.ssl else 'best.pt'))
        (run/'history.json').write_text(json.dumps(history,indent=2))
    if not a.nll and not a.ssl:print('Mean trained. Scale head is NOT calibrated; train observer/calibration before displaying intervals.')
if __name__=='__main__':main()
