"""Fit the three frozen Exp5K 2D inputs and evaluate measured carbon shifts."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
from argparse import ArgumentParser
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch import nn
from scipy.stats import t

ARMS=('Common','Root-local','Spatial')
SEEDS=(103,211,307,401,503)
BATCH=512
REPS=2000


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def model(d):
    return nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(.1),
                         nn.Linear(128,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,1)).cuda()


def make_input(z,arm,ix):
    k={'Root-local':'local','Spatial':'spatial','reconstructed':'reconstruction'}.get(arm)
    field=(z[k][ix] if k else np.zeros((len(ix),15),np.float32))
    return np.column_stack((z['common'][ix],field)).astype(np.float32)


def transform(x,sc):
    mu,sd=sc
    return np.clip((x-mu)/sd,-10,10)


@torch.no_grad()
def predict(m,x):
    m.eval()
    return torch.cat([m(b).flatten() for b in x.split(4096)]).cpu().numpy().astype(np.float64)


def fit(xt,yt,xv,yv,location,scale,seed):
    torch.manual_seed(seed)
    rng=torch.Generator(device='cuda').manual_seed(10000+seed)
    m=model(xt.shape[1])
    optim=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001,fused=True)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='min',factor=.5,patience=10,
                     threshold=0.,threshold_mode='abs',min_lr=.00003)
    best=np.inf
    curve,lr=[],[]
    for epoch in range(1,301):
        m.train()
        lr.append(float(optim.param_groups[0]['lr']))
        for ix in torch.randperm(len(yt),device='cuda',generator=rng).split(BATCH):
            optim.zero_grad(set_to_none=True)
            nn.functional.l1_loss(m(xt[ix]).flatten(),yt[ix]).backward()
            optim.step()
        err=float(np.abs(location+scale*predict(m,xv)-yv).mean())
        curve.append(err)
        sched.step(err)
        if err<best:
            best=err; chosen=epoch; state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        elif epoch-chosen>=35:
            break
    return state, dict(seed=seed,selected_epoch=chosen,stop_epoch=epoch,validation_mae_ppm=best,
                       byepoch_val_mae=curve,byepoch_learning_rate=lr)


def stat(y,preds,graph,mask,rngseed):
    target=y[mask]
    groups,idx=np.unique(graph[mask],return_inverse=True)
    n=len(groups); num=np.bincount(idx,minlength=n)
    rng=np.random.default_rng(rngseed)
    weights=rng.multinomial(n,np.full(n,1/n),size=REPS)
    counts=weights@num
    out=dict(atoms=len(target),molecules=n,arms={},comparisons={})
    error={}
    for arm,ps in preds.items():
        er=np.abs(ps[:,mask]-target)
        totals=np.stack([np.bincount(idx,weights=row,minlength=n) for row in er])
        pooled=er.mean(1); macro=(totals/num).mean(1)
        out['arms'][arm]=dict(mae_ppm=float(pooled.mean()),mae_ppm_by_seed=pooled.tolist(),
                              molecule_macro_mae_ppm=float(macro.mean()),macro_by_seed=macro.tolist())
        error[arm]=(pooled,totals)
    for ref in ('Common','Root-local'):
        delta=error[ref][0]-error['Spatial'][0]
        avg=float(delta.mean()); half=float(delta.std(ddof=1)*t.ppf(.975,len(delta)-1)/np.sqrt(len(delta)))
        diff=error[ref][1]-error['Spatial'][1]
        boot=(weights @ diff.mean(0))/counts
        out['comparisons'][ref+'-minus-Spatial']=dict(reduction_ppm=avg,by_seed=delta.tolist(),
                      paired_seed_t95=[avg-half,avg+half],paired_molecule95=np.quantile(boot,[.025,.975]).tolist(),
                      positive_seeds=int((delta>0).sum()))
    return out


def main():
    ap=ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    d=ap.parse_args().output
    assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.065)
    torch.backends.cuda.matmul.allow_tf32=True
    with np.load(d/'features_server_only.npz') as data:
        z={k:data[k] for k in data.files}
    split=z['split']; tr,va,te=[np.flatnonzero(split==k) for k in (0,1,2)]
    y=z['target']; location=float(y[tr].mean()); scale=float(y[tr].std())
    yt=torch.from_numpy(((y[tr]-location)/scale).astype(np.float32)).cuda()
    states,scalers,record={},{},{}
    for arm in ARMS:
        trainx=make_input(z,arm,tr)
        mu=trainx.mean(0,dtype=np.float64).astype(np.float32)
        sd=trainx.std(0,dtype=np.float64).clip(1e-4).astype(np.float32)
        scalers[arm]=mu,sd
        xt=torch.from_numpy(transform(trainx,(mu,sd))).cuda()
        xv=torch.from_numpy(transform(make_input(z,arm,va),(mu,sd))).cuda()
        states[arm],record[arm]=[],[]
        for seed in SEEDS:
            cp=d/f'fit_{arm}_{seed}_server_only.pt'
            if cp.exists():
                saved=torch.load(cp,map_location='cpu',weights_only=False)
                assert saved['source_sha256']==digest(__file__) and saved['protocol_sha256']==digest(d/'protocol.md')
                state,rec=saved['state'],saved['rec']
            else:
                state,rec=fit(xt,yt,xv,y[va],location,scale,seed)
                saved=dict(source_sha256=digest(__file__),protocol_sha256=digest(d/'protocol.md'),
                           state=state,rec=rec,scaler=scalers[arm],output_scaler=(location,scale))
                tmp=cp.with_suffix('.part');torch.save(saved,tmp);tmp.replace(cp)
            states[arm].append(state);record[arm].append(rec)
            print('selected',arm,seed,rec['selected_epoch'],rec['stop_epoch'],flush=True)
        del xt,xv
    assert sum(len(states[a]) for a in ARMS)==15
    print('15 validation-selected checkpoints fixed, beginning test',flush=True)
    inputs=json.loads((d/'input_aggregate.json').read_text())
    preds={}
    check=dict(unequal_test_input_entries=0,max_test_input=0.,max_ppm=0.,unequal_predictions=0)
    for arm in ARMS:
        tx=transform(make_input(z,arm,te),scalers[arm]);xt=torch.from_numpy(tx).cuda()
        if arm=='Spatial':
            txr=transform(make_input(z,'reconstructed',te),scalers[arm])
            check['unequal_test_input_entries']=int(np.count_nonzero(tx!=txr))
            check['max_test_input']=float(abs(tx-txr).max())
            xr=torch.from_numpy(txr).cuda()
        m=model(xt.shape[1]); arm_preds=[]
        for i,state in enumerate(states[arm]):
            m.load_state_dict(state)
            prediction=location+scale*predict(m,xt)
            arm_preds.append(prediction)
            record[arm][i]['test_mae_ppm']=float(np.abs(prediction-y[te]).mean())
            if arm=='Spatial':
                alt=location+scale*predict(m,xr)
                check['unequal_predictions']+=int(np.count_nonzero(prediction!=alt))
                check['max_ppm']=max(check['max_ppm'],float(np.abs(prediction-alt).max()))
        preds[arm]=np.stack(arm_preds)
    np.savez(d/'test_scores_server_only.npz',**preds,graph=z['graph'][te],atom=z['atom'][te],truth=y[te],
             duplicate=z['input_seen'][te],distance=z['distance'][te])
    results={}
    for key,mask in [('all',np.ones(len(te),bool)),('exact_d3',z['distance'][te]==3),
                     ('unseen_2d',~z['input_seen'][te])]:
        results[key]=stat(y[te],preds,z['graph'][te],mask,180425)
    package=dict(input=inputs,protocol_sha256=digest(d/'protocol.md'),source_sha256=digest(__file__),seeds=SEEDS,
                torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
                normalization='training population moments, clip [-10,10]',torch_tf32=True,
                repetitions=REPS,bootstrap_seed=180425,selection=record,results=results,
                peak_gpu_gib=torch.cuda.max_memory_reserved()/1024**3,frozen_reconstruction=check,
                trainable_parameters=sum(v.numel() for v in m.parameters()),network_input_width=int(xt.shape[1]))
    (d/'aggregate.json').write_text(json.dumps(package,indent=2,allow_nan=False)+'\n')
    print('aggregate written',flush=True)


if __name__=='__main__':main()
