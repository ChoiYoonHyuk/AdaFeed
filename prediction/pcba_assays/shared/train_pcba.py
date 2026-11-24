"""Fit matched multitask heads using validation only, then save server-local test scores."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from ogb.graphproppred import Evaluator
from prepare_pcba import digest

SEEDS=(103,211,307,401,503)
ARMS=("Common","Local","Spatial")


def split_arrays(root):
    return [pd.read_csv(root/"split"/"scaffold"/(name+".csv.gz"),header=None).to_numpy().ravel()
            for name in ("train","valid","test")]


def labels(root):
    return pd.read_csv(root/"raw"/"graph-label.csv.gz",header=None,dtype=np.float32).to_numpy()


def ap_sorted(y,score):
    neg=-score
    order=neg.argsort(kind="stable")
    a=neg[order]
    yy=y[order].astype(np.int32)
    positive=np.flatnonzero(yy)
    if len(positive)==0 or len(positive)==len(y):return np.nan
    ends=np.searchsorted(a,a[positive],side="right")
    return (yy.cumsum()[ends-1]/ends).mean()


def fast_ap(y,score):
    aps=[]
    for k in range(y.shape[1]):
        mask=np.isfinite(y[:,k])
        aps.append(ap_sorted(y[mask,k],score[mask,k]))
    return float(np.nanmean(aps))


def net(d):
    return nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Dropout(.10),
                         nn.Linear(256,128),nn.ReLU(),nn.Dropout(.10),nn.Linear(128,128)).cuda()


def predict(model,x):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(a) for a in x.split(4096)]).cpu().numpy()


def stdize(arr,tr):
    v=torch.as_tensor(np.array(arr),device="cuda",dtype=torch.float32)
    sd,mean=torch.std_mean(v[tr],dim=0,unbiased=False)
    return ((v-mean)/sd.clamp_min(1e-5)).clamp(-8,8)


def learn(xtr,xva,targets,observed,pos_weight,yv,seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model=net(xtr.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=-1
    for epoch in range(1,21):
        model.train()
        for ids in torch.randperm(len(xtr),device="cuda").split(2048):
            opt.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(model(xtr[ids]),targets[ids],
                                                     pos_weight=pos_weight,reduction="none")
            known=observed[ids]
            objective=(loss*known).sum()/known.sum()
            objective.backward()
            opt.step()
        if epoch%4==0:
            val=fast_ap(yv,predict(model,xva))
            if val>best:
                best,checkpoint,be=val,copy.deepcopy(model.state_dict()),epoch
    return checkpoint,be,best,sum(p.numel() for p in model.parameters())


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("dataset",type=Path)
    cli.add_argument("features",type=Path)
    cli.add_argument("work",type=Path)
    a=cli.parse_args()
    a.work.mkdir(parents=True,exist_ok=True)
    assert "H100" in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    tr,va,te=split_arrays(a.dataset)
    y=labels(a.dataset)
    assert np.array_equal(np.sort(np.r_[tr,va,te]),np.arange(len(y)))
    assert y.shape==(437929,128) and np.isin(y[np.isfinite(y)],[0,1]).all()
    train_labels=torch.as_tensor(y[tr],device="cuda")
    known=torch.isfinite(train_labels)
    target=train_labels.nan_to_num()
    positives=target.sum(0)
    assert bool((positives>0).all())
    weights=((known.sum(0)-positives)/positives).sqrt().clamp(1,20)
    common=stdize(np.load(a.features/"common.npy",mmap_mode="r"),tr)
    local=stdize(np.load(a.features/"local.npy",mmap_mode="r"),tr)
    spatial=stdize(np.load(a.features/"spatial.npy",mmap_mode="r"),tr)
    highs=[torch.zeros_like(local),local,spatial]
    checkpoints={}
    diag={}
    for arm,hi in zip(ARMS,highs):
        x=torch.cat([common,hi],1)
        xt=x[tr]
        xv=x[va]
        checks=[]
        for seed in SEEDS:
            path=a.work/f"{arm}-{seed}.pt"
            if path.exists():
                record=torch.load(path,map_location="cuda",weights_only=True)
            else:
                best,ep,v,params=learn(xt,xv,target,known,weights,y[va],seed)
                record={"state":best,"epoch":ep,"validation_ap":v,"seed":seed,"parameters":params}
                tmp=path.with_suffix(".tmp")
                torch.save(record,tmp)
                os.replace(tmp,path)
            checks.append({k:v for k,v in record.items() if k!="state"})
            print("validation-selected",arm,seed,"epoch",record["epoch"],flush=True)
        checkpoints[arm]=checks
        del xt,xv,x
    dims=common.shape[1]+local.shape[1]
    evalr=Evaluator(name="ogbg-molpcba")
    for arm,hi in zip(ARMS,highs):
        test=torch.cat([common[te],hi[te]],1)
        model=net(dims)
        diag[arm]=[]
        for seed in SEEDS:
            state=torch.load(a.work/f"{arm}-{seed}.pt",map_location="cuda",weights_only=True)
            model.load_state_dict(state["state"])
            pred=predict(model,test)
            official=evalr.eval({"y_true":y[te],"y_pred":pred})["ap"]
            assert abs(official-fast_ap(y[te],pred))<1e-11
            np.save(a.work/f"{arm}-{seed}-test.npy",pred)
            diag[arm].append(float(official))
        print("checked official evaluator",arm,flush=True)
    assert len({r["parameters"] for val in checkpoints.values() for r in val})==1
    report={"source_sha256":digest(__file__),"protocol_sha256":digest(Path(__file__).with_name("protocol.md")),
            "device":torch.cuda.get_device_name(),"torch":str(torch.__version__),
            "seeds":list(SEEDS),"checkpoints":checkpoints,"official_seed_ap":diag,
            "split_n":[len(s) for s in [tr,va,te]],"common_dim":int(common.shape[1]),
            "input_dim":int(dims),"split_sha256":{s:digest(a.dataset/"split"/"scaffold"/(s+".csv.gz")) for s in ("train","valid","test")}}
    (a.work/"fit_summary.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print("fixed models scored",flush=True)


if __name__=="__main__":
    main()
