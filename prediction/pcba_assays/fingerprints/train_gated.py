"""Matched official-PCBA training with context-dependent gates and test lock."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BINS=64
FP_BITS=2048


class ContextHead(nn.Module):
    def __init__(self, common_dim):
        super().__init__()
        self.shared=nn.Linear(common_dim,512)
        self.gates=nn.Linear(512,BINS)
        self.merge=nn.Linear(512+2*BINS,256)
        self.out=nn.Linear(256,128)
        self.drop=nn.Dropout(.15)

    def forward(self, common, extra):
        state=self.drop(F.relu(self.shared(common)))
        gates=torch.sigmoid(self.gates(state)).unsqueeze(2)
        circ=(gates*extra).flatten(1)
        return self.out(self.drop(F.relu(self.merge(torch.cat((state,circ),1)))))


def atomic_json(data,path):
    tmp=path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,path)


def readonly_scores(model, x, high, idx, zero_bits=False):
    model.eval()
    with torch.inference_mode():
        out=[]
        for ids in idx.split(2048):
            inp=x[ids]
            if zero_bits:inp[:,-FP_BITS:]=0
            out.append(model(inp,high[ids]))
        return torch.cat(out).cpu().numpy()


def train_model(base, x, high, train, valid, target, known, weights, y_valid, seed, zero_bits=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model=ContextHead(x.shape[1]).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=-1.
    best_epoch=0
    curve=[]
    started=time.perf_counter()
    for epoch in range(1,201):
        model.train()
        order=torch.randperm(len(train),device="cuda")
        for perm in order.split(2048):
            ids=train[perm]
            xx=x[ids]
            if zero_bits:xx[:,-FP_BITS:]=0
            opt.zero_grad(set_to_none=True)
            z=model(xx,high[ids])
            bce=F.binary_cross_entropy_with_logits(z,target[perm],pos_weight=weights,reduction="none")
            observed=known[perm]
            loss=(bce*observed).sum()/observed.sum()
            loss.backward()
            opt.step()
        if epoch%4:continue
        score=base.fast_ap(y_valid,readonly_scores(model,x,high,valid,zero_bits))
        curve.append(score)
        if score>best:
            best=score;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
        if epoch==4 or epoch%32==0:
            print("validation",seed,epoch,round(score,5),flush=True)
        if epoch-best_epoch>=32:break
    return {"state":state,"seed":seed,"epoch":best_epoch,"validation_ap":best,"stop_epoch":epoch,
            "reason":"validation_patience" if epoch-best_epoch>=32 else "hard_cap",
            "validation_aps_every_four_epochs":curve,"parameters":sum(p.numel() for p in model.parameters()),
            "train_seconds":round(time.perf_counter()-started,3)}


def make_inputs(base, root, cache, train):
    prev=np.load(root/"features/common.npy",mmap_mode="r")
    ctx=np.load(cache/"contexts.npy",mmap_mode="r")
    fp=np.load(cache/"morgan.npy",mmap_mode="r")
    N=len(prev);old_dim=prev.shape[1]
    assert (N,old_dim)==(437929,300) and ctx.shape==(N,64,15) and fp.shape==(N,256)
    x=torch.empty((N,old_dim+64*11+FP_BITS),device="cuda")
    high=[torch.empty((N,64,2),device="cuda") for k in (0,1)]
    for start in range(0,N,4096):
        sl=slice(start,min(start+4096,N))
        a=np.array(ctx[sl],copy=True)
        assert np.isfinite(a).all()
        x[sl,:old_dim].copy_(torch.from_numpy(np.array(prev[sl])))
        x[sl,old_dim:-FP_BITS].copy_(torch.from_numpy(a[:,:,:11].reshape(len(a),-1).copy()))
        x[sl,-FP_BITS:].copy_(torch.from_numpy(np.unpackbits(fp[sl],axis=1,bitorder="little")))
        for dest,channel in zip(high,(11,13)):
            dest[sl].copy_(torch.from_numpy(a[:,:,channel:channel+2].copy()))
    for a in (x[:,:-FP_BITS],*high):
        sd,mean=torch.std_mean(a[train],dim=0,unbiased=False)
        a.sub_(mean).div_(sd.clamp_min(1e-5)).clamp_(-8,8)
    return x,[torch.zeros_like(high[0]),*high]


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("base",type=Path);cli.add_argument("cache",type=Path);cli.add_argument("fits",type=Path)
    a=cli.parse_args()
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str(a.base.resolve()))
    import train_pcba as base
    from prepare_pcba import digest
    from ogb.graphproppred import Evaluator
    a.fits.mkdir(parents=True,exist_ok=True)
    handle=open(a.fits/"training.lock","a")
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.fits/"fit_summary.json").exists():return
    torch.set_num_threads(1);torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.39)
    assert "H100" in torch.cuda.get_device_name()
    audit={k:json.loads((a.cache/name).read_text()) for k,name in (("context","context_audit.json"),("fingerprint","morgan_audit.json"))}
    assert all(part["n"]==437929 for part in audit.values())
    atomic_json(audit,a.cache/"feature_audit.json")
    hashes={p.name:digest(p) for p in (here/"protocol.md",here/"prepare_context.py",here/"prepare_fingerprints.py",
                  here/"data-recovery.md",here/"train_gated.py",a.base/"train_pcba.py",a.base/"prepare_pcba.py",a.base/"evaluate_pcba.py")}
    tr,va,te=base.split_arrays(a.base/"pcba")
    y=base.labels(a.base/"pcba")
    assert y.shape==(437929,128) and np.array_equal(np.sort(np.r_[tr,va,te]),np.arange(len(y)))
    train,valid=torch.as_tensor(tr,device="cuda"),torch.as_tensor(va,device="cuda")
    train_y=torch.as_tensor(y[tr],device="cuda")
    known=torch.isfinite(train_y); target=train_y.nan_to_num()
    pos=target.sum(0);assert bool((pos>0).all())
    weights=((known.sum(0)-pos)/pos).sqrt().clamp(1,20)
    x,highs=make_inputs(base,a.base,a.cache,train)
    print("GPU feature cache ready",tuple(x.shape),flush=True)
    checks={}
    def one(arm,h,seed,zero_bits=False):
        path=a.fits/f"{arm}-{seed}.pt"
        if path.exists():
            r=torch.load(path,map_location="cuda",weights_only=True)
            assert r["source_hashes"]==hashes
        else:
            print("fitting",arm,seed,flush=True)
            r=train_model(base,x,h,train,valid,target,known,weights,y[va],seed,zero_bits)
            r.update(source_hashes=hashes)
            tmp=path.with_suffix(".tmp")
            torch.save(r,tmp);os.replace(tmp,path)
        print("stopped",arm,seed,r["epoch"],r["stop_epoch"],round(r["validation_ap"],6),flush=True)
        return {k:v for k,v in r.items() if k not in ("state","source_hashes")}
    for arm,h in zip(base.ARMS,highs):
        checks[arm]=[one(arm,h,seed) for seed in base.SEEDS]
    assert sum(map(len,checks.values()))==15
    atomic_json({"hashes":hashes,"checkpoints":checks},a.fits/"primary_selection.json")
    ablation=one("Common-no-fp-validation-only",highs[0],base.SEEDS[0],True)
    print("All primary fits locked; scoring the official test once.",flush=True)
    test=torch.as_tensor(te,device="cuda")
    evaluator=Evaluator(name="ogbg-molpcba")
    scores={}
    for arm,h in zip(base.ARMS,highs):
        model=ContextHead(x.shape[1]).cuda()
        scores[arm]=[]
        for seed in base.SEEDS:
            loc=a.fits/f"{arm}-{seed}-test.npy"
            if loc.exists():
                pred=np.load(loc)
            else:
                record=torch.load(a.fits/f"{arm}-{seed}.pt",map_location="cuda",weights_only=True)
                model.load_state_dict(record["state"])
                pred=readonly_scores(model,x,h,test)
                with loc.with_suffix(".tmp").open("wb") as f:np.save(f,pred)
                os.replace(loc.with_suffix(".tmp"),loc)
            score=float(evaluator.eval({"y_true":y[te],"y_pred":pred})["ap"])
            assert abs(score-base.fast_ap(y[te],pred))<1e-11
            scores[arm].append(score)
    result={"source_sha256":hashes["train_gated.py"],"protocol_sha256":hashes["protocol.md"],"source_hashes":hashes,
         "device":torch.cuda.get_device_name(),"torch":str(torch.__version__),"seeds":list(base.SEEDS),
         "checkpoints":checks,"validation_only_no_fingerprint":ablation,"official_seed_ap":scores,
         "split_n":[len(tr),len(va),len(te)],"common_dim":int(x.shape[1]),"input_dim":int(x.shape[1]+128),
         "max_gpu_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
         "split_sha256":{s:digest(a.base/"pcba/split/scaffold"/(s+".csv.gz")) for s in ("train","valid","test")}}
    atomic_json(result,a.fits/"fit_summary.json")
    print("Scoring complete; compute aggregate paired intervals.",flush=True)
if __name__=="__main__":
    main()
