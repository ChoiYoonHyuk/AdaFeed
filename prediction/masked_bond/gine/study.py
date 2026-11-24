"""Fit the preregistered paired three-layer GINE heads, freeze selection, then score."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import t
from sklearn.metrics import average_precision_score
import torch
from torch import nn
import torch.nn.functional as F

SEEDS=(103,211,307,401,503)
ARMS=("Common","Root-local","Spatial")
WIDTH=64
PLAN_DIGEST="b2789b1221f3a7c1575393fcbab1be9921619481ecb65961c6798ba15397d0a3"


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""):
            digest.update(block)
    return digest.hexdigest()


class GINE(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.atom=nn.Embedding(11,WIDTH)
        self.edge=nn.Parameter(torch.zeros(3,WIDTH))
        self.eps=nn.Parameter(torch.zeros(3))
        self.update=nn.ModuleList([nn.Sequential(nn.Linear(WIDTH,WIDTH),nn.ReLU(),
            nn.Linear(WIDTH,WIDTH),nn.ReLU()) for _ in range(3)])
        self.head=nn.Sequential(nn.Linear(WIDTH+dim,WIDTH),nn.ReLU(),nn.Linear(WIDTH,WIDTH),
                                nn.ReLU(),nn.Linear(WIDTH,1))

    def roots(self,layers):
        h=self.atom.weight
        for i,(own,neighbors) in enumerate(layers):
            msg=F.pad(F.relu(h+self.edge[i]),(0,0,0,1))
            agg=msg[neighbors].sum(1)
            h=self.update[i]((1+self.eps[i])*h[own]+agg)
        return h

    def logits(self,embeddings,root,x):
        return self.head(torch.cat((embeddings[root],x),dim=1)).flatten()


def check_compression():
    from prepare import signatures,compress
    torch.manual_seed(738)
    elem=np.asarray([0,1,0,2,1,0,2,1,0,0])
    edges=np.asarray([(0,1),(1,2),(2,3),(3,4),(4,0),(5,6),(6,7),(7,8),(8,5),(8,9)])
    lookup=[{},{},{}]
    last=signatures(elem,edges,lookup,np.arange(11))
    ids,root=compress(lookup,last)
    ranges=[(torch.from_numpy(ids["self"+str(i)]),torch.from_numpy(ids["neighbor"+str(i)])) for i in range(3)]
    model=GINE(3).double()
    with torch.no_grad():
        model.edge.copy_(torch.randn_like(model.edge)*.2);model.eps.copy_(torch.tensor([.1,.2,-.1]))
    other=copy.deepcopy(model)
    x=torch.randn(len(elem),3,dtype=torch.double)
    a=model.logits(model.roots(ranges),torch.from_numpy(root),x)
    h=other.atom(torch.from_numpy(elem))
    e=np.concatenate((edges,edges[:,::-1]))
    src,dst=torch.from_numpy(e[:,0]),torch.from_numpy(e[:,1])
    for i in range(3):
        agg=torch.zeros_like(h).index_add_(0,dst,F.relu(h[src]+other.edge[i]))
        h=other.update[i]((1+other.eps[i])*h+agg)
    b=other.logits(h,torch.arange(len(h)),x)
    weight=torch.linspace(-1,1,len(a)).double()
    a.dot(weight).backward();b.dot(weight).backward()
    gap=float((a.detach()-b.detach()).abs().max())
    grad=max(float((p.grad-q.grad).abs().max()) for p,q in zip(model.parameters(),other.parameters()))
    assert max(gap,grad)<1e-12
    return {"max_abs_logit":gap,"max_abs_parameter_gradient":grad}


class Cohort:
    def __init__(self,path,scale=None):
        td=lambda x:torch.as_tensor(x,device="cuda")
        with np.load(path) as z:
            raw=np.column_stack((z["shared"],z["high"])).astype(np.float32)
            if scale is None:
                mean=raw.mean(axis=0,dtype=np.float64).astype(np.float32)
                sd=raw.std(axis=0,dtype=np.float64).clip(1e-4).astype(np.float32)
                scale=(mean,sd)
            self.scale=scale
            raw=np.clip((raw-scale[0])/scale[1],-10,10)
            common,local,spatial=td(raw[:,:-2].copy()),td(raw[:,-2:-1].copy()),td(raw[:,-1:].copy())
            self.x={arm:torch.cat((common,s),dim=1) for arm,s in
                    zip(ARMS,(torch.zeros_like(local),local,spatial))}
            self.root=td(z["root"]);self.labels=z["y"];self.y=td(self.labels.astype(np.float32))
            self.mol=z["graph"];self.layers=[(td(z["self"+str(i)]),td(z["neighbor"+str(i)])) for i in range(3)]
        self.counts={"roots":len(self.root),"molecules":int(len(np.unique(self.mol))),
                     "positive":int(self.labels.sum()),"signature_counts":[len(i) for i,_ in self.layers]}


@torch.no_grad()
def scores(model,cohort,arm):
    model.eval()
    emb=model.roots(cohort.layers)
    x=cohort.x[arm]
    return torch.cat([model.logits(emb,r,a) for r,a in zip(cohort.root.split(8192),x.split(8192))]).cpu().numpy()


def fit(train,valid,arm,seed,checkpoint):
    torch.manual_seed(seed)
    generator=torch.Generator(device="cuda").manual_seed(seed+10000)
    x=train.x[arm]
    model=GINE(x.shape[1]).cuda()
    total=sum(p.numel() for p in model.parameters())
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,fused=True)
    best=-1.;selected=0;history=[]
    for epoch in range(1,101):
        model.train()
        for ix in torch.randperm(len(train.y),device="cuda",generator=generator).split(4096):
            optim.zero_grad(set_to_none=True)
            logits=model.logits(model.roots(train.layers),train.root[ix],x[ix])
            F.binary_cross_entropy_with_logits(logits,train.y[ix]).backward()
            optim.step()
        val=float(average_precision_score(valid.labels,scores(model,valid,arm)))
        history.append(val)
        if val>best:
            best=val;selected=epoch
            torch.save(model.state_dict(),checkpoint)
        if epoch-selected>=15:break
    return {"seed":seed,"arm":arm,"best_epoch":selected,"trained_epochs":epoch,
            "validation_ap":best,"validation_curve":history,"parameters":total,
            "checkpoint_sha256":sha(checkpoint)}


def weights(groups,seed):
    unique,code=np.unique(groups,return_inverse=True)
    n=len(unique)
    rng=np.random.default_rng(seed)
    arr=np.vstack((np.ones(n),rng.multinomial(n,np.full(n,1/n),size=1000))).astype(np.int32)
    return torch.as_tensor(arr,device="cuda"),torch.as_tensor(code,device="cuda")


def clustered_ap(pred,cohort,w,code):
    order=np.argsort(-pred,kind="stable")
    end=np.flatnonzero(np.r_[np.diff(pred[order])!=0,True])
    labels=cohort.y[torch.as_tensor(order,device="cuda")].double()
    cluster=code[torch.as_tensor(order,device="cuda")]
    end=torch.as_tensor(end,device="cuda")
    result=[]
    for part in w.split(12):
        mult=part[:,cluster].double()
        tp=(mult*labels).cumsum(1)[:,end]
        total=mult.cumsum(1)[:,end]
        mass=tp-torch.cat((torch.zeros_like(tp[:,:1]),tp[:,:-1]),dim=1)
        ap=((tp/total.clamp_min(1))*mass).sum(1)/tp[:,-1]
        assert ap.isfinite().all()
        result.append(ap.cpu().numpy())
    answer=np.concatenate(result)
    assert abs(answer[0]-average_precision_score(cohort.labels,pred))<1e-11
    return answer


def summarize(predictions,cohort,rng):
    w,code=weights(cohort.mol,rng)
    out={a:np.stack([clustered_ap(x,cohort,w,code) for x in predictions[a]]) for a in ARMS}
    arms={a:{"mean_ap":float(v[:,0].mean()),"seed_ap":v[:,0].tolist()} for a,v in out.items()}
    contrasts={}
    for name in ("Root-local","Common"):
        diff=out["Spatial"]-out[name]
        single=diff[:,0]
        delta=float(single.mean())
        half=float(t.ppf(.975,len(SEEDS)-1)*single.std(ddof=1)/np.sqrt(len(SEEDS)))
        contrasts["Spatial-minus-"+name]={"delta":delta,"positive_seeds":int((single>0).sum()),
            "seed_differences":single.tolist(),"paired_seed_t95":[delta-half,delta+half],
            "paired_molecule95":np.quantile(diff[:,1:].mean(0),[.025,.975]).tolist()}
    return {"counts":cohort.counts,"arms":arms,"comparisons":contrasts}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output",type=Path);a=cli.parse_args()
    where=a.output.resolve();cache=where/"data_server_only";check=where/"checkpoint_server_only"
    check.mkdir(exist_ok=True)
    assert sha(where/"protocol.md")==PLAN_DIGEST
    assert torch.cuda.is_available() and "H100" in torch.cuda.get_device_name()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    audit=check_compression()
    train=Cohort(cache/"train.npz")
    valid=Cohort(cache/"valid.npz",train.scale)
    print("train/validation counts",train.counts,valid.counts,"exact check",audit,flush=True)
    records=[]
    for seed in SEEDS:
        for arm in ARMS:
            checkpoint=check/f"{seed}_{arm}.pt"
            record=check/f"{seed}_{arm}.json"
            if record.exists():
                row=json.loads(record.read_text());assert row["checkpoint_sha256"]==sha(checkpoint)
                assert row["script_sha256"]==sha(Path(__file__)) and row["protocol_sha256"]==PLAN_DIGEST
            else:
                row=fit(train,valid,arm,seed,checkpoint)
                row.update(script_sha256=sha(Path(__file__)),protocol_sha256=PLAN_DIGEST)
                record.write_text(json.dumps(row))
            records.append(row)
            print("selected",seed,arm,row["best_epoch"],row["trained_epochs"],round(row["validation_ap"],5),flush=True)
    (check/"all_frozen.json").write_text(json.dumps(records))
    assert len(records)==15
    result={"protocol_sha256":PLAN_DIGEST,"source_hashes":{f:sha(where/f) for f in
            ("prepare.py","study.py","verify.py")},"gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__),
            "cuda":str(torch.version.cuda),"paired_seeds":list(SEEDS),"compression_test":audit,
            "fit_counts":{"train":train.counts,"valid":valid.counts},"selection":records,"outcomes":{}}
    for name,key in (("molhiv","test"),("PCBA","transfer")):
        evaluation=Cohort(cache/(key+".npz"),train.scale)
        all_pred={arm:[] for arm in ARMS}
        model=GINE(evaluation.x["Common"].shape[1]).cuda()
        for row in records:
            arm=row["arm"]
            model.load_state_dict(torch.load(check/f"{row['seed']}_{arm}.pt",map_location="cuda",weights_only=True))
            all_pred[arm].append(scores(model,evaluation,arm))
        np.savez(where/(key+"_predictions_server_only.npz"),**{k:np.stack(v) for k,v in all_pred.items()})
        result["outcomes"][name]=summarize(all_pred,evaluation,190324+(key=="transfer"))
        print(name,"completed all outcome and interval calculations",flush=True)
    result["joint_superiority"]=all(min(row["paired_molecule95"][0],row["paired_seed_t95"][0])>0
        for section in result["outcomes"].values() for row in section["comparisons"].values())
    file=where/"aggregate.json"
    temp=file.with_suffix(".part");temp.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");temp.replace(file)
    print("Finished. Matched GINE joint superiority:",result["joint_superiority"],flush=True)


if __name__=="__main__":
    main()
