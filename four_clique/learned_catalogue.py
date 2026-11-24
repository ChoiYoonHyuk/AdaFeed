"""Fit a nodewise self-loop and fifth-moment readout on whole-core SRG splits.

Requires the public graph6 cache and companion srg_counts.py on the experiment
server. The labels are enumerated four-cliques containing the attachment vertex.
The model receives degree-conditioned standardized fourth moments and fifth
moment responses. It does not receive clique labels as inputs at any stage.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import comb
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import networkx as nx
import torch
from srg_counts import clique_counts

SEEDS = [103,211,307,401,503]
FAMILIES = [(25,12,5,6),(26,10,3,4),(28,12,6,4),(29,14,6,7),(35,16,6,8),(35,18,9,9)]
EPOCHS = 180
MODES = [("local","sum"),("local","gated"),("spatial","sum"),("spatial","gated"),("frozen_spatial","gated")]
EXPONENTS = np.array([(i,total-i) for total in range(6) for i in range(total+1)],int)
GRID = EXPONENTS.copy()
VANDER = np.array([[float(x)**i *float(y)**j for i,j in EXPONENTS] for x,y in GRID])
INV = np.linalg.inv(VANDER)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

def decorations(adj):
    n=len(adj)
    graph=np.zeros((n,n+1,n+1),dtype=np.float64)
    graph[:,:n,:n]=adj
    roots=np.arange(n)
    graph[roots,roots,n]=1
    graph[roots,n,roots]=1
    return graph

def features(adj, check=False):
    m=decorations(adj)
    n,N,_=m.shape
    square=m@m
    degree=square.diagonal(axis1=1,axis2=2)
    fourth=(square*square).sum(axis=2)
    z=np.zeros_like(fourth)
    for value in np.unique(degree):
        mask=degree==value
        size=mask.sum(axis=1,keepdims=True)
        mean=(fourth*mask).sum(axis=1,keepdims=True)/size
        centered=(fourth-mean)*mask
        sd=np.sqrt((centered*centered).sum(axis=1,keepdims=True)/size)
        z+=centered/np.where(sd>1e-9,sd,1.)
    # Observed z has 3 levels on these inputs. Binary masks allow us to cache
    # the polynomial in both nonzero field values, for arbitrary gate weights.
    masks=np.stack([abs(z)<1e-9,z>1e-9,z< -1e-9],axis=2)
    assert masks.sum(axis=2).min()==masks.sum(axis=2).max()==1
    zlevels=np.array([z[masks[:,:,k]].mean() for k in range(3)])
    assert np.max(abs(z-masks@zlevels))<1e-9
    local=np.zeros((n,len(GRID),3),dtype=np.float64)
    spatial=np.empty_like(local)
    ls=[np.ones_like(z),np.zeros_like(z),degree,(m*square).sum(axis=2),fourth,(square*(square@m)).sum(axis=2)]
    for k,moment in enumerate(ls):
        totals=(moment[:,:,None]*masks).sum(axis=1)
        power=5-k
        local[:,0,0]+=totals[:,0] if power==0 else 0
        for g,(i,j) in enumerate([(power,0),(0,power)],1):
            loc=int(np.flatnonzero((EXPONENTS==[i,j]).all(axis=1))[0])
            local[:,loc,g]+=comb(5,k)*totals[:,g]
    for k,(x,y) in enumerate(GRID):
        h=m.copy()
        ids=np.arange(N)
        h[:,ids,ids]=x*masks[:,:,1]+y*masks[:,:,2]
        second=h@h
        d5=(second*(second@h)).sum(axis=2)
        spatial[:,k,:]=(d5[:,:,None]*masks).sum(axis=1)
    coeff=np.einsum('ik,bkg->big',INV,spatial)
    coeff=np.rint(coeff)
    assert np.array_equal(np.einsum('ik,bkg->big',VANDER,coeff),spatial)
    result=np.stack([local,coeff],axis=1).astype(np.int64)
    if check:
        # Independent full floating powers and root-specific uniform shifts on
        # two roots and field values unrelated to the interpolation grid.
        vals=np.array([0.,0.371,-0.428])
        mon=vals[1]**EXPONENTS[:,0]*vals[2]**EXPONENTS[:,1]
        for root in (0,n-1):
            p=masks[root]@vals
            actual=np.diag(np.linalg.matrix_power(m[root]+np.diag(p),5))
            local_actual=np.array([np.linalg.matrix_power(m[root]+v*np.eye(N),5)[i,i] for i,v in enumerate(p)])
            for k,target in enumerate([local_actual,actual]):
                assert np.allclose(mon@result[root,k],target@masks[root],rtol=1e-12,atol=1e-7)
    return result, zlevels

def training_tensors(rows):
    keys=list(rows)
    a=np.stack([np.frombuffer(raw,dtype=np.int64).reshape(2,len(GRID),3) for raw,y in keys])
    return torch.tensor(a,dtype=torch.float64),torch.tensor([y for raw,y in keys],dtype=torch.float64),torch.tensor([rows[key] for key in keys],dtype=torch.float64)

def train(split,zlevels,mode,seed):
    module,pool=mode
    spatial_index=int(module!="local")
    z=torch.tensor(zlevels)
    i=torch.tensor(EXPONENTS[:,0]);j=torch.tensor(EXPONENTS[:,1])
    rng=np.random.default_rng(seed)
    theta=torch.nn.Parameter(torch.tensor([rng.uniform(.15,.6),rng.uniform(-.1,.1)]),requires_grad=module!="frozen_spatial")
    head=torch.nn.Parameter(torch.tensor([rng.normal(0,.1),0.]))
    theta0=theta.detach().clone()
    data=[(X[:,spatial_index],Y,W) for X,Y,W in split]
    def scores(X):
        p=torch.tanh(theta[0]*z+theta[1]*z*z)
        m=p[1]**i*p[2]**j
        w=torch.ones_like(p) if pool=="sum" else p
        return torch.sum((X@w)*m[None,:],dim=1)
    X,Y,W=data[0]
    def ave(q,weights):
        return torch.sum(q*weights)/torch.sum(weights)
    with torch.no_grad():
        r0=scores(X)
        s0=torch.sqrt(ave((r0-ave(r0,W))**2,W)).clamp(min=1.)
        ym=ave(Y,W)
        ys=torch.sqrt(ave((Y-ym)**2,W)).clamp(min=1.)
    def pred(X_,center):
        return ym+ys*(head[0]*(scores(X_)-center)/s0+head[1])
    optimizer=torch.optim.Adam([head]+([theta] if theta.requires_grad else []),lr=.04)
    best=None
    for ep in range(EPOCHS+1):
        center=ave(scores(X),W)
        with torch.no_grad():
            vx,vy,vw=data[1]
            loss=float(ave((pred(vx,center)-vy)**2,vw))
            if best is None or loss<best[0]:
                best=(loss,ep,head.detach().clone(),theta.detach().clone())
        if ep==EPOCHS: break
        optimizer.zero_grad()
        obj=ave(((pred(X,center)-Y)/ys)**2,W)
        obj.backward()
        optimizer.step()
    with torch.no_grad():
        head.copy_(best[2]);theta.copy_(best[3])
        center=ave(scores(X),W)
        tx,ty,tw=data[2]
        err=pred(tx,center)-ty
        rounded=torch.floor(pred(tx,center)+.5+1e-12)
        return dict(seed=seed,epoch=best[1],gate_start=theta0.tolist(),gate_final=theta.tolist(),gate_change=float(torch.linalg.norm(theta-theta0)),mae=float(ave(torch.abs(err),tw)),rmse=float(ave(err**2,tw)**.5),correct=int(((rounded==ty)*tw).sum()),n=int(tw.sum()))

def run_family(params,cache):
    filename='sr'+''.join(map(str,params))+'.g6'
    path=cache/filename
    raw=path.read_bytes()
    lines=[s.removeprefix(b'>>graph6<<') for s in raw.splitlines() if s.strip()]
    lines=[s for s in lines if s]
    cores=len(lines);n,d,_,_=params
    split_seed=20271107+100*n+d
    order=np.random.default_rng(split_seed).permutation(cores)
    ntr=int(.6*cores);nv=max(1,int(.2*cores))
    labels=np.full(cores,2)
    labels[order[:ntr]]=0;labels[order[ntr:ntr+nv]]=1
    counters=[Counter() for _ in range(3)]
    shape=None
    for idx,line in enumerate(lines):
        adj=nx.to_numpy_array(nx.from_graph6_bytes(line),nodelist=list(range(n)),dtype=np.int64)
        co,z=features(adj,check=idx==0)
        if shape is None:shape=z
        assert np.allclose(shape,z)
        targets=clique_counts(adj)
        for X,target in zip(co,targets):
            counters[labels[idx]][(X.tobytes(),target)]+=1
        if idx and idx%500==0: print("features",params,idx,flush=True)
    sets=list(map(training_tensors,counters))
    test_hist=Counter()
    for (_,y),ct in counters[2].items():test_hist[y]+=ct
    groups=[int(t.sum().item()) for _,_,t in sets]
    return dict(parameters=params,split_seed=split_seed,cores=[ntr,nv,cores-ntr-nv],roots=groups,source_sha256=sha256(raw).hexdigest(),split_sha256=sha256(labels.astype(np.uint8).tobytes()).hexdigest(),best_constant_on_test=max(test_hist.values()),test_target_range=[min(test_hist),max(test_hist)],modes={mod+'_'+pool:[train(sets,shape,(mod,pool),seed) for seed in SEEDS] for mod,pool in MODES})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('cache',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    assert not args.output.exists()
    t0=time.monotonic();out=dict(epochs=EPOCHS,seeds=SEEDS,scipy_free=True,script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),numpy=np.__version__,torch=torch.__version__,families=[])
    for params in FAMILIES:
        report=run_family(params,args.cache)
        out['families'].append(report)
        print(params,report['cores'],report['roots'],{k: [(round(x['mae'],3),x['correct']) for x in v] for k,v in report['modes'].items()},flush=True)
        out['seconds']=round(time.monotonic()-t0,1)
        args.output.write_text(json.dumps(out,indent=2))
    print('COMPLETED',flush=True)

if __name__=='__main__':main()
