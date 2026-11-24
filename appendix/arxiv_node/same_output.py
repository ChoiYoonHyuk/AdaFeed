"""Time two numerically equivalent all-root return banks on the fixed 4096-node ogbn-arxiv subgraph."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order
import torch

K=24
B=128
NODES=4096
DT=torch.float64
DEVICE="cuda"


def input_graph(dataset):
    with ZipFile(dataset) as archive:
        paths=[z for z in archive.namelist() if z.endswith("/raw/edge.csv.gz")]
        assert len(paths)==1
        with archive.open(paths[0]) as inp:
            e=pd.read_csv(inp,compression="gzip",header=None,dtype=np.int32).values.T
    g=sparse.coo_matrix((np.ones(e.shape[1],dtype=np.int8),e),shape=(169343,169343)).tocsr()
    g.data.fill(1); g=g.maximum(g.T);g.setdiag(0);g.eliminate_zeros();g.sort_indices()
    ids=breadth_first_order(g,36138,directed=False,return_predecessors=False)[:NODES]
    ix=hashlib.sha256(ids.astype("<i4").tobytes()).hexdigest()
    assert ix.startswith("6f7705fd3ae74264aef94257a6d4e88c")
    h=g[ids,:][:,ids].tocsr().astype(np.float64)
    assert h.nnz==22014
    dmax=float(np.asarray(h.sum(1)).max())
    h.setdiag(0);h.sort_indices()
    row=np.repeat(np.arange(NODES),np.diff(h.indptr))
    diag=torch.as_tensor(np.flatnonzero(row==h.indices),device=DEVICE)
    a=torch.sparse_csr_tensor(torch.as_tensor(h.indptr,device=DEVICE),torch.as_tensor(h.indices,device=DEVICE),torch.as_tensor(h.data,device=DEVICE),size=h.shape,device=DEVICE)
    return a,diag,dmax,ix


def csr(template,values):
    return torch.sparse_csr_tensor(template.crow_indices(),template.col_indices(),values,size=template.shape,device=DEVICE)


def cheb_bank(h):
    n=len(h.crow_indices())-1
    ans=torch.empty((n,K),dtype=DT,device=DEVICE)
    for start in range(0,n,B):
        width=min(B,n-start)
        cols=torch.arange(width,device=DEVICE)
        roots=cols+start
        a=torch.zeros((n,width),dtype=DT,device=DEVICE)
        a[roots,cols]=1.
        b=torch.sparse.mm(h,a)
        ans[roots,0]=b[roots,cols]
        for k in range(1,K):
            a,b=b,torch.sparse.mm(h,b).mul_(2).sub_(a)
            ans[roots,k]=b[roots,cols]
    return ans


def eigen_bank(h):
    d,u=torch.linalg.eigh(h,UPLO="L")
    a,b=torch.ones_like(d),d
    vals=[b]
    for k in range(1,K):
        a,b=b,2*d*b-a
        vals.append(b)
    return u.square()@torch.stack(vals,1)


def two_sparse(a,diag,dmax,w):
    values=a.values()/dmax
    first=cheb_bank(csr(a,values))
    p=torch.tanh(2*(first@w))
    shifted=values*.5
    shifted[diag]=p*.5
    second=cheb_bank(csr(a,shifted))
    return first,second


def two_eigh(a,diag,dmax,w):
    u=a.to_dense().div_(dmax)
    first=eigen_bank(u)
    p=torch.tanh(2*(first@w))
    u.diagonal().add_(p)
    u.mul_(.5)
    return first,eigen_bank(u)


def external_gpu_jobs():
    try:
        import pynvml as nv
        nv.nvmlInit()
        h=nv.nvmlDeviceGetHandleByIndex(0)
        return sum(rec.pid!=os.getpid() for rec in nv.nvmlDeviceGetComputeRunningProcesses(h))
    except Exception:
        return None


def trial(method,args):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline=torch.cuda.memory_allocated()
    other=external_gpu_jobs()
    torch.cuda.synchronize()
    start=time.perf_counter()
    y=method(*args)
    torch.cuda.synchronize()
    t=time.perf_counter()-start
    mem=(torch.cuda.max_memory_allocated()-baseline)/2**20
    return [v.cpu() for v in y],dict(seconds=t,extra_torch_peak_MiB=mem,other_compute_contexts=other)


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--output",type=Path,default=Path("same_output_results.json"))
    opt=parser.parse_args()
    assert "H100" in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32=False
    a,diag,dmax,ih=input_graph(opt.archive)
    weights=np.random.default_rng(1729).normal(size=K)
    weights/=np.linalg.norm(weights)
    args=(a,diag,dmax,torch.tensor(weights,device=DEVICE,dtype=DT))
    trials={"sparse":[],"eigh":[]}
    fns={"sparse":two_sparse,"eigh":two_eigh}
    for f in fns.values():trial(f,args)
    largest=[0.,0.]
    for i in range(3):
        outs={}
        for key in (("sparse","eigh") if i%2==0 else ("eigh","sparse")):
            outs[key],metric=trial(fns[key],args)
            trials[key].append(metric)
        for k in range(2):
            largest[k]=max(largest[k],float((outs["sparse"][k]-outs["eigh"][k]).abs().max()))
    stats={k:{"median_seconds":statistics.median(r["seconds"] for r in recs),"peak_MiB":max(r["extra_torch_peak_MiB"] for r in recs)} for k,recs in trials.items()}
    res=dict(nodes=NODES,edges=11007,root=36138,k=K,block=B,gate_seed=1729,ids_sha256=ih,dmax=dmax,device=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=str(torch.version.cuda),max_abs_error=largest,trials=trials,aggregate=stats,protocol_sha256=hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    opt.output.write_text(json.dumps(res,indent=2)+"\n",encoding="utf8")
    print(json.dumps(res),flush=True)


if __name__=="__main__":
    run()
