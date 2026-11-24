"""Extract the exact three-bond exposure from OGB graphs without computing seventh powers."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import torch


def raw_graph(root):
    def read(name, cols=None):
        return pd.read_csv(root/"raw"/(name+".csv.gz"), header=None,
                           usecols=cols, dtype=np.int32).to_numpy()
    ns, ms = read("num-node-list").ravel(),read("num-edge-list").ravel()
    element = read("node-feat",[0]).ravel()+1
    edges = read("edge")
    assert len(ns)==41127 and ns.sum()==len(element) and ms.sum()==len(edges)
    offsets = np.r_[0, np.cumsum(ns)]
    edges += np.repeat(offsets[:-1], ms).astype(np.int32)[:,None]
    a = coo_matrix((np.ones(len(edges),dtype=np.int32), edges.T),
                   shape=(len(element),len(element))).tocsr()
    a = a.maximum(a.T)
    a.setdiag(0)
    a.eliminate_zeros()
    a.sort_indices()
    a.data.fill(1)
    p=(element!=1)&(element!=6)
    near=a@p.astype(np.int32)
    eligible=(element==6)&(near==0)&((a@near)==0)
    return a,p,eligible


def bfs_exposure(indptr, indices, marked, targets):
    """At each eligible root sum forward shortest-path mass before squaring."""
    n=len(indptr)-1
    out=np.zeros(len(targets),np.float64)
    distance=np.empty(n,np.int16)
    owner=np.full(n,-1,np.int32)
    mass=np.empty(n,np.float64)
    queue=np.empty(n,np.int32)
    for ix,root in enumerate(targets):
        owner[root]=root
        mass[root]=1.
        distance[root]=0
        queue[0]=root
        lo,hi=0,1
        while lo<hi:
            u=queue[lo]
            lo+=1
            deg=indptr[u+1]-indptr[u]
            if distance[u]==3:
                if marked[u]:
                    out[ix]+=(indptr[root+1]-indptr[root])*mass[u]*mass[u]/deg
                continue
            for e in range(indptr[u],indptr[u+1]):
                v=indices[e]
                if owner[v]!=root:
                    owner[v]=root
                    distance[v]=distance[u]+1
                    mass[v]=0.
                    queue[hi]=v
                    hi+=1
                if distance[v]==distance[u]+1:
                    mass[v]+=mass[u]/deg
    return out


def gpu_csr(a):
    d=np.diff(a.indptr)
    rows=np.repeat(np.arange(len(d)),d)
    w=1/np.sqrt(d[rows].astype(np.float64)*d[a.indices])
    return torch.sparse_csr_tensor(torch.as_tensor(a.indptr,device="cuda"),
             torch.as_tensor(a.indices,device="cuda"),torch.as_tensor(w,device="cuda"),
             size=a.shape,device="cuda")


def calc(s,p,roots):
    m2=torch.sparse.mm(s,s)
    m3=torch.sparse.mm(m2,s)
    z=torch.sparse_csr_tensor(m3.crow_indices(),m3.col_indices(),m3.values().square(),
                              size=s.shape,device="cuda")
    return torch.sparse.mm(z,p).ravel()[roots],m2._nnz(),m3._nnz()


def contexts():
    try:
        import pynvml as nv
        nv.nvmlInit()
        h=nv.nvmlDeviceGetHandleByIndex(0)
        return sum(c.pid!=os.getpid() for c in nv.nvmlDeviceGetComputeRunningProcesses(h))
    except Exception:
        return None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ogb",type=Path)
    ap.add_argument("output",type=Path)
    opt=ap.parse_args()
    a,p,eligible=raw_graph(opt.ogb)
    roots=np.flatnonzero(eligible).astype(np.int32)
    try:
        from numba import njit
        bfs=njit(cache=False)(bfs_exposure)
    except ImportError:
        bfs=bfs_exposure
    # Independent adjacency-list implementation uses only degrees, connectivity and field.
    cpu=bfs(a.indptr,a.indices,p,roots)
    dname=torch.cuda.get_device_name()
    assert "H100" in dname
    torch.cuda.set_per_process_memory_fraction(0.15)
    s=gpu_csr(a)
    field=torch.as_tensor(p[:,None],dtype=torch.float64,device="cuda")
    ix=torch.as_tensor(roots,device="cuda",dtype=torch.long)
    with torch.inference_mode():
        y,n2,n3=calc(s,field,ix)
        q=y.cpu().numpy()
        err=float(np.max(np.abs(q-cpu)))
        assert err<1e-12 and np.array_equal(cpu>0,q>0)
        del y
        trials=[]
        for _ in range(5):
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base=torch.cuda.memory_allocated()
            other=contexts()
            start=time.perf_counter()
            y,_,_=calc(s,field,ix)
            torch.cuda.synchronize()
            trials.append(dict(seconds=time.perf_counter()-start,
                               extra_torch_MiB=(torch.cuda.max_memory_allocated()-base)/2**20,
                               external_contexts=other))
            del y
    summary=dict(graphs=41127,atoms=len(p),undirected_edges=a.nnz//2,
                 carbon_without_mark_within_two=len(roots),exact_distance_three=int((cpu>0).sum()),
                 max_absolute_error=err,second_nonzeros=n2,third_nonzeros=n3,device=dname,
                 torch_version=str(torch.__version__),trials=trials,
                 median_seconds=float(np.median([t["seconds"] for t in trials])),
                 peak_MiB=max(t["extra_torch_MiB"] for t in trials),
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 protocol_sha256=hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest())
    opt.output.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
