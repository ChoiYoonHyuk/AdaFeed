"""Compute the cubic spatial-minus-local return at every vertex of ogbn-arxiv."""
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
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order
import torch

DTYPE = torch.float64
DEVICE = "cuda"
SIZES = (1024, 4096, 169343)
BLOCK = 64


def graph(archive):
    with ZipFile(archive) as f:
        edge_file, = [n for n in f.namelist() if n.endswith("/raw/edge.csv.gz")]
        with f.open(edge_file) as raw:
            edges = pd.read_csv(raw, header=None, compression="gzip", dtype=np.int32).to_numpy().T
    a = sp.coo_matrix((np.ones(edges.shape[1], dtype=np.int32), edges), shape=(SIZES[-1], SIZES[-1])).tocsr()
    a.data.fill(1)
    a = a.maximum(a.T)
    a.setdiag(0)
    a.eliminate_zeros()
    a.sort_indices()
    bfs = breadth_first_order(a, 36138, directed=False, return_predecessors=False)
    assert hashlib.sha256(bfs[:4096].astype("<i4").tobytes()).hexdigest().startswith("6f7705fd3ae74264")
    return a, bfs


def make_inputs(a):
    n = a.shape[0]
    deg = np.asarray(a.sum(axis=1)).ravel()
    inv = np.zeros(n, dtype=np.float64)
    np.divide(1., np.sqrt(deg), out=inv, where=deg > 0)
    rows = np.repeat(np.arange(n), np.diff(a.indptr))
    weights = inv[rows] * inv[a.indices]
    def to_gpu(x):
        return torch.as_tensor(x, device=DEVICE)
    s = torch.sparse_csr_tensor(to_gpu(a.indptr), to_gpu(a.indices), to_gpu(weights), size=a.shape, device=DEVICE)
    b = torch.sparse_csr_tensor(s.crow_indices(), s.col_indices(), s.values().square(), size=s.shape, device=DEVICE)
    r = torch.sparse.mm(b, torch.ones((n, 1), dtype=DTYPE, device=DEVICE))
    p = 2*r-1
    return s, b, r, p


def edge_formula(b, r, p):
    return torch.sparse.mm(b, p) - r*p


def direct_returns(s, p, indices):
    n = p.shape[0]
    output = torch.empty(len(indices), dtype=DTYPE, device=DEVICE)
    for k in range(0, len(indices), BLOCK):
        roots = indices[k:k+BLOCK]
        cols = torch.arange(len(roots), device=DEVICE)
        basis = torch.zeros((n, len(roots)), dtype=DTYPE, device=DEVICE)
        basis[roots, cols] = 1.
        first = torch.sparse.mm(s, basis)
        first[roots, cols] += p[roots,0]   # H e_i is the same on the first step.
        second_base = torch.sparse.mm(s, first)
        # diag(H^3) = <H e_i,H^2 e_i> for each symmetric operator.
        spatial = (first * (second_base + p * first)).sum(dim=0)
        local = (first * (second_base + p[roots,0] * first)).sum(dim=0)
        output[k:k+BLOCK] = spatial - local
    return output


def external_jobs():
    try:
        import pynvml as nv
        nv.nvmlInit()
        h = nv.nvmlDeviceGetHandleByIndex(0)
        return len([j for j in nv.nvmlDeviceGetComputeRunningProcesses(h) if j.pid != os.getpid()])
    except Exception:
        return None


def timed(f, repetitions):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    contexts = external_jobs()
    start = time.perf_counter()
    for _ in range(repetitions):
        y = f()
        del y
    torch.cuda.synchronize()
    return dict(seconds=(time.perf_counter()-start)/repetitions,
                peak_additional_MiB=(torch.cuda.max_memory_allocated()-baseline)/2**20,
                external_gpu_contexts=contexts)


def evaluate(a, size):
    s,b,r,p = make_inputs(a)
    roots = torch.arange(size, device=DEVICE)
    checks = roots if size < SIZES[-1] else torch.as_tensor(
        np.sort(np.random.default_rng(1701).choice(size, size=128, replace=False)), device=DEVICE)
    fast = lambda: edge_formula(b,r,p)
    ref = lambda: direct_returns(s,p,checks)
    truth, actual = ref(), fast()
    err = (truth - actual[checks,0]).abs().max().item()
    assert err < 1.e-11 and actual.abs().max().item() > 1.e-4
    # CPU unsigned edge sum is an independent check of the weighted pooling identity.
    rows = np.repeat(np.arange(size), np.diff(a.indptr))
    p_host = p[:,0].cpu().numpy()
    wij = b.values().cpu().numpy()
    dirichlet = .5*np.dot(wij, (p_host[rows] - p_host[a.indices])**2)
    residual = abs(float(torch.sum(p*actual))+dirichlet)
    assert residual <= 1.e-9 * max(1., dirichlet)
    assert abs(float(actual.sum())) <= 1.e-10*max(1., float(actual.abs().sum()))
    del actual, truth
    methods = {"edge": (fast, 16)}
    if size < SIZES[-1]:
        methods["return"] = (ref, 1)
    trials = {key: [] for key in methods}
    for f,rep in methods.values():
        timed(f,2)
    for i in range(5):
        order = list(methods) if i%2 == 0 else list(reversed(methods))
        for key in order:
            trials[key].append(timed(*methods[key]))
    return dict(vertices=size, undirected_edges=a.nnz//2, checked_roots=len(checks), abs_error=err,
                energy=dirichlet, energy_residual=residual, trials=trials,
                medians_ms={k:1000*statistics.median(v["seconds"] for v in vs) for k,vs in trials.items()},
                peaks_MiB={k:max(v["peak_additional_MiB"] for v in vs) for k,vs in trials.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    name=torch.cuda.get_device_name()
    assert "H100" in name
    a,bfs=graph(args.archive)
    results=[]
    with torch.inference_mode():
        for n in SIZES:
            h=a if n==SIZES[-1] else a[bfs[:n],:][:,bfs[:n]].tocsr()
            result=evaluate(h,n)
            results.append(result)
            print(n, result["medians_ms"],result["abs_error"],flush=True)
    out=dict(device=name, torch=str(torch.__version__), cuda=str(torch.version.cuda), block=BLOCK,
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             protocol_sha256=hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest(), results=results)
    args.output.write_text(json.dumps(out,indent=2)+"\n")


if __name__ == "__main__":
    main()
