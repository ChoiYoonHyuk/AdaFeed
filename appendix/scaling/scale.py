"""FP64 construction of single-graph spectral representations on one GPU."""
import gc
import json
import os
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import pynvml
import torch
from scipy import sparse

BASE = Path(__file__).resolve().parent
DT = torch.float64
GPU = "cuda"
K, B = 24, 128
SIZES = (512, 1024, 2048, 4096, 8192, 16384)
PAIR_LIMIT = 2048
PAIR_SECONDS = 45
REPEATS = 3
weights = np.random.default_rng(1729).normal(size=K)
weights /= np.linalg.norm(weights)
w = torch.tensor(weights, dtype=DT, device=GPU)
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
proc = psutil.Process()

def load_graph(n):
    mat = sparse.load_npz(BASE / f"graph{n}.npz").astype(np.float64)
    degree = np.asarray(mat.sum(1)).ravel()
    mat.setdiag(0)
    mat.sort_indices()
    row = np.repeat(np.arange(n), np.diff(mat.indptr))
    diag = torch.tensor(np.flatnonzero(row == mat.indices), device=GPU)
    g = torch.sparse_csr_tensor(torch.tensor(mat.indptr, device=GPU), torch.tensor(mat.indices, device=GPU),
                 torch.tensor(mat.data, device=GPU), size=(n,n), device=GPU)
    return g, diag, float(degree.max())

def op(g, v):
    return torch.sparse_csr_tensor(g.crow_indices(), g.col_indices(), v, size=g.shape, device=GPU)

def bank(h):
    n = h.shape[0]
    result = torch.empty((n, K), dtype=DT, device=GPU)
    for start in range(0, n, B):
        size = min(B, n-start)
        j = torch.arange(size, device=GPU)
        i = start + j
        prev = torch.zeros((n, size), dtype=DT, device=GPU)
        prev[i,j] = 1
        x = torch.sparse.mm(h, prev)
        result[i,0] = x[i,j]
        for k in range(1, K):
            nxt = torch.sparse.mm(h, x).mul_(2).sub_(prev)
            result[i,k] = nxt[i,j]
            prev, x = x, nxt
    return result

def two_queries(data):
    g, diag, dmax = data
    values = g.values()/dmax
    static = bank(op(g, values))
    p = torch.tanh(2 * (static @ w))
    shifted = values.mul(0.5)
    shifted[diag] = p*0.5
    return static, bank(op(g, shifted))

def eig(data):
    return torch.linalg.eigh(data[0].to_dense(), UPLO="L")

def pairwl(data, cap=None):
    start = time.perf_counter()
    vals, v = eig(data)
    n = len(vals)
    boundaries = torch.nonzero(torch.diff(vals)>1e-8).flatten().add(1).cpu().tolist()
    bounds = [0, *boundaries, n]
    pair = torch.zeros(n*n, device=GPU, dtype=torch.long)
    for j,(lo,hi) in enumerate(zip(bounds, bounds[1:])):
        block = v[:,lo:hi]
        entries = (block @ block.T).reshape(-1)
        # Coordinate quantization followed by lossless compression of tuples.
        coordinate = torch.round(entries*1e8).to(torch.long)
        _, labels = torch.unique(coordinate, return_inverse=True)
        base = n*n + 1
        _, pair = torch.unique(pair * base + labels, return_inverse=True)
        if cap and j % 4 == 0:
            torch.cuda.synchronize()
            if time.perf_counter() - start > cap:
                raise TimeoutError(f"{j+1}/{len(bounds)-1} projectors")
    pair = pair.reshape(n,n)
    colors = torch.zeros(n, device=GPU, dtype=torch.long)
    classes = 1
    rounds = 0
    while True:
        keys = (pair * (n+1) + colors[None,:]).sort(dim=1).values
        rows = torch.cat((colors[:,None],keys), dim=1)
        groups, colors = torch.unique(rows, dim=0, return_inverse=True)
        rounds += 1
        if len(groups)==classes:
            break
        classes = len(groups)
    return colors, len(bounds)-1, rounds

def check():
    g,d,dmax = load_graph(128)
    U = g.to_dense()/dmax
    first, second = two_queries((g,d,dmax))
    def dense_bank(H):
        vals,v = torch.linalg.eigh(H)
        a,b = torch.ones_like(vals),vals
        spectra = [vals]
        for k in range(1,K):
            a,b = b,2*vals*b-a
            spectra.append(b)
        return v.square() @ torch.stack(spectra,1)
    a = dense_bank(U)
    b = dense_bank((U+torch.diag(torch.tanh(2*(a@w))))*0.5)
    err = float(torch.max(torch.stack([(first-a).abs().max(), (second-b).abs().max()])))
    perm = torch.randperm(len(U), device=GPU, generator=torch.Generator(device=GPU).manual_seed(5))
    dense = g.to_dense()[perm][:,perm]
    c,_,_ = pairwl((g,d,dmax))
    c2,_,_ = pairwl((dense.to_sparse_csr(),None,None))
    c = c[perm]
    perm_ok = bool(torch.equal(c[:,None]==c[None,:],c2[:,None]==c2[None,:]))
    assert err<1e-10 and perm_ok
    return dict(evd_bank_max_abs_error=err, numerical_partition_permutation_match=perm_ok, test_vertices=128)

def gpu_used():
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used

def measure(fn, method, data):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    rss = proc.memory_info().rss
    dev_used = gpu_used()
    peak = [rss,dev_used]
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0],proc.memory_info().rss)
            peak[1] = max(peak[1],gpu_used())
            stop.wait(0.005)
    thread = threading.Thread(target=sample,daemon=True)
    thread.start()
    start=time.perf_counter()
    timed_out=None
    try:
        out = fn(data)
        torch.cuda.synchronize()
    except TimeoutError as e:
        timed_out=str(e)
    finally:
        elapsed = time.perf_counter()-start
        stop.set()
        thread.join()
    record = dict(seconds=elapsed,torch_peak_MiB=(torch.cuda.max_memory_allocated()-resident)/2**20,
                  device_delta_MiB=(max(peak[1],gpu_used())-dev_used)/2**20,
                  host_delta_MiB=(max(peak[0],proc.memory_info().rss)-rss)/2**20)
    if timed_out:
        record["timeout"]=timed_out
    elif method=="feedback":
        record["checksum"]=float(out[1].sum())
    elif method=="EVD":
        m2 = float(data[0].values().sum())
        record["trace_square_error"]=abs(float(out[0].square().sum())-m2)
        assert record["trace_square_error"] < 1e-5
    else:
        c,r,it = out
        record.update(projectors=r, rounds=it, classes=int(c.max())+1)
    return record

def other_processes():
    return [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid!=os.getpid()]

def run():
    torch.set_num_threads(1)
    assert not other_processes(), "GPU in use by another compute process"
    meta = json.loads((BASE/'design.json').read_text())
    aggregate = dict(design=meta,device=torch.cuda.get_device_name(),torch=str(torch.__version__),
                     cuda=str(torch.version.cuda),driver=pynvml.nvmlSystemGetDriverVersion(),
                     block=B,k=K,checks=check(),records=[])
    path = BASE/'aggregate.json'
    for n in SIZES:
        data=load_graph(n)
        funcs={"feedback":two_queries,"EVD":eig}
        if n<=PAIR_LIMIT:
            funcs["EPWL"]=lambda dt:pairwl(dt,PAIR_SECONDS)
        for name,fn in funcs.items():
            trials=[]
            for t in range(REPEATS if (name!="EPWL" or n<PAIR_LIMIT) else 1):
                assert not other_processes(), "Concurrent GPU compute during timing"
                trials.append(measure(fn,name,data))
                print(n,name,t,trials[-1],flush=True)
                if "timeout" in trials[-1]:
                    break
            aggregate["records"].append(dict(n=n,method=name,trials=trials))
            path.with_suffix('.tmp').write_text(json.dumps(aggregate,indent=2))
            path.with_suffix('.tmp').replace(path)
        del data
    (BASE/'finished').write_text('ok\n')

if __name__=="__main__":
    run()
