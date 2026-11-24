"""Single-core BREC cost profiles. Run on the compute server with the public data file."""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"

from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import gc
import numpy as np
import scipy
import networkx as nx
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_info, threadpool_limits

K = 24
BLOCK = 8
EIG_TOL = 1e-7
FLOOR = 2e-11
DIRECTION = np.random.default_rng(1729).normal(size=K)
DIRECTION /= np.linalg.norm(DIRECTION)
READS = {}
for width in (K, 2*K):
    weight = np.random.default_rng(6101).normal(size=(width, 2))
    READS[width] = weight / np.abs(weight).sum(0)
now = time.perf_counter


def profile(x):
    return np.sort(np.column_stack((x, x @ READS[x.shape[1]])), axis=0)


def bank(m):
    n = m.shape[0]
    prev = np.eye(n)
    cur = m.toarray() if hasattr(m, "toarray") else m.copy()
    out = np.empty((n, K))
    out[:, 0] = cur.diagonal()
    for k in range(1, K):
        new = m @ cur
        new *= 2
        new -= prev
        prev, cur = cur, new
        out[:, k] = cur.diagonal()
    return out


def feedback(graphs, sparse):
    tic = now()
    base = [a / max(1, a.sum(1).max()) for a in graphs]
    if sparse:
        base = [csr_matrix(m) for m in base]
    x = [bank(m) for m in base]
    profiles1 = [profile(h) for h in x]
    gap1 = np.abs(profiles1[0]-profiles1[1]).max()
    t1 = now()-tic
    outputs = []
    for m, h in zip(base, x):
        p = np.tanh(2*(h @ DIRECTION))
        shifted = .5*m
        if sparse:
            shifted.setdiag(.5*p)
        else:
            np.fill_diagonal(shifted, .5*p)
        y = bank(shifted)
        outputs.append(profile(np.column_stack((h,y))))
    gap = np.abs(outputs[0]-outputs[1]).max()
    total = now()-tic
    return dict(full=total, early=t1 if gap1>FLOOR else total,
                initial=t1, remaining=total-t1, two=total,
                decision=bool(gap>FLOOR), depth_two_decision=bool(gap>FLOOR),
                first=bool(gap1>FLOOR), max_n=len(graphs[0]))


def eigenspaces(a, joint):
    d = a.sum(1)
    ops = [a]
    if joint:
        di = 1/np.sqrt(np.maximum(d,1))
        ops.extend((np.diag(d)-a, np.eye(len(a))-di[:,None]*a*di[None,:]))
    keys = []
    blocks = []
    for m in ops:
        vals, u = eigh(m, driver="evd", check_finite=False)
        cut = np.r_[0, np.flatnonzero(np.diff(vals)>EIG_TOL)+1, len(vals)]
        signature=[]
        for l,r in zip(cut[:-1],cut[1:]):
            signature.append((float(vals[l:r].mean()),int(r-l)))
            blocks.append(u[:,l:r])
        keys.append(tuple(signature))
    return keys,blocks


def ranks(matrix):
    m=np.ascontiguousarray(matrix)
    words=m.view(np.dtype((np.void,m.shape[1]*m.itemsize))).reshape(-1)
    unique, inv=np.unique(words,return_inverse=True)
    return inv.astype(np.int32),len(unique)


def interleave_spaces(blocks):
    """Cluster equal projector entries on both sides before lossless tuple interning."""
    sizes = [len(spaces[0])**2 for spaces in blocks]
    offsets = np.r_[0, np.cumsum(sizes)]
    old = np.zeros(sum(sizes), np.int32)
    for start in range(0, len(blocks[0]), BLOCK):
        count = min(BLOCK, len(blocks[0])-start)
        table = np.empty((len(old), count+1), dtype=np.int32)
        table[:, 0] = old
        for j in range(count):
            values = np.empty(len(old))
            for side, spaces in enumerate(blocks):
                u = spaces[start+j]
                values[offsets[side]:offsets[side+1]] = (u @ u.T).ravel()
            order = np.argsort(values)
            sorted_values = values[order]
            # Consecutive-gap clustering avoids boundary artifacts from fixed grid quantization.
            ranks_1d = np.r_[0, np.cumsum(np.diff(sorted_values) > EIG_TOL)]
            table[order, j+1] = ranks_1d
        old, _ = ranks(table)
    return [old[l:r].reshape(len(b[0]),len(b[0]))
            for l,r,b in zip(offsets[:-1],offsets[1:],blocks)]


def matching_spectra(left, right):
    for a, b in zip(left, right):
        if len(a) != len(b):
            return False
        if any(na != nb or abs(va-vb) > EIG_TOL for (va,na),(vb,nb) in zip(a,b)):
            return False
    return True



def epwl(graphs,joint):
    tic=now()
    spec=[eigenspaces(a,joint) for a in graphs]
    sig_equal=matching_spectra(spec[0][0], spec[1][0])
    t_eig=now()-tic
    if sig_equal:
        labels=interleave_spaces([s[1] for s in spec])
    else:
        labels=[interleave_spaces([s[1]])[0] for s in spec]
        labels[1] += int(labels[0].max())+1
    t_setup=now()-tic
    n=len(graphs[0])
    assert n==len(graphs[1])
    colors=np.zeros(2*n,np.int32)
    ncolors=1
    first_diff=None if sig_equal else t_eig
    first=None
    rounds=0
    two=None
    det2=None
    while True:
        rows=[]
        for side,tokens in enumerate(labels):
            local=colors[side*n:(side+1)*n]
            encoded=tokens.astype(np.int64)*ncolors+local[None,:]
            encoded.sort(axis=1)
            rows.append(np.column_stack((local,encoded)))
        colors,new_count=ranks(np.vstack(rows))
        separated=not np.array_equal(np.sort(colors[:n]),np.sort(colors[n:]))
        rounds+=1
        stamp=now()-tic
        if first_diff is None and separated: first_diff=stamp
        if rounds==1: first=separated
        if rounds==2: two,det2=stamp,separated
        if new_count==ncolors: break
        ncolors=new_count
    total=now()-tic
    if two is None: two,det2=total,separated
    return dict(full=total, early=total if first_diff is None else first_diff,
                initial=t_setup, remaining=total-t_setup, eigen=t_eig, two=two,
                decision=separated, depth_two_decision=det2,first=first, rounds=rounds,
                projector_ct=[len(v[1]) for v in spec], max_n=n)


METHODS={"feedback_csr":lambda a:feedback(a,True),"feedback_dense":lambda a:feedback(a,False),
         "epwl_A":lambda a:epwl(a,False),"epwl_joint":lambda a:epwl(a,True)}
PARTS=[("basic",0,60),("regular",60,160),("extension",160,260),
       ("cfi",260,360),("four_vertex",360,380),("distance_regular",380,400)]


def read_data(file):
    data=np.load(file,allow_pickle=False,mmap_mode="r")
    assert data.shape==(51200,)
    graphs=[]
    for p in range(400):
        pair=[]
        for raw in data[64*p:64*p+2]:
            g=nx.from_graph6_bytes(bytes(raw).strip())
            a=nx.to_numpy_array(g,dtype=np.float64)
            pair.append(a)
        assert pair[0].shape==pair[1].shape
        graphs.append(pair)
    return graphs


def summaries(records, selections):
    out={}
    for subset,ids in selections.items():
        out[subset]={}
        for m in METHODS:
            chosen=[[records[r][idx][m] for idx in ids] for r in range(len(records))]
            timing={}
            for key in ("full","two","early","initial","remaining","eigen"):
                if key not in chosen[0][0]: continue
                vals=np.array([[t[key] for t in rep] for rep in chosen])
                med=np.median(vals,axis=0)
                timing[key]={"pass_totals_s":vals.sum(axis=1).tolist(),
                             "median_pass_s":float(np.median(vals.sum(axis=1))),
                             "p50_pair_ms":float(1000*np.median(med)),
                             "p90_pair_ms":float(1000*np.quantile(med,.90))}
            chosen0=chosen[0]
            counts={key:sum(int(r[key]) for r in chosen0) for key in ("first","decision","depth_two_decision")}
            row=dict(pairs=len(ids),counts=counts,timing=timing)
            if m.startswith("epwl"):
                row["rounds_quantiles"]=np.quantile([r["rounds"] for r in chosen0],[0,.5,.9,1]).tolist()
                vals=[8*r["max_n"]**2*q for r in chosen0 for q in r["projector_ct"]]
                row["hypothetical_uncompressed_projector_bytes_per_graph"]=np.quantile(vals,[0,.5,.9,1]).tolist()
            out[subset][m]=row
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    assert not args.output.exists()
    if hasattr(os,"sched_getaffinity"):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    graphs=read_data(args.data)
    n=np.array([len(x[0]) for x in graphs])
    sizes=list(map(int,np.quantile(n,[0,.25,.5,.75,.90,1],method="nearest")))
    print("loaded",sizes,flush=True)
    selections={"all":list(range(400)),**{p:list(range(l,r)) for p,l,r in PARTS}}
    for lo,hi in [(0,31),(32,63),(64,127),(128,10000)]:
        ids=np.flatnonzero((n>=lo)&(n<=hi)).tolist()
        if ids: selections[f"order_{lo}_{hi}"]=ids
    methods=list(METHODS)
    records=[]
    iso_ids=[l for _,l,_ in PARTS]
    # One pair per category, outside timing; use a matched G/G-permutation null control.
    for idx in iso_ids:
        a=graphs[idx][0]
        pi=np.random.default_rng(9010+idx).permutation(len(a))
        null=[a,a[np.ix_(pi,pi)]]
        for m in methods:
            result=METHODS[m](null)
            assert not result["decision"],("relabel",m,idx)
    gc.disable()
    rng=np.random.default_rng(20271105)
    for rep in range(3):
        batch=[{} for _ in range(400)]
        start=now()
        for step,idx in enumerate(rng.permutation(400)):
            for name in rng.permutation(methods):
                batch[idx][name]=METHODS[name](graphs[idx])
            if (step+1)%100==0:
                print("pass",rep+1,"completed",step+1,"elapsed_s",round(now()-start,2),flush=True)
        records.append(batch)
    gc.enable()
    for idx in range(400):
        for m in methods:
            assert len({rep[idx][m]["decision"] for rep in records})==1
        assert records[0][idx]["feedback_dense"]["decision"]==records[0][idx]["feedback_csr"]["decision"]
    # Independent output check: use the finite-bank decisions from the previous run when available.
    diag=args.data.parent/"pair_diagnostics.json"
    certificate={}
    if diag.is_file():
        rows=json.loads(diag.read_text())
        bits={row["id"]:next(o["separated"] for o in row["observations"] if o["degree"]==24 and
             o["rounds"]==2 and o["seed"]==1729 and o["arithmetic"]=="modular") for row in rows}
        certificate["mismatch_vs_independent_modular_count"]=sum(bool(bits[i])!=records[0][i]["feedback_csr"]["decision"] for i in range(400))
    bysize={}
    for sub,ids in selections.items():
        ords=n[ids]
        bysize[sub]={"n_min":int(ords.min()),"n_max":int(ords.max()),"n_pairs":len(ids),
          "median_n":float(np.median(ords)),
          "node_state_bytes_median":float(np.median(16*K*ords)),
          "compact_pair_label_bytes_median":float(np.median(4*ords**2))}
    cpu=next(x.split(":",1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("model name"))
    with args.data.open("rb") as file: datahash=hashlib.file_digest(file,"sha256").hexdigest()
    result=dict(data_sha256=datahash,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      cpu=cpu, affinity=sorted(os.sched_getaffinity(0)),native_pools=[{k:p.get(k) for k in ("internal_api","num_threads","version","threading_layer")} for p in threadpool_info()],
      versions={"numpy":np.__version__,"scipy":scipy.__version__,"networkx":nx.__version__,"python":platform.python_version()},
      timing_unit="seconds; per-pair quantiles in milliseconds",order_quantiles=sizes,
      data_indices="64*p,64*p+1, p=0..399",generation_seed=20271105,
      protocol="FP64 feature extraction from dense adjacency, graph decoding excluded, 3 interleaved passes. CSR and dense Chebyshev order 24; two projected sorted graph readouts. Eigh, eigenvalue clusters at 1e-7; matched means, multiplicities and sorted consecutive-gap clustering at 1e-7 per projector across the pair; exact integer interning in blocks of 8 projector coordinates. Full encoding through stable partition vs decision-prefix cost and a two-round fixed-depth measurement.",
      relabeling_controls=iso_ids,certificate=certificate,by_size=bysize,
      summary=summaries(records,selections),
      positives={m:[i for i,t in enumerate(records[0]) if t[m]["decision"]] for m in methods})
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("complete", {m:result["summary"]["all"][m]["counts"] for m in methods},flush=True)

if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
