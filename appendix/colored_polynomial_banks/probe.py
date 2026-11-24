"""Measure all-root two-bank polynomial returns using exact finite-distance probing."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime, timezone
import ctypes as ct
import gc,hashlib,json,statistics,subprocess,threading,time,warnings
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.sparse import csgraph
import torch

SIZES=(4096,16384,65536)
ORDERS=(8,24)
WIDTH=64
CAP=120.
LOCKED_HASH='450b8733635d887466a2b96b26411f6e62caf7006f8a264f59cf9b5d75cdf549'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def hasharray(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()


def contexts():
    def smi(query):
        return subprocess.check_output(['nvidia-smi',query,'--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    lines=smi('--query-compute-apps=pid,used_gpu_memory').splitlines()
    other=[ln.split(',') for ln in lines if ln and int(ln.split(',')[0])!=os.getpid()]
    return {'external_compute_contexts':len(other),'external_allocated_MiB':sum(int(x[1].strip()) for x in other if x[1].strip().isdigit()),
           'gpu_model_clock_MHz_memory_used_MiB':smi('--query-gpu=name,clocks.sm,memory.used')}


class RSS:
    def __init__(self):
        self.process=psutil.Process();self.initial=self.process.memory_info().rss;self.peak=self.initial
        self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.01):self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start()
    def close(self):
        self.peak=max(self.peak,self.process.memory_info().rss);self.stop.set();self.thread.join()
        return {'initial_MiB':self.initial/2**20,'peak_MiB':self.peak/2**20,'increment_MiB':(self.peak-self.initial)/2**20}


class Coloring:
    def __init__(self,library):
        dll=ct.CDLL(str(library.resolve()))
        ptr=ct.POINTER(ct.c_int32)
        self.col=dll.color_balls;self.check=dll.verify_balls
        for f in (self.col,self.check):f.argtypes=[ct.c_int,ptr,ptr,ct.c_int,ptr]
        self.col.restype=ct.c_int;self.check.restype=ct.c_int64;self.ptr=ptr
    def invoke(self,f,a,k,colors):
        return f(a.shape[0],a.indptr.ctypes.data_as(self.ptr),a.indices.ctypes.data_as(self.ptr),k,colors.ctypes.data_as(self.ptr))


def graph_from_file(path,n):
    t=time.perf_counter();edge=pd.read_csv(path,compression='gzip',sep=r'\s+',comment='#',header=None,
                                        dtype=np.int32).to_numpy()
    rawrows=len(edge);full_n=int(edge.max())+1
    adjacency=sparse.coo_matrix((np.ones(len(edge),np.uint8),(edge[:,0],edge[:,1])),shape=(full_n,full_n)).tocsr()
    adjacency=adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0);adjacency.eliminate_zeros();adjacency.data.fill(1);adjacency.sort_indices()
    deg=np.diff(adjacency.indptr)
    assert np.count_nonzero(deg)==1088092 and adjacency.nnz//2==1541898
    root=int(np.flatnonzero(deg)[0])
    order=csgraph.breadth_first_order(adjacency,root,directed=True,return_predecessors=False)
    fallback=False
    if len(order)<max(SIZES):
        _,cc=csgraph.connected_components(adjacency,directed=False)
        sizes=np.bincount(cc); candidates=np.flatnonzero(sizes[cc]==sizes.max())
        root=int(candidates.min());fallback=True
        order=csgraph.breadth_first_order(adjacency,root,directed=True,return_predecessors=False)
    assert len(order)>=max(SIZES)
    ids=np.sort(order[:n])
    # Every prefix shares exactly the same BFS order before induced reindexing.
    prefixes={str(size):hasharray(np.sort(order[:size]).astype('<i4')) for size in SIZES}
    csr=adjacency[ids,:][:,ids].tocsr();csr.sort_indices(); d=np.diff(csr.indptr)
    assert (d>0).all() and csr.indptr.dtype==csr.indices.dtype==np.int32
    inv=1./np.sqrt(d.astype(np.float64))
    r=np.repeat(np.arange(n),d)
    csr=csr.astype(np.float64);csr.data=inv[r]*inv[csr.indices]
    report=dict(full_nodes=1088092,full_edges=1541898,raw_directed_rows=rawrows,
                bfs_start_original_id=root,used_largest_component_fallback=fallback,bfs_component_nodes=len(order),
                prefixes_sha256=prefixes,prefix_edges=csr.nnz//2,degree_min=int(d.min()),degree_max=int(d.max()),
                normalized_csr_sha256=hashlib.sha256(csr.indptr.tobytes()+csr.indices.tobytes()+csr.data.tobytes()).hexdigest())
    return csr,report,time.perf_counter()-t


def upload(csr):
    dev=lambda x: torch.as_tensor(x,device='cuda')
    return torch.sparse_csr_tensor(dev(csr.indptr),dev(csr.indices),dev(csr.data),
                                     size=csr.shape,device='cuda',dtype=torch.float64)


def plan(n,colors=None,subset=None):
    if colors is None:
        rows=np.arange(n,dtype=np.int64) if subset is None else np.asarray(subset,np.int64)
        groups=np.arange(len(rows));output=np.arange(len(rows));number=len(rows)
    else:
        rows=np.argsort(colors,kind='stable');groups=colors[rows].astype(np.int64)
        number=int(groups[-1])+1;output=rows
    offsets=np.searchsorted(groups,np.arange(0,number+WIDTH,WIDTH)).tolist()
    dev=lambda q:torch.as_tensor(np.asarray(q,dtype=np.int64),device='cuda')
    return dev(rows),dev(groups%WIDTH),dev(output),list(zip(offsets[:-1],offsets[1:])),len(rows)


@torch.no_grad()
def run_bank(s,probes,k,external_p=None):
    rows,cols,indices,blocks,outsz=probes
    n=s.size(0);nb=0
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
    t=time.perf_counter()
    ans=torch.empty((2,k+1,outsz),dtype=torch.float64,device='cuda')
    ans[:,0,:]=1.
    for bank in (0,1):
        if bank==1:
            pp=(2*ans[0,2]-1 if external_p is None else external_p)[:,None]
        for start,end in blocks:
            rr=rows[start:end]; cc=cols[start:end]; oo=indices[start:end]
            x=torch.zeros((n,WIDTH),dtype=torch.float64,device='cuda')
            x[rr,cc]=1.
            for q in range(1,k+1):
                nxt=torch.sparse.mm(s,x)
                if bank==1:
                    nxt.addcmul_(pp,x);nxt.mul_(.5)
                ans[bank,q,oo]=nxt[rr,cc]
                x=nxt
            nb+=1
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-t
            if elapsed>=CAP:
                return detail(elapsed,base,nb,len(blocks),True),None
    return detail(elapsed,base,nb,len(blocks),False),ans


def detail(sec,base,done,blocks,censored):
    peak=torch.cuda.max_memory_allocated()
    return dict(seconds=sec,censored=censored,completed_blocks=done,planned_blocks=2*blocks,
                resident_input_MiB=base/2**20,peak_allocated_MiB=peak/2**20,peak_additional_MiB=(peak-base)/2**20)


def method(path,n,k,arm,clr):
    gc.collect();torch.cuda.empty_cache();ctx=contexts();clock=utc()
    meter=RSS();start=time.perf_counter()
    csr,meta,common_cpu=graph_from_file(path,n)
    colors=None;ctsec=0.;num=n
    if arm=='color':
        t=time.perf_counter();colors=np.empty(n,np.int32)
        num=clr.invoke(clr.col,csr,k,colors);ctsec=time.perf_counter()-t
        assert int(colors.max())+1==num and len(np.unique(colors))==num
    t=time.perf_counter()
    s=upload(csr);probes=plan(n,colors);torch.cuda.synchronize()
    staging=time.perf_counter()-t
    cold,out=run_bank(s,probes,k)
    e2e=time.perf_counter()-start;rss=meter.close()
    # First output transfer occurs after the synchronized end-to-end timer.
    answer=None if out is None else out.cpu().numpy()
    del out
    warm=[]
    if not cold['censored']:
        for i in range(3):
            trial,x=run_bank(s,probes,k);warm.append(trial);del x
            if trial['censored']:break
    verify_sec=0.;ball=None
    if colors is not None:
        t=time.perf_counter();ball=clr.invoke(clr.check,csr,k,colors);verify_sec=time.perf_counter()-t
        assert ball>=0
    median=statistics.median(r['seconds'] for r in warm) if len(warm)==3 and not any(r['censored'] for r in warm) else None
    entry=dict(utc=clock,contexts_before=ctx,contexts_after=contexts(),graph=meta,method=arm,columns=num,
               blocks_per_bank=len(probes[3]),width=WIDTH,cold_gpu=cold,cold_total_seconds=e2e,
               cpu_common_graph_seconds=common_cpu,cpu_coloring_seconds=ctsec,gpu_staging_seconds=staging,
               process_rss_cold=rss,verify_coloring_seconds=verify_sec,ordered_pairs_checked_in_balls=ball,
               color_sha256=hasharray(colors) if colors is not None else None,
               resident_trials=warm,resident_median_seconds=median)
    del s,probes;gc.collect();torch.cuda.empty_cache()
    print(n,k,arm,'cold_e2e',round(e2e,3),'resident',median,'cols',num,flush=True)
    return entry,answer,csr


def references(csr,n,k,result,full):
    assert result is not None
    roots=np.sort(np.random.Generator(np.random.PCG64(180829+n)).choice(n,128,replace=False))
    direct=2*np.asarray(csr.multiply(csr).sum(axis=1)).ravel()-1
    p_err=float(abs(2*result[0,2]-1-direct).max())
    s=upload(csr);probes=plan(n,subset=roots)
    trial,sub=run_bank(s,probes,k,torch.as_tensor(direct,device='cuda'))
    assert sub is not None and not trial['censored']
    sample_errors=np.max(np.abs(sub.cpu().numpy()-result[:,:,roots]),axis=(1,2))
    complete_errors=np.max(np.abs(full-result),axis=(1,2)).tolist() if full is not None else None
    assert p_err<1e-11 and np.max(sample_errors)<1e-11
    if complete_errors is not None:assert max(complete_errors)<1e-11
    return dict(crosschecked_sample_roots=len(roots),root_sample_sha256=hasharray(roots.astype('<i8')),
                maximum_field_error=p_err,sampled_errors=sample_errors.tolist(),full_bank_errors=complete_errors)


def main():
    ap=ArgumentParser(description=__doc__);ap.add_argument('archive',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--repeat',choices=('0905','0942'))
    a=ap.parse_args();here=Path(__file__).resolve().parent
    assert sha(a.archive)==LOCKED_HASH
    clr=Coloring(here/'color.so')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name()
    tick=time.perf_counter();torch.zeros(1,device='cuda');torch.cuda.synchronize();initsec=time.perf_counter()-tick
    mode='full' if a.repeat is None else a.repeat
    out=a.out/('aggregate.json' if mode=='full' else 'repeat_'+mode+'.json')
    assert not out.exists(),out
    r=dict(mode=mode,began_utc=utc(),gpu=torch.cuda.get_device_name(),gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
           cuda=torch.version.cuda,torch=str(torch.__version__),width=WIDTH,cap_seconds=CAP,fp='FP64',
           cuda_init_seconds=initsec,host_cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0].strip(': \t'),
           archive_sha256=LOCKED_HASH,source_sha256=sha(__file__),cpp_sha256=sha(here/'color.cpp'),
           binary_sha256=sha(here/'color.so'),protocol_sha256=sha(here/'protocol.md'),cases=[])
    combos=[(n,k) for n in SIZES for k in ORDERS] if a.repeat is None else [(65536,24)]
    for case,(n,k) in enumerate(combos):
        sequence=(('identity','color') if case%2==0 else ('color','identity')) if a.repeat is None else ('color',)
        outputs={}; entries={}
        for arm in sequence:
            entries[arm],outputs[arm],csr=method(a.archive,n,k,arm,clr)
        ver=references(csr,n,k,outputs['color'],outputs.get('identity'))
        if 'identity' in entries:
            assert entries['identity']['graph']==entries['color']['graph']
        section=dict(n=n,k=k,order=sequence,methods=entries,verification=ver)
        r['cases'].append(section)
        del entries,outputs,csr
    r['finished_utc']=utc()
    tmp=out.with_suffix('.part');tmp.write_text(json.dumps(r,allow_nan=False,indent=2)+'\n');tmp.replace(out)
    print('aggregate written',mode,flush=True)

if __name__=='__main__':main()
