"""Prepare the server-local atom study from untouched OGB molecular CSVs."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from pathlib import Path
from math import comb
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ELEMENTS = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)
GROUP = np.full(120, len(ELEMENTS), dtype=int)
for idx, elem in enumerate(ELEMENTS):
    GROUP[elem] = idx

def rows(path, cols=None):
    return pd.read_csv(path, header=None, usecols=cols).to_numpy(dtype=np.int32)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def powers(h, k):
    current = np.eye(len(h))
    diagonal = [current.diagonal().copy()]
    for j in range(1, k+1):
        current = current @ h
        diagonal.append(current.diagonal().copy())
    return np.stack(diagonal, axis=1)

def compare_bank(s, r, p, check):
    full = powers((s+np.diag(p))/2, 6)[:, 2:]
    local = np.stack([sum(comb(k,j)*p**(k-j)*r[:,j] for j in range(k+1))/2**k for k in range(2,7)], axis=1)
    if check:
        for i in range(min(6,len(s))):
            direct = powers((s+p[i]*np.eye(len(s)))/2, 6)[i,2:]
            np.testing.assert_allclose(local[i], direct, atol=1e-12)
        np.testing.assert_allclose((full[:,1]-local[:,1])*8,(s*s)@p-p*(s*s).sum(1),atol=1e-12)
    return full, local

def encode(element, a, check):
    deg = a.sum(1)
    inv = 1/np.sqrt(np.maximum(deg,1))
    s = a*inv[:,None]*inv[None,:]
    r = powers(s,13)
    group = np.eye(len(ELEMENTS)+1)[GROUP[element]]
    typed = ((element != 1) & (element != 6)).astype(float)
    topology = 2*r[:,2]-1
    typed_full, typed_local = compare_bank(s,r,typed,check)
    topo_full, topo_local = compare_bank(s,r,topology,check)
    counts = a @ group
    common = np.column_stack((np.full(len(a), np.log1p(len(a))), np.log1p(deg),r[:,2:9]))
    return dict(top=common, chem=np.column_stack((group,np.broadcast_to(group.mean(0),group.shape))),
                neigh=counts, typed_full=typed_full, typed_local=typed_local,
                topo_full=topo_full, topo_local=topo_local, longer=r[:,9:14],
                field=np.column_stack((typed,topology)), carbon=(element==6), has_hetero=((a @ typed)>0))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ogb_directory",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    raw=args.ogb_directory/'raw'
    args.output.mkdir(exist_ok=True, parents=True)
    fetch=lambda name,cols=None: rows(raw/(name+'.csv.gz'),cols)
    nodes=fetch('num-node-list').ravel()
    bonds=fetch('num-edge-list').ravel()
    elements=1+fetch('node-feat',[0]).ravel()
    edges=fetch('edge')
    bond_code=fetch('edge-feat',[0]).ravel()
    assert len(nodes)==len(bonds)==41127
    assert nodes.sum()==len(elements) and bonds.sum()==len(edges)==len(bond_code)
    splits = np.full(len(nodes),-1,dtype=int)
    for k,part in enumerate(['train','valid','test']):
        ids=rows(args.ogb_directory/'split/scaffold'/(part+'.csv.gz')).ravel()
        assert (splits[ids]==-1).all()
        splits[ids]=k
    assert (splits>=0).all()
    node_offsets=np.r_[0,np.cumsum(nodes)]; edge_offsets=np.r_[0,np.cumsum(bonds)]
    buf={}
    y=np.zeros(len(elements),dtype=np.int8)
    for i,(n,e) in enumerate(zip(nodes,bonds)):
        ns=slice(node_offsets[i],node_offsets[i+1])
        es=slice(edge_offsets[i],edge_offsets[i+1])
        a=np.zeros((n,n))
        src,dst=edges[es].T
        a[src,dst]=a[dst,src]=1.
        assert a.diagonal().sum()==0
        hit=bond_code[es]==1
        target=np.zeros(n,dtype=np.int8)
        target[src[hit]]=target[dst[hit]]=1
        y[ns]=target
        feats=encode(elements[ns],a,i%4000==0)
        if i<12:
            order=np.random.default_rng(i+391).permutation(n)
            alt=encode(elements[ns][order],a[np.ix_(order,order)],False)
            for name,val in feats.items():
                np.testing.assert_allclose(val[order],alt[name],atol=1e-12)
        for name,val in feats.items():
            if name not in buf:
                buf[name]=np.empty((len(elements),)+val.shape[1:],np.float32)
            buf[name][ns]=val
        if (i+1)%5000==0: print('prepared',i+1,flush=True)
    g=np.repeat(np.arange(len(nodes),dtype=np.int32),nodes)
    split=splits[g].astype(np.int8)
    np.savez(args.output/'atom_features.npz',**buf,graph=g,y=y,split=split)
    raw_files=[raw/(p+'.csv.gz') for p in ['num-node-list','num-edge-list','edge','node-feat','edge-feat']]
    split_files=list((args.ogb_directory/'split/scaffold').glob('*.csv.gz'))
    report=dict(atom_counts=np.bincount(split).tolist(),positive_atoms=np.bincount(split,weights=y).astype(int).tolist(),
                molecules=np.bincount(splits).tolist(),nodes=int(len(y)), max_count=int(nodes.max()),
                input_checksums={p.name:digest(p) for p in raw_files+split_files}, script_sha256=digest(Path(__file__)))
    (args.output/'data_summary.json').write_text(json.dumps(report,indent=2))
    print('prepared all molecular features', flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
