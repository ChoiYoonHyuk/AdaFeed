"""Build graph-level GINE sum indices and matched root readouts from OGB raw files."""
import os
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"
import argparse
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd


def signatures(element, edges, lookup, bins):
    """Colors only index repeated node states; no trainable color lookup is used."""
    neighbors=[[] for _ in element]
    for i,j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    current=bins[element].tolist()
    for table in lookup:
        next_ids=[]
        for i,ns in enumerate(neighbors):
            key=(current[i],tuple(sorted(current[j] for j in ns)))
            if key not in table:
                table[key]=len(table)
            next_ids.append(table[key])
        current=next_ids
    return np.asarray(current,np.int64)


def compress(lookup,final):
    """Retain ancestors used by eligible roots. Tables always map to the preceding layer."""
    remap,root=np.unique(final,return_inverse=True)
    gather={}
    for depth in reversed(range(3)):
        keys=list(lookup[depth])
        own=np.asarray([keys[r][0] for r in remap],np.int64)
        nbr=[keys[r][1] for r in remap]
        prev=np.unique(np.r_[own,np.concatenate([np.asarray(n,np.int64) for n in nbr])]) if depth else np.arange(11)
        width=max(1,max(map(len,nbr)))
        index=np.full((len(remap),width),len(prev),np.int64)
        for i,ns in enumerate(nbr):
            index[i,:len(ns)]=np.searchsorted(prev,ns)
        gather[f"self{depth}"]=np.searchsorted(prev,own)
        gather[f"neighbor{depth}"]=index
        remap=prev
    return gather,root


def root_parts(row):
    active=row["carbon"] & (row["dist"]==3)
    orders=np.arange(2,7)
    lower=row["top"][active][:,orders]/np.exp2(orders)
    shared=np.column_stack((row["common"][active],lower,row["shell_count"][active,1])).astype(np.float32)
    high=np.stack((row["local"][active,1],row["spatial"][active,1]),axis=1).astype(np.float32)
    return active,shared,high


def make_split(raw,ids,where,extraction):
    lookup=[{}, {}, {}]
    values={k:[] for k in ("shared","high","y","graph","signature")}
    for g in ids:
        elem,edge,bond=raw.molecule(g)
        row,_=extraction.build(elem,edge,bond)
        active,shared,high=root_parts(row)
        if not active.any():
            continue
        final=signatures(elem,edge,lookup,extraction.ELEM)
        values["shared"].append(shared)
        values["high"].append(high)
        values["y"].append(row["y"][active])
        values["graph"].append(np.full(active.sum(),g,np.int32))
        values["signature"].append(final[active])
    result={k:np.concatenate(xs) for k,xs in values.items()}
    indices,root=compress(lookup,result.pop("signature"))
    result.update(indices)
    result["root"]=root
    np.savez(where,**result)
    print(where.name, "roots",len(root),"root types",len(np.unique(root)),"layer types",
          [len(indices[f"self{i}"]) for i in range(3)],flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ("source","destination","cohort","feature_code","output"):
        cli.add_argument(k,type=Path)
    a=cli.parse_args()
    sys.path.insert(0,str(a.feature_code.resolve()))
    import transfer_features as extraction
    cache=a.output/"data_server_only"
    cache.mkdir(exist_ok=True)
    raw=extraction.Raw(a.source,False)
    for name in ("train","valid","test"):
        f=cache/(name+".npz")
        if not f.exists():
            ids=pd.read_csv(a.source/"split/scaffold"/(name+".csv.gz"),header=None).to_numpy().ravel()
            make_split(raw,ids,f,extraction)
    del raw
    f=cache/"transfer.npz"
    if not f.exists():
        proof=json.loads((a.cohort/"scaffold_certificate.json").read_text())
        assert proof["retained_n"]==42938 and proof["remaining_possible_overlap"]==0
        ids=np.load(a.cohort/"selected_server_only.npz")["graph"]
        assert len(ids)==len(np.unique(ids))==42938
        make_split(extraction.Raw(a.destination,False),ids,f,extraction)
    for name,r,y,g in [("test",13422,593,3412),("transfer",133325,4362,39221)]:
        with np.load(cache/(name+".npz")) as z:
            assert len(z["root"])==r and z["y"].sum()==y and len(np.unique(z["graph"]))==g
    print("Source and destination confirm the previous held-out cohorts.",flush=True)

if __name__=="__main__":
    main()
