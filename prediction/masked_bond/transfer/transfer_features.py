"""Rebuild frozen atom-head inputs from official graph files and audit source alignment."""
import os
for env_name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[env_name]="1"
from collections import deque
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from ogb.utils.features import allowable_features
from rdkit import Chem,RDLogger
RDLogger.DisableLog("rdApp.warning");RDLogger.DisableLog("rdApp.error")

ELEM=np.full(120,10,dtype=int)
ELEM[[1,6,7,8,9,15,16,17,35,53]]=np.arange(10)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for buf in iter(lambda:f.read(2<<20),b""):h.update(buf)
    return h.hexdigest()


class Raw:
    def __init__(self,root,need_mapping=True):
        read=lambda x,c=None:pd.read_csv(root/"raw"/(x+".csv.gz"),header=None,usecols=c,dtype=np.int32).to_numpy()
        n,m=read("num-node-list").ravel(),read("num-edge-list").ravel()
        self.ns=np.r_[0,n.cumsum()];self.ms=np.r_[0,m.cumsum()]
        self.elem=read("node-feat",[0]).ravel()+1
        self.edge=read("edge");self.bond=read("edge-feat",[0]).ravel()
        assert self.ns[-1]==len(self.elem) and self.ms[-1]==len(self.edge)==len(self.bond)
        assert self.elem.min()>=1 and self.elem.max()<=118
        self.smi=pd.read_csv(root/"mapping/mol.csv.gz",usecols=["smiles"]).smiles.to_numpy() if need_mapping else None
        assert allowable_features["possible_bond_type_list"][1]=="DOUBLE"
    def molecule(self,g):
        x=slice(self.ns[g],self.ns[g+1]);e=slice(self.ms[g],self.ms[g+1])
        return self.elem[x],self.edge[e],self.bond[e]


def min_dist_bfs(adj,field):
    ans=np.full(len(adj),4,dtype=np.int8)
    starts=np.flatnonzero(field)
    ans[starts]=0;queue=deque(starts)
    while queue:
        i=queue.popleft()
        if ans[i]>=3:continue
        for v in np.flatnonzero(adj[i]):
            if ans[v]>1+ans[i]:
                ans[v]=ans[i]+1;queue.append(v)
    return ans


def build(e,edges,code):
    n=len(e);a=np.zeros((n,n))
    i,j=edges.T
    a[i,j]=a[j,i]=1
    assert a.sum()==2*len(edges) and not a.diagonal().any()
    field=((e!=1)&(e!=6)).astype(float)
    elem=np.eye(11)[ELEM[e]]
    deg=a.sum(1);inv=1/np.sqrt(np.maximum(1,deg))
    s=inv[:,None]*a*inv[None,:];h=s+np.diag(field)
    sk=np.eye(n);hk=np.eye(n);ds=[];dh=[];powers={}
    for k in range(1,9):
        sk=sk@s;hk=hk@h
        ds.append(sk.diagonal().copy());dh.append(hk.diagonal().copy())
        if k in (2,3):powers[k]=sk
    ds=np.stack(ds,1);dh=np.stack(dh,1)
    top=np.column_stack((np.full(n,np.log1p(n)),np.log1p(deg),ds[:,1:8])).astype(np.float32)
    chem=np.column_stack((elem,np.broadcast_to(elem.mean(0),elem.shape))).astype(np.float32)
    adj_counts=(a@elem).astype(np.float32)
    common=np.column_stack((top,chem,adj_counts)).astype(np.float32)
    y=np.zeros(n,np.int8);hit=code==1;y[i[hit]]=1;y[j[hit]]=1
    dbl=np.zeros((n,n),dtype=bool);dbl[i,j]=hit;dbl[j,i]=hit
    np.testing.assert_array_equal(y,np.any(dbl,axis=1))
    dist=np.full(n,4,np.int8);dist[field>0]=0
    seen=np.eye(n,dtype=bool);front=seen
    count=np.empty((n,2,11),np.int16)
    recmax=0.
    for radius in (1,2,3):
        front=(front@a>0)&~seen;seen=seen|front
        dist[(dist==4)&((front@field)>0)]=radius
        if radius>=2:
            count[:,radius-2]=front@elem
            mask=(e==6)&(dist==radius)
            if mask.any():
                col=2*radius
                check=dh[:,col]-ds[:,col]-(powers[radius]**2)@field
                recmax=max(recmax,float(np.abs(check[mask]).max()),float(np.abs(dh[mask,:col]-ds[mask,:col]).max()))
    np.testing.assert_array_equal(dist,min_dist_bfs(a,field))
    res=dict(top=top,common=common,dist=dist,carbon=e==6,y=y,shell_count=count,
             local=ds[:,[4,6]]/np.array([32.,128.]),spatial=dh[:,[4,6]]/np.array([32.,128.]))
    return res,recmax


def compare_source(raw,srcdir,cached,srcmodel,source,out):
    import distance_study as model
    pack=np.load(cached/"atom_features.npz")
    derived=np.load(srcdir/"distance_features.npz")
    cekpath=srcdir/"selected_server_only.pt"
    assert sha(cekpath)=="a9e83a1c236baa0bd5d1290b4e78c59af0c3b23397fd444a8448bbe30ee7311d"
    frozen=torch.load(cekpath,map_location="cpu",weights_only=False)
    atoms=("top","common","dist","carbon","y","shell_count","local","spatial")
    ref={"common":np.column_stack((pack["top"],pack["chem"],pack["neigh"]))}
    for key in atoms:
        if key not in ref:ref[key]=derived[key] if key in derived else pack[key]
    keep_test=pd.read_csv(source/"split/scaffold/test.csv.gz",header=None).to_numpy().ravel()
    sample=np.sort(np.random.default_rng(20261122).choice(keep_test,size=96,replace=False))
    checked=0;rawerr=0.;fp64err=0.;logit_delta=0.;cache_replay=0.
    inputdiff={};new={k:[] for k in atoms};old={k:[] for k in atoms}
    for g in sample:
        row,err=build(*raw.molecule(g));rawerr=max(err,rawerr)
        sl=slice(raw.ns[g],raw.ns[g+1]);checked+=len(row["top"])
        for key in atoms:
            a,b=row[key],ref[key][sl]
            if key in ("local","spatial"):
                z=float(np.max(np.abs(a-b)));fp64err=max(fp64err,z)
                assert z<2e-12
            else:np.testing.assert_array_equal(a,b)
            new[key].append(a);old[key].append(b)
    new={k:np.concatenate(v) for k,v in new.items()};old={k:np.concatenate(v) for k,v in old.items()}
    idx=np.flatnonzero(ref["carbon"].astype(bool)&(pack["split"]==2))
    for d in (3,2):
        sub=np.flatnonzero(new["carbon"]&(new["dist"]==d))
        src_ix=idx[ref["dist"][idx]==d]
        for sens in ("common","shell_count"):
            prior=np.load(srcdir/f"predictions_d{d}_{sens}_server_only.npz")
            np.testing.assert_array_equal(src_ix,prior["idx"])
            for arm in ("local","spatial"):
                name=f"d{d}/{sens}/{arm}"
                mu,sd=frozen["scalers"][name]
                a=model.take(new,d,sens,arm,sub);b=model.take(old,d,sens,arm,sub)
                inputdiff[name]=float(np.max(np.abs(a-b)))
                np.testing.assert_array_equal(a,b)
                va=model.to_gpu(model.standard(a,mu,sd));vb=model.to_gpu(model.standard(b,mu,sd))
                xv=model.to_gpu(model.standard(model.take(ref,d,sens,arm,src_ix),mu,sd))
                for i,state in enumerate(frozen["states"][name]):
                    net=model.predictor(xv.shape[1]);net.load_state_dict(state);net.eval()
                    aa=model.score(net,va);bb=model.score(net,vb)
                    logit_delta=max(logit_delta,float(np.abs(aa-bb).max()))
                    zz=model.score(net,xv)
                    cache_replay=max(cache_replay,float(np.abs(zz-prior[arm][i]).max()))
    assert logit_delta==0 and cache_replay==0
    report=dict(source_molecules=len(sample),source_atoms=checked,source_feature_FP32_max=max(inputdiff.values()),
        source_feature_FP64_max=fp64err,source_new_vs_cached_logits_max=logit_delta,source_checkpoint_replay_max=cache_replay,
        checked_source_checkpoints=40,source_identity_error=rawerr,source_checkpoint_sha256=sha(cekpath))
    (out/"source_alignment.json").write_text(json.dumps(report,indent=2));print("Source validation passed",report,flush=True)


def destination(raw,where,out):
    frozen=np.load(out/"selected_server_only.npz");graphs=frozen["graph"];cl=frozen["scaffold"]
    audits=json.loads((out/"scaffold_certificate.json").read_text())
    assert len(graphs)==audits["retained_n"]==42938 and audits["remaining_possible_overlap"]==0
    errs=dict(rdk_unparsed=0,rdk_element_mismatches=0,rdk_target_mismatches=0,rdk_edge_count_mismatches=0)
    nroot=0;maxerr=0.;groups=[]
    for start in range(0,len(graphs),1024):
        block={}
        for rownum,g in enumerate(graphs[start:start+1024],start):
            e,edges,code=raw.molecule(g);r,err=build(e,edges,code);maxerr=max(err,maxerr)
            m=Chem.MolFromSmiles(raw.smi[g])
            if m is None:
                errs["rdk_unparsed"]+=1
            else:
                ge=np.array([a.GetAtomicNum() for a in m.GetAtoms()])
                if not np.array_equal(ge,e):errs["rdk_element_mismatches"]+=1
                if m.GetNumBonds()!=len(edges):errs["rdk_edge_count_mismatches"]+=1
                yy=np.zeros(m.GetNumAtoms(),np.int8)
                for b in m.GetBonds():
                    if b.GetBondType()==Chem.BondType.DOUBLE:yy[b.GetBeginAtomIdx()]=yy[b.GetEndAtomIdx()]=1
                if not np.array_equal(yy,r["y"]):errs["rdk_target_mismatches"]+=1
            active=r["carbon"]&np.isin(r["dist"],[2,3]);nroot+=int(active.sum())
            r["graph"]=np.full(len(e),g,np.int32);r["scaffold"]=np.full(len(e),cl[rownum],np.int32)
            for k,v in r.items():
                if k!="carbon":block.setdefault(k,[]).append(v[active])
        groups.append({k:np.concatenate(v) for k,v in block.items()})
        if len(groups)%10==0:print("checked",min(start+1024,len(graphs)),"retained destination graphs",flush=True)
    audit={"candidate_graphs_checked":len(graphs),"eligible_atoms":nroot,"first_order_identity_max":maxerr,**errs}
    (out/"destination_alignment.json").write_text(json.dumps(audit,indent=2))
    print("Destination label and feature audit",audit,flush=True)
    assert not any(errs.values()) and maxerr<1e-10
    data={k:np.concatenate([b[k] for b in groups]) for k in groups[0]}
    np.savez(out/"distance_destination_server_only.npz",**data)


def main():
    cli=argparse.ArgumentParser()
    for a in ("src","dst","base","source_models","cache"):cli.add_argument(a,type=Path)
    a=cli.parse_args()
    sys.path.insert(0,str(a.source_models))
    torch.set_num_threads(1);torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(.11)
    compare_source(Raw(a.src,False),a.source_models,a.base,None,a.src,a.cache)
    destination(Raw(a.dst),a.source_models,a.cache)
if __name__=="__main__":main()
