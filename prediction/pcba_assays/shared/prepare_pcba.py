"""Prepare graph-only matched channels for the official ogbg-molpcba archive."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ogb.utils.features import allowable_features

NODE_KEYS = ("possible_atomic_num_list", "possible_chirality_list", "possible_degree_list",
             "possible_formal_charge_list", "possible_numH_list", "possible_number_radical_e_list",
             "possible_hybridization_list", "possible_is_aromatic_list", "possible_is_in_ring_list")
BOND_KEYS = ("possible_bond_type_list", "possible_bond_stereo_list", "possible_is_conjugated_list")
GROUPS = 6
DT = torch.float64


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def csv(path, dtype):
    return pd.read_csv(path, header=None, dtype=dtype).to_numpy()


def element_groups():
    table = []
    field = []
    for z in allowable_features[NODE_KEYS[0]]:
        if z == "misc":
            table.append(5)
            field.append(-1)
            continue
        table.append(z - 6 if z in (6, 7, 8) else 3 if z in (9, 17, 35, 53, 85, 117)
                     else 4 if z in (15, 16) else 5)
        field.append(int(z > 1 and z != 6))
    return torch.tensor(table, device="cuda"), torch.tensor(field, device="cuda", dtype=DT)


def grouped_features(x, adj, edge_bins, lookup, field_lookup):
    batch, n = adj.shape[:2]
    grp = lookup[x[:, :, 0]]
    p = field_lookup[x[:, :, 0]]
    assert p.min().item() >= 0
    group = F.one_hot(grp, GROUPS).to(DT).transpose(1, 2)
    deg = adj.sum(2)
    norm = torch.where(deg > 0, deg.clamp_min(1).rsqrt(), 0)
    s = adj * norm.unsqueeze(1) * norm.unsqueeze(2)
    z = (s + torch.diag_embed(p)) * .5
    one = torch.eye(n, device="cuda", dtype=DT).expand(batch,n,n)
    s_k = one
    z_k = one
    d = [one.diagonal(dim1=-2,dim2=-1)]
    ds = []
    dl = []
    exposure = None
    for k in range(1,7):
        s_k = torch.bmm(s_k,s)
        z_k = torch.bmm(z_k,z)
        d.append(s_k.diagonal(dim1=-2,dim2=-1))
        if k == 2:
            exposure = torch.bmm(s_k.square(),p.unsqueeze(2)).squeeze(2)
        if k >= 2:
            ds.append(z_k.diagonal(dim1=-2,dim2=-1))
            dl.append(sum(math.comb(k,r)*d[r]*p.pow(k-r) for r in range(k+1))/(2**k))
    static = torch.stack(d[2:], dim=2)
    local = torch.stack(dl, dim=2)
    spatial = torch.stack(ds, dim=2)
    eligible = (grp == 0) & (torch.bmm(adj,p.unsqueeze(2)).squeeze(2)==0)
    count = eligible.sum(1)
    mask = eligible.unsqueeze(2)
    def summarize(bank):
        a = torch.where(mask,bank,0).sum(1) / count.clamp_min(1).unsqueeze(1)
        sd = (torch.where(mask,(bank-a.unsqueeze(1)).square(),0).sum(1)
              / count.clamp_min(1).unsqueeze(1)).sqrt()
        mx = torch.where(mask,bank,-torch.inf).amax(1)
        return torch.cat((a,sd,torch.where(count[:,None]>0,mx,0)),dim=1)
    def pool(bank):
        return torch.bmm(group,bank).flatten(1)/math.sqrt(n)
    def high(bank):
        return torch.cat((pool(bank[:,:,3:]),summarize(bank[:,:,3:])),dim=1)
    common = [torch.full((batch,1),math.log1p(n),device="cuda",dtype=DT),count[:,None]/n]
    for j,key in enumerate(NODE_KEYS):
        card = len(allowable_features[key])
        out = torch.zeros(batch,card,dtype=DT,device="cuda")
        out.scatter_add_(1,x[:,:,j],torch.ones_like(p))
        common.append(out/math.sqrt(n))
    common.extend((edge_bins/math.sqrt(n),torch.bmm(torch.bmm(group,adj),group.transpose(1,2)).flatten(1)/math.sqrt(n),
                   pool(static),pool(torch.cat((local[:,:,:3],spatial[:,:,1:3]),2)),summarize(static)))
    error = torch.where(eligible,(spatial[:,:,3]-local[:,:,3]-exposure/32).abs(),0).max().item()
    n2 = (eligible & (exposure>0)).sum().item()
    return torch.cat(common,dim=1), high(local), high(spatial), (error,count.sum().item(),n2)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("dataset",type=Path)
    cli.add_argument("output",type=Path)
    cli.add_argument("--archive",type=Path)
    args=cli.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    assert "H100" in torch.cuda.get_device_name()
    raw=args.dataset/"raw"
    sizes=csv(raw/"num-node-list.csv.gz",np.int32).ravel()
    edges=csv(raw/"num-edge-list.csv.gz",np.int32).ravel()
    nodes=csv(raw/"node-feat.csv.gz",np.int16)
    edge=csv(raw/"edge.csv.gz",np.int32)
    bond=csv(raw/"edge-feat.csv.gz",np.int16)
    assert len(sizes)==437929 and sizes.sum()==len(nodes) and edges.sum()==len(edge)==len(bond)
    N=len(sizes)
    graph_ids=np.repeat(np.arange(N,dtype=np.int32),edges)
    binned=[]
    for j,key in enumerate(BOND_KEYS):
        dim=len(allowable_features[key])
        assert bond[:,j].min()>=0 and bond[:,j].max()<dim
        idx=graph_ids*dim+bond[:,j]
        binned.append(np.bincount(idx,minlength=N*dim).reshape(N,dim))
    binned=np.concatenate(binned,axis=1)
    del bond,graph_ids,idx
    na=np.r_[0,sizes.cumsum()]
    ea=np.r_[0,edges.cumsum()]
    gtab,ptab=element_groups()
    maxerr=0.
    eligible=focal=done=0
    out=None
    for n0 in sorted(np.unique(sizes)):
        n=int(n0)
        which=np.flatnonzero(sizes==n)
        step=min(384,max(1,2500000//(n*n)))
        for start in range(0,len(which),step):
            idx=which[start:start+step]
            x=nodes[na[idx,None]+np.arange(n)]
            ne=edges[idx]
            g=np.repeat(np.arange(len(idx)),ne)
            local_idx=np.arange(ne.sum())+np.repeat(ea[idx]-np.r_[0,ne.cumsum()[:-1]],ne)
            i,j=edge[local_idx].T
            adj=np.zeros((len(idx),n,n),np.uint8)
            adj[g,i,j]=adj[g,j,i]=1
            assert adj.sum()==2*ne.sum() and (i!=j).all()
            arr=grouped_features(torch.as_tensor(x,dtype=torch.long,device="cuda"),
                torch.as_tensor(adj,dtype=DT,device="cuda"),torch.as_tensor(binned[idx],dtype=DT,device="cuda"),gtab,ptab)
            v,l,h=(t.float().cpu().numpy() for t in arr[:3])
            if out is None:
                out=tuple(np.lib.format.open_memmap(args.output/(name+".npy"),mode="w+",dtype=np.float32,shape=(N,a.shape[1]))
                          for name,a in zip(("common","local","spatial"),(v,l,h)))
            for dest,a in zip(out,(v,l,h)): dest[idx]=a
            err,el,fo=arr[3]
            maxerr=max(maxerr,err)
            eligible+=el
            focal+=fo
            done+=len(idx)
        if n%10==0: print("prepared",done,"molecules",flush=True)
    assert done==N and maxerr<1e-10
    for a in out:a.flush()
    gt=gtab.cpu().numpy()
    pt=ptab.cpu().numpy()
    others=np.unique(nodes[:,0][gt[nodes[:,0]]==5])
    outcome={"n":N,"common_dim":out[0].shape[1],"high_dim":out[1].shape[1],
             "eligible_carbons":eligible,"two_hop_exposed_carbons":focal,
             "eligible_identity_max_abs":maxerr,"other_group_observed_field_values":np.unique(pt[others]).tolist(),
             "max_atoms":int(sizes.max()),"source_sha256":digest(__file__),
             "raw_archive_sha256":digest(args.archive) if args.archive else None}
    (args.output/"feature_audit.json").write_text(json.dumps(outcome,indent=2))
    print("feature cache ready",flush=True)


if __name__=="__main__":
    main()
