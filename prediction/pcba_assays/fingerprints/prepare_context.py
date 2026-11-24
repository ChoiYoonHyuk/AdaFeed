"""Prepare pre-specified context bins and sparse bit fingerprints, server only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

N = 437929
BINS = 64
PRIME = 1000003
GEN = None


def fp_bytes(smiles):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    global GEN
    if GEN is None:
        RDLogger.DisableLog("rdApp.warning")
        GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Official molecule failed RDKit parsing")
    return DataStructs.BitVectToBinaryText(GEN.GetFingerprint(mol))


def fingerprint(data, out):
    import rdkit
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    frame = pd.read_csv(data / "mapping/mol.csv.gz", usecols=lambda key: key.lower()=="smiles")
    assert frame.shape == (N, 1)
    raw = frame.iloc[:,0].tolist()
    test = Chem.MolFromSmiles(raw[0])
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True).GetFingerprint(test)
    bits = np.zeros(2048, np.uint8)
    DataStructs.ConvertToNumpyArray(fp, bits)
    np.testing.assert_array_equal(bits, np.unpackbits(np.frombuffer(fp_bytes(raw[0]),np.uint8), bitorder="little"))
    result = np.lib.format.open_memmap(out / "morgan.npy", mode="w+", dtype=np.uint8, shape=(N,256))
    with ProcessPoolExecutor(max_workers=12) as workers:
        for i, packed in enumerate(workers.map(fp_bytes, raw, chunksize=384)):
            result[i] = np.frombuffer(packed, np.uint8)
    result.flush()
    (out / "morgan_audit.json").write_text(json.dumps({"n":i+1, "radius":2, "bits":2048, "chirality":True, "rdkit":rdkit.__version__}))
    assert i+1 == N
    print("All fingerprints ready.", flush=True)


def fold_fields(fields):
    acc = np.zeros(fields.shape[:-1], dtype=np.int64)
    for r in range(fields.shape[-1]):
        acc = (257 * acc + fields[..., r] + 1) % PRIME
    return acc


def contexts(x, bond, g, i, j):
    atom = fold_fields(x)
    edge = fold_fields(bond)
    total = np.zeros_like(atom)
    square = np.zeros_like(atom)
    degree = np.zeros_like(atom)
    for root, nbr in ((i,j),(j,i)):
        signature = (911*atom[g,nbr] + 353*edge + 1) % PRIME
        np.add.at(total, (g,root), signature)
        np.add.at(square, (g,root), signature**2 % PRIME)
        np.add.at(degree, (g,root), 1)
    return ((1031*atom + 8191*total + 89*square + 17*degree) % PRIME) % BINS


def context_test():
    rng = np.random.default_rng(102)
    x = rng.integers(0,7,(1,9,9))
    i=np.array([0,0,1,3,3,4,5,6,6]);j=np.array([1,2,3,4,8,5,6,7,8])
    e=rng.integers(0,4,(len(i),3));g=np.zeros(len(i),int)
    p=rng.permutation(9);order=rng.permutation(len(i));xp=np.empty_like(x);xp[:,p]=x
    np.testing.assert_array_equal(contexts(x,e,g,i,j),contexts(xp,e[order],g,p[j[order]],p[i[order]])[:,p])


def returns(adj, p, bins):
    batch, n = adj.shape[:2]
    deg = adj.sum(2)
    norm = torch.where(deg>0, deg.clamp_min(1).rsqrt(), 0)
    s = adj * norm.unsqueeze(1) * norm.unsqueeze(2)
    z = (s + torch.diag_embed(p)) / 2
    power = torch.eye(n,device="cuda",dtype=adj.dtype).expand(batch,n,n)
    feedback = power
    d = [power.diagonal(dim1=-2,dim2=-1)]
    spatial,local = [], []
    for k in range(1,7):
        power = torch.bmm(power,s)
        feedback = torch.bmm(feedback,z)
        d.append(power.diagonal(dim1=-2,dim2=-1))
        if k>=2:
            spatial.append(feedback.diagonal(dim1=-2,dim2=-1))
            local.append(sum(math.comb(k,r)*d[r]*p.pow(k-r) for r in range(k+1))/(2**k))
    channels = torch.stack([torch.ones_like(p), *d[2:], *local[:3], *spatial[1:3], *local[3:], *spatial[3:]],2)
    one_hot = torch.nn.functional.one_hot(bins,BINS).to(adj.dtype).transpose(1,2)
    pooled = torch.bmm(one_hot,channels) / math.sqrt(n)
    # Verify bin counts and conservation across all static and feedback channels.
    err = (pooled.sum(1)-channels.sum(1)/math.sqrt(n)).abs().max().item()
    return pooled.float().cpu().numpy(), err


def features(data, out, base):
    from prepare_pcba import csv, element_groups, digest
    context_test()
    raw = data / "raw"
    sizes = csv(raw/"num-node-list.csv.gz",np.int32).ravel()
    edges = csv(raw/"num-edge-list.csv.gz",np.int32).ravel()
    nodes = csv(raw/"node-feat.csv.gz",np.int16)
    edge = csv(raw/"edge.csv.gz",np.int32)
    bond = csv(raw/"edge-feat.csv.gz",np.int16)
    assert len(sizes)==N and sizes.sum()==len(nodes) and edges.sum()==len(edge)==len(bond)
    na,ea = np.r_[0,sizes.cumsum()],np.r_[0,edges.cumsum()]
    _, ptab = element_groups()
    target = np.lib.format.open_memmap(out/"contexts.npy",mode="w+",dtype=np.float32,shape=(N,BINS,15))
    done=0;error=0.;prior_error=0.
    prev={name:np.load(base/"features"/(name+".npy"),mmap_mode="r") for name in ("local","spatial")}
    maxn=int(sizes.max())
    for raw_n in np.unique(sizes):
        n=int(raw_n); ids=np.flatnonzero(sizes==n)
        step=min(384,max(1,2500000//(n*n)))
        for first in range(0,len(ids),step):
            idx=ids[first:first+step]; x=nodes[na[idx,None]+np.arange(n)]
            ne=edges[idx]; g=np.repeat(np.arange(len(idx)),ne)
            ix=np.arange(ne.sum())+np.repeat(ea[idx]-np.r_[0,ne.cumsum()[:-1]],ne)
            i,j=edge[ix].T
            adj=np.zeros((len(idx),n,n),np.uint8);adj[g,i,j]=adj[g,j,i]=1
            assert adj.sum()==2*ne.sum()
            group=contexts(x,bond[ix],g,i,j)
            p=ptab[torch.as_tensor(x[:,:,0].copy(),device="cuda",dtype=torch.long)]
            vals,err=returns(torch.as_tensor(adj,device="cuda",dtype=torch.float64),p,torch.as_tensor(group,device="cuda"))
            error=max(error,err)
            for name,ch in (("local",11),("spatial",13)):
                old=prev[name][idx,:12].reshape(len(idx),6,2).sum(1)
                total=vals[:,:,ch:ch+2].sum(1)
                prior_error=max(prior_error,float(np.abs(old-total).max()))
            target[idx]=vals
            done+=len(idx)
        if n%10==0:print("processed",done,"graphs",flush=True)
    assert done==N and error<1e-9 and prior_error<1e-5
    target.flush()
    old=json.loads((base/"features/feature_audit.json").read_text())
    info={"n":done,"context_bins":BINS,"context_common_channels":11,"higher_channels":2,
          "old_common_dim":old["common_dim"],"old_raw_archive_sha256":old["raw_archive_sha256"],
          "sum_reconstruction_max_abs":error,"old_high_sum_max_abs_float32":prior_error,
          "max_atoms":maxn,"permutation_check":True,"prep_sha256":digest(__file__)}
    (out/"context_audit.json").write_text(json.dumps(info))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["fp","context"]);parser.add_argument("base",type=Path);parser.add_argument("out",type=Path)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.base.resolve()))
    if a.stage=="fp":
        fingerprint(a.base/"pcba",a.out)
    else:
        assert "H100" in torch.cuda.get_device_name()
        torch.set_num_threads(1)
        features(a.base/"pcba",a.out,a.base)
if __name__=="__main__":
    main()
