"""Compute both paired contrasts, exact grouped AP deletions and scaffold sensitivity."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
from sklearn.metrics import average_precision_score
import ogb
from prepare_pcba import digest
from train_pcba import ARMS,SEEDS,labels,split_arrays

G=100


def scaffold(smiles):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid official mapping SMILES")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=True)


def groupings(dataset,y,te):
    mapping=pd.read_csv(dataset/"mapping"/"mol.csv.gz",low_memory=False)
    assert len(mapping)==437929
    cols=[c for c in mapping.columns if c.upper().startswith("PCBA-")]
    assert len(cols)==128
    np.testing.assert_equal(mapping.iloc[te][cols].to_numpy(dtype=np.float32),y)
    smi=[c for c in mapping.columns if c.lower()=="smiles"]
    assert len(smi)==1
    smiles=mapping[smi[0]].to_numpy()[te]
    del mapping
    with ProcessPoolExecutor(max_workers=12) as pool:
        scaf=list(pool.map(scaffold,smiles,chunksize=160))
    counts=Counter(scaf)
    bins=np.zeros(G,np.int64)
    by_scaffold={}
    for key,n in sorted(counts.items(),key=lambda t:(-t[1],t[0])):
        g=int(bins.argmin())
        by_scaffold[key]=g
        bins[g]+=n
    assert bins.min()>0
    sk=np.array([by_scaffold[k] for k in scaf],dtype=np.int64)
    gen=np.random.default_rng(160331)
    mol=np.empty(len(y),dtype=np.int64)
    mol[gen.permutation(len(y))]=np.arange(len(y))%G
    return (mol,sk),{"molecules":len(y),"num_scaffolds":len(counts),
        "max_scaffold_size":max(counts.values()),"empty_scaffold_molecules":counts.get("",0),
        "molecule_bin_sizes":np.bincount(mol,minlength=G).tolist(),"scaffold_bin_sizes":bins.tolist()}


def deletion_ap(y,score,ids,do_check=False):
    # At each distinct positive-score threshold, tabulate molecules above it and positives tied there.
    order=np.argsort(-score,kind="stable")
    neg=-score[order]
    truth=y[order]
    pos=np.flatnonzero(truth)
    a=len(pos)
    if a==0 or a==len(truth):return np.full(1+G*len(ids),np.nan)
    rank=np.searchsorted(neg,neg[pos],side="right")
    ends,threshold_index,count=np.unique(rank,return_inverse=True,return_counts=True)
    full=float((count.cumsum()/ends*count).sum()/a)
    interval=np.searchsorted(ends,np.arange(len(truth)),side="right")
    nb=len(ends)
    deleted=[]
    for partition in ids:
        g=partition[order]
        minus=np.bincount(interval*G+g,minlength=(nb+1)*G).reshape(nb+1,G)[:nb]
        denom=ends[:,None]-minus.cumsum(0)
        tied_drop=np.bincount(threshold_index*G+g[pos],minlength=nb*G).reshape(nb,G)
        tied=count[:,None]-tied_drop
        true_pos=tied.cumsum(0)
        numerator=(tied*true_pos/np.maximum(denom,1)).sum(0)
        positives=true_pos[-1]
        total=len(truth)-np.bincount(g,minlength=G)
        valid=(positives>0)&(positives<total)
        out=np.where(valid,numerator/np.maximum(positives,1),np.nan)
        if do_check:
            assert abs(average_precision_score(y,score)-full)<1e-12
            for k in (0,17,99):
                keep=partition!=k
                if valid[k]:
                    assert abs(average_precision_score(y[keep],score[keep])-out[k])<1e-12
        deleted.extend(out)
    return np.r_[full,deleted]


def ci_groups(point,theta):
    avg=theta.mean()
    se=np.sqrt((G-1)/G*np.square(theta-avg).sum())
    half=float(stats.t.ppf(.975,G-1)*se)
    return [float(point-half),float(point+half)]


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("dataset",type=Path)
    cli.add_argument("features",type=Path)
    cli.add_argument("work",type=Path)
    cli.add_argument("output",type=Path)
    a=cli.parse_args()
    assert not a.output.exists()
    fits=json.loads((a.work/"fit_summary.json").read_text())
    audit=json.loads((a.features/"feature_audit.json").read_text())
    tr,va,te=split_arrays(a.dataset)
    y=labels(a.dataset)[te]
    parts,structure=groupings(a.dataset,y,te)
    scores={}
    support=(y==1).sum(0)
    observed=np.isfinite(y)
    elig=(support>0)&((y==0).sum(0)>0)
    usable=None
    for arm in ARMS:
        curves=[]
        for i,seed in enumerate(SEEDS):
            pred=np.load(a.work/f"{arm}-{seed}-test.npy",mmap_mode="r")
            per=[]
            for j in range(y.shape[1]):
                mask=observed[:,j]
                per.append(deletion_ap(y[mask,j],pred[mask,j],[g[mask] for g in parts],
                                       do_check=(i==0 and j%43==0)))
            per=np.array(per)
            count=np.isfinite(per).sum(0)
            if usable is None:usable=count
            np.testing.assert_array_equal(count,usable)
            curve=np.nanmean(per,axis=0)
            assert abs(curve[0]-fits["official_seed_ap"][arm][i])<1e-11
            curves.append(curve)
        scores[arm]=np.array(curves)
        print("checked deletions",arm,flush=True)
    paired={}
    for comparator in ("Local","Common"):
        delta=scores["Spatial"]-scores[comparator]
        d=delta[:,0]
        mean=float(d.mean())
        half=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
        paired[comparator]={"mean":mean,"per_seed":d.tolist(),"positive_fits":int((d>0).sum()),
            "seed95":[float(mean-half),float(mean+half)],
            "molecule95":ci_groups(mean,delta[:,1:1+G].mean(0)),
            "scaffold95":ci_groups(mean,delta[:,1+G:].mean(0))}
    decisions=[v[k][0]>0 for v in paired.values() for k in ("seed95","molecule95")]
    out={"benchmark":"ogbg-molpcba","official_metric":"equal-weight mean AP across eligible assays",
         "feature_audit":audit,"fit":fits,"test_structure":structure,"eligible_assays":int(elig.sum()),
         "eligible_after_delete":{"molecule":[int(usable[1:1+G].min()),int(usable[1:1+G].max())],
                                  "scaffold":[int(usable[1+G:].min()),int(usable[1+G:].max())]},
         "observed_test_labels":int(observed.sum()),"positive_test_labels":int(support.sum()),
         "test_positive_support_min_median_max":[int(support[elig].min()),float(np.median(support[elig])),int(support[elig].max())],
         "test_ap":{x:float(arr[:,0].mean()) for x,arr in scores.items()},
         "spatial_minus":paired,"joint_superiority":bool(all(decisions)),
         "source_sha256":digest(__file__),"ogb_version":ogb.__version__,"sklearn_version":sklearn.__version__}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print("finished aggregate-only evaluation",flush=True)


if __name__=="__main__":
    main()
