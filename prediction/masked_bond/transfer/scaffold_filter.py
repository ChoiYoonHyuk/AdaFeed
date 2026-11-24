"""Compute a full-source scaffold exclusion before any external model scoring."""
from pathlib import Path
import sys
import json
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def key(smiles):
    m=Chem.MolFromSmiles(smiles)
    fallback=m is None
    if fallback:
        m=Chem.MolFromSmiles(smiles,sanitize=False)
        if m is not None:
            try:
                Chem.SanitizeMol(m,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except (ValueError,RuntimeError):
                m=None
    if m is None:return None,fallback
    try:return MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=True),fallback
    except (ValueError,RuntimeError):return None,fallback


def main():
    src,dst,out=map(Path,sys.argv[1:4]);out.mkdir(exist_ok=True,parents=True)
    read=lambda p:pd.read_csv(p/"mapping/mol.csv.gz",usecols=["smiles"]).smiles.to_numpy()
    s=read(src); dx=read(dst)
    assert len(s)==41127 and len(dx)==437929
    destination=pd.read_csv(dst/"split/scaffold/test.csv.gz",header=None).to_numpy().ravel()
    assert len(destination)==43793 and len(set(destination))==43793
    src_parts=[pd.read_csv(src/"split/scaffold"/(n+".csv.gz"),header=None).to_numpy().ravel() for n in ("train","valid","test")]
    assert np.array_equal(np.sort(np.concatenate(src_parts)),np.arange(len(s)))
    with ProcessPoolExecutor(max_workers=10) as pool:
        sp=list(pool.map(key,s,chunksize=128));dp=list(pool.map(key,dx[destination],chunksize=128))
    source=[k for k,fb in sp];dest=[k for k,fb in dp]
    keys=set(source);ds=set(dest)
    keep=np.array([k is not None and k not in keys for k in dest])
    accepted=[k for k,choose in zip(dest,keep) if choose]
    uniq,group=np.unique(accepted,return_inverse=True)
    assert not (set(uniq)&keys)
    audit={"rdkit":rdBase.rdkitVersion,"source_n":len(s),"source_split_counts":[len(v) for v in src_parts],
        "source_distinct_known":len(keys-{None}),"source_fallback":sum(f for k,f in sp),"source_unresolved":source.count(None),
        "source_empty":source.count(""),"candidate_n":len(dest),"candidate_distinct_known":len(ds-{None}),
        "candidate_fallback":sum(f for k,f in dp),"candidate_unresolved":dest.count(None),"candidate_empty":dest.count(""),
        "common_distinct_known":len((keys&ds)-{None}),"excluded_source_scaffold":sum(k is not None and k in keys for k in dest),
        "retained_n":int(keep.sum()),"retained_scaffolds":len(uniq),"remaining_overlap":0}
    (out/"scaffold_audit.json").write_text(json.dumps(audit,indent=2))
    print(audit,flush=True)
    assert len(accepted)>0 and source.count(None)==0
    assert sum(audit[k] for k in ("candidate_unresolved","excluded_source_scaffold","retained_n"))==len(destination)
    np.savez(out/"selected_server_only.npz",graph=destination[keep],scaffold=group)
if __name__=="__main__":
    main()
