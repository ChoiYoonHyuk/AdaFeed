"""Close the seven strict-valence failures without weakening source-scaffold exclusion."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import sys,json
import numpy as np
import pandas as pd
from rdkit import Chem,RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold as MS
RDLogger.DisableLog("rdApp.warning");RDLogger.DisableLog("rdApp.error")


def invariant(m):
    # This is necessary for core isomorphism even after dropping chirality, charge and bond order.
    return tuple(sorted((a.GetAtomicNum(),a.GetDegree(),tuple(sorted(n.GetAtomicNum() for n in a.GetNeighbors()))) for a in m.GetAtoms()))


def pair(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None:
        m=Chem.MolFromSmiles(smiles,sanitize=False)
        if m is None:raise ValueError("Missing molecular graph")
        Chem.SanitizeMol(m,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    # These are RDKit's exact GetScaffoldForMol steps; strict valence is only a cache validation.
    core=Chem.MurckoDecompose(m);core.ClearComputedProps();core.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(core)
    relaxed=Chem.MolToSmiles(core,isomericSmiles=True)
    try:
        std=MS.GetScaffoldForMol(m)
        key=Chem.MolToSmiles(std,isomericSmiles=True)
        assert key==relaxed and invariant(std)==invariant(core)
    except Chem.AtomValenceException:key=None
    return key,relaxed,invariant(core) if key is None else None


def candidate(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None:raise ValueError("Destination chemistry alignment changed")
    core=MS.GetScaffoldForMol(m)
    key=Chem.MolToSmiles(core,isomericSmiles=True)
    return key,invariant(core)


def main():
    s,d,out=map(Path,sys.argv[1:4]);a=json.loads((out/"scaffold_audit.json").read_text())
    src=pd.read_csv(s/"mapping/mol.csv.gz",usecols=["smiles"]).smiles
    di=pd.read_csv(d/"split/scaffold/test.csv.gz",header=None).to_numpy().ravel()
    dst=pd.read_csv(d/"mapping/mol.csv.gz",usecols=["smiles"]).smiles.to_numpy()[di]
    with ProcessPoolExecutor(max_workers=10) as pool:
        sr=list(pool.map(pair,src,chunksize=96))
        unresolved=[v for v in sr if v[0] is None]
        std={v[0] for v in sr if v[0] is not None}
        relaxed={v[1] for v in unresolved}
        signatures={v[2] for v in unresolved}
        possible=Counter();canon_overlap=0
        destination=[]
        for key,inv in pool.map(candidate,dst,chunksize=96):
            destination.append(key)
            if inv in signatures:possible[inv]+=1
            canon_overlap+=int(key in relaxed)
    assert len(destination)==43793 and len(unresolved)==a["source_unresolved"]==7
    assert len(std)==a["source_distinct_known"]
    ok=np.array([k not in std for k in destination])
    unique,cl=np.unique(np.array(destination)[ok],return_inverse=True)
    report={"strict_valid_source":len(sr)-len(unresolved),"source_valence_failures":len(unresolved),
            "recovered_core_invariant_count":len(signatures),"recovered_distinct_keys":len(relaxed),
            "all_candidate_core_graphs_checked":len(destination),"invariant_collision_count":sum(possible.values()),
            "relaxed_canonical_matches":canon_overlap,"strict_comparable_sources_verified":len(sr)-len(unresolved),
            "retained_n":int(ok.sum()),"retained_scaffolds":len(unique),"remaining_possible_overlap":sum(possible.values())}
    (out/"scaffold_certificate.json").write_text(json.dumps(report,indent=2))
    print(report,flush=True)
    assert not possible and canon_overlap==0
    assert int(ok.sum())==a["retained_n"] and len(unique)==a["retained_scaffolds"]
    np.savez(out/"selected_server_only.npz",graph=di[ok],scaffold=cl)
if __name__=="__main__":main()
