"""Diagnose source-format differences before any destination score or requirement change."""
import sys,json
from pathlib import Path
from collections import Counter
import numpy as np
from rdkit import Chem
from transfer_features import Raw,min_dist_bfs


def target(m):
    y=np.zeros(m.GetNumAtoms(),np.int8)
    for b in m.GetBonds():
        if b.GetBondType()==Chem.BondType.DOUBLE:y[b.GetBeginAtomIdx()]=y[b.GetEndAtomIdx()]=1
    return y


def main():
    dst,cache=map(Path,sys.argv[1:3]);raw=Raw(dst);ids=np.load(cache/"selected_server_only.npz")["graph"]
    counters=Counter()
    for g in ids:
        e,edge,code=raw.molecule(g);n=len(e);y=np.zeros(n,np.int8)
        hit=code==1;i,j=edge.T;y[i[hit]]=y[j[hit]]=1
        m=Chem.MolFromSmiles(raw.smi[g]);parsed=target(m)
        if np.array_equal(y,parsed):continue
        counters["affected_molecules"]+=1
        a=np.zeros((n,n),bool);a[i,j]=a[j,i]=True
        d=min_dist_bfs(a,(e!=1)&(e!=6))
        different=y!=parsed
        counters["affected_atoms"]+=int(different.sum())
        for k in (2,3):
            ix=different&(e==6)&(d==k)
            counters[f"eligible_mismatch_d{k}"]+=int(ix.sum())
            counters[f"eligible_molecules_d{k}"]+=int(ix.any())
        bonded={(min(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),max(b.GetBeginAtomIdx(),b.GetEndAtomIdx())):str(b.GetBondType()) for b in m.GetBonds()}
        raw_pair={(min(i,j),max(i,j)):str(("SINGLE","DOUBLE","TRIPLE","AROMATIC","misc")[c]) for (i,j),c in zip(edge,code)}
        counters["topology_disagreement"]+=int(set(bonded)!=set(raw_pair))
        for b,z in raw_pair.items():
            if bonded.get(b)!=z:counters["transition_"+z+"_to_"+str(bonded.get(b))]+=1
        partial=Chem.MolFromSmiles(raw.smi[g],sanitize=False)
        skip=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        Chem.SanitizeMol(partial,sanitizeOps=skip)
        counters["skip_aromaticity_full_graph_matches"]+=int(np.array_equal(y,target(partial)))
    (cache/"label_diagnosis.json").write_text(json.dumps(dict(counters),indent=2));print(dict(counters),flush=True)
if __name__=="__main__":main()
