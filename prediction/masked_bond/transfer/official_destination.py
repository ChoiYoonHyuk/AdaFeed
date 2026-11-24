"""Validate the declared target with an independent official graph loader on all retained graphs."""
import sys
from pathlib import Path
import json
import numpy as np
from rdkit import Chem
from ogb.io.read_graph_raw import read_csv_graph_raw
from ogb.utils.features import allowable_features
from transfer_features import Raw,build
from label_diagnosis import target


def main():
    src,cache=map(Path,sys.argv[1:3])
    state=np.load(cache/"selected_server_only.npz");graphs,sc=state["graph"],state["scaffold"]
    audit=json.loads((cache/"label_diagnosis.json").read_text())
    assert audit["eligible_mismatch_d2"]==audit["eligible_mismatch_d3"]==0
    assert json.loads((cache/"scaffold_certificate.json").read_text())["remaining_possible_overlap"]==0
    assert json.loads((cache/"source_alignment.json").read_text())["source_checkpoint_replay_max"]==0
    raw=Raw(src)
    official=read_csv_graph_raw(str(src/"raw"),add_inverse_edge=True)
    assert len(official)==437929
    typ=allowable_features["possible_bond_type_list"].index("DOUBLE")
    errs={k:0 for k in ("official_node_mismatch","official_edge_or_type_mismatch","official_double_mismatch",
                      "rdkit_input_topology_or_element_mismatch","rdkit_eligible_target_mismatch")}
    full_bad=affected=eligible=full_atoms=0;maxerr=0.;blocks=[]
    for first in range(0,len(graphs),1024):
        block={}
        for r,g in enumerate(graphs[first:first+1024],first):
            z=official[g];elem,edge,bond=raw.molecule(g)
            og_elem=z["node_feat"][:,0]+1;n=z["num_nodes"]
            oe=z["edge_index"];ob=z["edge_feat"][:,0]
            assert oe.shape==(2,2*len(edge)) and n==len(elem) and len(ob)==oe.shape[1]
            if not np.array_equal(og_elem,elem):errs["official_node_mismatch"]+=1
            if not np.array_equal(oe[:,::2],edge.T) or not np.array_equal(oe[:,1::2],edge.T[::-1]) or not np.array_equal(ob,np.repeat(bond,2)):
                errs["official_edge_or_type_mismatch"]+=1
            oy=np.bincount(oe[0,ob==typ],minlength=n)>0
            record,err=build(elem,edge,bond);maxerr=max(maxerr,err)
            if not np.array_equal(oy,record["y"]):errs["official_double_mismatch"]+=1
            m=Chem.MolFromSmiles(raw.smi[g])
            assert m is not None
            inds={tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in m.GetBonds()}
            oeidx={tuple(sorted(pair)) for pair in edge}
            if inds!=oeidx or not np.array_equal(og_elem,[a.GetAtomicNum() for a in m.GetAtoms()]):
                errs["rdkit_input_topology_or_element_mismatch"]+=1
            raw_bad=target(m)!=oy;affected+=int(raw_bad.sum());full_bad+=int(raw_bad.any())
            active=record["carbon"]&np.isin(record["dist"],[2,3])
            errs["rdkit_eligible_target_mismatch"]+=int((active&raw_bad).sum())
            eligible+=int(active.sum());full_atoms+=n
            record["graph"]=np.full(n,g,np.int32);record["scaffold"]=np.full(n,sc[r],np.int32)
            for k,v in record.items():
                if k!="carbon":block.setdefault(k,[]).append(v[active])
        blocks.append({k:np.concatenate(v) for k,v in block.items()})
        if len(blocks)%10==0:print("OGB/SMILES aligned",min(first+1024,len(graphs)),flush=True)
    alignment={"graphs":len(graphs),"all_atoms":full_atoms,"scored_root_count":eligible,
        "outside_strata_rdkit_aromaticity_graphs":full_bad,"outside_strata_rdkit_aromaticity_atoms":affected,
        "identity_max":maxerr,"official_loader":"ogb.io.read_graph_raw.read_csv_graph_raw(add_inverse_edge=True)",**errs}
    (cache/"official_label_certificate.json").write_text(json.dumps(alignment,indent=2))
    assert full_bad==audit["affected_molecules"] and affected==audit["affected_atoms"]
    assert eligible==385670 and maxerr<1e-10 and not any(errs.values())
    print(alignment,flush=True)
    data={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    out=cache/"distance_destination_server_only.npz"
    tmp=out.with_suffix(".tmp")
    with tmp.open("wb") as f:np.savez(f,**data)
    tmp.replace(out)
if __name__=="__main__":main()
