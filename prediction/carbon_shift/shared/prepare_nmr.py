"""Generate coordinates-free Exp5K inputs, original molecule split and measured targets."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from itertools import chain
import json
import hashlib
import gzip
import numpy as np
import pandas as pd
import requests
from rdkit import Chem,rdBase


PIN = '4e6f5a9e50162c80cf2382397bdfa41334ff2e0d'
SOURCES = {
    'data/Exp5K/Exp5K.csv.gz':'eb9d8126774b2398699fde34eb39715152cff6d6d89e09521d480f60a1046dc7',
    'data/Exp5K/FF.sdf.gz':'5c76d7aac59d18f3e6c977dedbed84893b256a6b2c17730628e27232bbc965a8',
    'code/predicting_model/C/ExpNN-ff/preprocess_pubchem.py':'f890ca2625cefab2b1e693952dff3e8fe6522adfd6272d0bc208c897737bdbe5'}
ELEMENT = (5,6,7,8,9,14,15,16,17,35,53)
BOND = ('SINGLE','DOUBLE','TRIPLE','AROMATIC')
HYB = ('UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2')
ORDERS = (3,5,7)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one(value, bins):
    row = np.zeros(len(bins)+1)
    row[bins.index(value) if value in bins else len(bins)] = 1
    return row


def vertex(mol, heavy, ids, adj):
    elem = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in heavy])
    group = np.stack([one(z,ELEMENT) for z in elem])
    desc = []
    deg = adj.sum(1)
    for i, ix in enumerate(heavy):
        atom = mol.GetAtomWithIdx(ix)
        nh = atom.GetTotalNumHs(includeNeighbors=True)
        desc.append(np.concatenate([group[i],one(int(deg[i]),(0,1,2,3,4,5,6)),
                one(nh,(0,1,2,3,4,5,6)),one(atom.GetFormalCharge(),(-3,-2,-1,0,1,2,3)),
                one(str(atom.GetHybridization()),HYB),[float(atom.GetIsAromatic()),float(atom.IsInRing())]]))
    desc = np.array(desc)
    neigh = np.zeros((len(heavy),len(BOND)+1,desc.shape[1]))
    for bond in mol.GetBonds():
        i,j = bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if i not in ids or j not in ids:
            continue
        i,j = ids[i],ids[j]
        k = str(bond.GetBondType())
        t = BOND.index(k) if k in BOND else len(BOND)
        neigh[i,t] += desc[j]
        neigh[j,t] += desc[i]
    return elem,group,desc,neigh.reshape(len(heavy),-1)


def encode(mol, indexed):
    # Atom indices refer to the full SDF including H. Use them before heavy-graph reindexing.
    mol.RemoveAllConformers()
    Chem.SanitizeMol(mol)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()>1]
    ids = {v:i for i,v in enumerate(heavy)}
    root = np.array([ids[i] for i in indexed],dtype=np.int64)
    assert all(mol.GetAtomWithIdx(int(i)).GetAtomicNum()==6 for i in indexed)
    n = len(heavy); adj = np.zeros((n,n))
    for e in mol.GetBonds():
        i,j=e.GetBeginAtomIdx(),e.GetEndAtomIdx()
        if i in ids and j in ids:
            adj[ids[i],ids[j]]=adj[ids[j],ids[i]]=1.
    elem,group,desc,neighbors = vertex(mol,heavy,ids,adj)
    deg=adj.sum(1)
    inv=np.divide(1.,np.sqrt(deg),out=np.zeros_like(deg),where=deg>0)
    s=inv[:,None]*adj*inv[None,:]
    smat=[np.eye(n)]
    for _ in range(8):
        smat.append(smat[-1]@s)
    static=np.stack([m.diagonal() for m in smat[2:]],1)
    seen=np.eye(n,dtype=bool)
    front=seen
    dist=np.where(elem==6,4,0)
    shell=[]
    for d in (1,2,3):
        front=(front@adj>0)&~seen
        seen|=front
        dist[(dist==4)&((front@(elem!=6))>0)]=d
        if d>=2:
            shell.append(front@group)
    common=np.column_stack([desc,neighbors,np.broadcast_to(group.mean(0),group.shape),
                            np.full(n,np.log1p(n)),*shell,static])
    fields=[elem==6,elem==7,elem==8,np.isin(elem,(9,17,35,53))]
    fields.append(~np.logical_or.reduce(fields))
    assert np.all(np.sum(fields,0)==1)
    loc,spat,check=[],[],[]
    for p in fields:
        p=p.astype(float)
        shifted=s+np.diag(p)
        direct=np.eye(n)
        # For carbon roots p_i is constant within each field; Local powers are evaluated
        # independently as powers of S+p_i I for the common carbon value.
        localmat=s+p[root[0]]*np.eye(n)
        ld=np.eye(n)
        terms=[np.eye(n)]  # W(k,q), initialized with zero pauses.
        for k in range(1,8):
            direct=direct@shifted
            ld=ld@localmat
            new=[w@s for w in terms]+[np.zeros_like(direct)]
            for q,w in enumerate(terms):new[q+1]+=w*p[None,:]
            terms=new
            if k in ORDERS:
                den=2**k
                spat.append(direct.diagonal()[root]/den)
                loc.append(ld.diagonal()[root]/den)
                check.append(sum(w.diagonal()[root] for w in terms)/den)
    spatial=np.column_stack(spat); recon=np.column_stack(check)
    np.testing.assert_allclose(spatial,recon,atol=3e-15,rtol=1e-13)
    identity=Chem.MolToSmiles(Chem.RemoveHs(mol),canonical=True,isomericSmiles=False)
    return (common[root].astype(np.float32),np.column_stack(loc).astype(np.float32),
            spatial.astype(np.float32),recon.astype(np.float32),dist[root],identity,float(abs(spatial-recon).max()))


def main():
    parser=ArgumentParser(description=__doc__); parser.add_argument('folder',type=Path)
    parser.add_argument('--fetch',action='store_true',help='Fetch only pinned author files onto the benchmark server')
    a=parser.parse_args(); d=a.folder.resolve(); d.mkdir(exist_ok=True,parents=True)
    for file,hashval in SOURCES.items():
        dest=d/file
        if a.fetch and not dest.exists():
            dest.parent.mkdir(exist_ok=True,parents=True)
            r=requests.get('https://raw.githubusercontent.com/patonlab/CASCADE/'+PIN+'/'+file,timeout=60)
            r.raise_for_status(); dest.write_bytes(r.content)
        assert sha(dest)==hashval,file
    df=pd.read_csv(d/'data/Exp5K/Exp5K.csv.gz',index_col=0)
    df=df.loc[df.atom_type==6].copy()
    assert df.Shift.notna().all() and np.isfinite(df.Shift).all()
    dupe=int(df.duplicated(['mol_id','atom_index']).sum()); before=len(df)
    df=df.groupby(['mol_id','atom_index'],sort=True).Shift.mean().reset_index()
    assert before==53331 and dupe==0
    ymap={int(k):g for k,g in df.groupby('mol_id',sort=True)}
    grouped=pd.DataFrame({'mol_id':list(ymap)})
    test=grouped.sample(n=500,random_state=666)
    valid=grouped[~grouped.mol_id.isin(test.mol_id)].sample(n=500,random_state=666)
    train=grouped[~grouped.mol_id.isin(set(test.mol_id)|set(valid.mol_id))]
    belong={int(i):k for k,part in enumerate([train,valid,test]) for i in part.mol_id}
    expected_id_hash=[hashlib.sha256(np.sort(g.mol_id.to_numpy(dtype='<i8')).tobytes()).hexdigest()
                      for g in (train,valid,test)]
    blocks={k:[] for k in ('common','local','spatial','reconstruction','distance','target','graph','split','atom')}
    seen=set(); rowgroups={}; total=0; nrepeat=0; unused=0; maxerr=0.
    with gzip.open(d/'data/Exp5K/FF.sdf.gz','rb') as f:
        for mol in Chem.ForwardSDMolSupplier(f,sanitize=False,removeHs=False):
            total+=1; assert mol is not None
            key=int(mol.GetProp('_Name'))
            if key in seen:
                nrepeat+=1; continue
            seen.add(key)
            if key not in ymap:
                unused+=1;continue
            g=ymap[key]; ix=g.atom_index.to_numpy(np.int64); y=g.Shift.to_numpy(np.float64)
            common,local,spatial,recon,dist,identity,err=encode(mol,ix)
            rowgroups[key]=identity; maxerr=max(maxerr,err)
            for k,val in zip(blocks,(common,local,spatial,recon,dist,y,np.full(len(ix),key),
                                     np.full(len(ix),belong[key],np.int8),ix),strict=True):
                blocks[k].append(val)
            if len(rowgroups)%1000==0: print('prepared',len(rowgroups),flush=True)
    arrays={k:np.concatenate(v) for k,v in blocks.items()}; graph=arrays['graph']; split=arrays['split']
    prior_keys={key for g,key in rowgroups.items() if belong[g]!=2}
    arrays['input_seen']=np.array([rowgroups[g] in prior_keys for g in graph])
    groups=[np.unique(graph[split==k]) for k in (0,1,2)]
    groups_n=[len(g) for g in groups]; atoms=[int((split==k).sum()) for k in (0,1,2)]
    assert groups_n==[4140,498,500] and atoms==[42869,5225,5156]
    np.savez(d/'features_server_only.npz',**arrays)
    stats=dict(pin=PIN,source_sha256=SOURCES,prepare_sha256=sha(__file__),pandas=pd.__version__,rdkit=rdBase.rdkitVersion,
               input_files_sha256=SOURCES,original_sampled_id_sha256=expected_id_hash,
               available_id_sha256=[hashlib.sha256(g.astype('<i8').tobytes()).hexdigest() for g in groups],
               available_graph_atom_sha256=[hashlib.sha256(np.column_stack((graph[split==k],arrays['atom'][split==k])).astype('<i8').tobytes()).hexdigest() for k in (0,1,2)],
               sdf_records=total,duplicate_structures_first_retained=nrepeat,sdf_ids_without_label=unused,
               csv_molecule_count=len(belong),missing_sdf_ids=len(set(belong)-seen),duplicate_shift_rows=dupe,
               atoms=atoms,molecules=groups_n,common_width=arrays['common'].shape[1],shift_width=arrays['local'].shape[1],
               fp64_marked_expansion_max=maxerr,
               max_fp32_marked_expansion_delta=float(abs(arrays['spatial']-arrays['reconstruction']).max()),
               fp32_marked_expansion_unequal_entries=int(np.count_nonzero(arrays['spatial']!=arrays['reconstruction'])),
               duplicate_test_molecules=len(np.unique(graph[(split==2)&arrays['input_seen']])),
               duplicate_test_atoms=int(np.sum((split==2)&arrays['input_seen'])))
    (d/'input_aggregate.json').write_text(json.dumps(stats,indent=2,allow_nan=False)+'\n')
    print('prepared all sources',flush=True)

if __name__=='__main__':main()
