"""Compute fingerprint side channel and account for RDKit version differences."""
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator

GEN=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def make(smiles):
    mol=Chem.MolFromSmiles(smiles)
    fallback=mol is None
    if fallback:
        mol=Chem.MolFromSmiles(smiles,sanitize=False)
        if mol is None:
            raise ValueError("SMILES syntax unreadable")
        mol.UpdatePropertyCache(strict=False)
        mol=Chem.RemoveHs(mol,sanitize=False)
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
        Chem.AssignStereochemistry(mol,cleanIt=False,force=True)
    bits=DataStructs.BitVectToBinaryText(GEN.GetFingerprint(mol))
    return bits, fallback, mol.GetNumAtoms(), mol.GetNumBonds()


def main():
    data=Path(sys.argv[1]);out=Path(sys.argv[2])
    smi=pd.read_csv(data/"mapping/mol.csv.gz",usecols=lambda c:c.lower()=="smiles").iloc[:,0].tolist()
    N=len(smi)
    assert N==437929
    nodes=pd.read_csv(data/"raw/num-node-list.csv.gz",header=None).to_numpy().ravel()
    edges=pd.read_csv(data/"raw/num-edge-list.csv.gz",header=None).to_numpy().ravel()
    dest=np.lib.format.open_memmap(out/"morgan.npy",mode="w+",dtype=np.uint8,shape=(N,256))
    counts={"fallback_sanitization":0,"atom_count_disagreements":0,"bond_count_disagreements":0}
    with ProcessPoolExecutor(max_workers=12) as pool:
        for i,(packed,fb,atoms,bonds) in enumerate(pool.map(make,smi,chunksize=384)):
            dest[i]=np.frombuffer(packed,np.uint8)
            counts["fallback_sanitization"]+=int(fb)
            counts["atom_count_disagreements"]+=int(atoms!=nodes[i])
            counts["bond_count_disagreements"]+=int(bonds!=edges[i])
            if (i+1)%100000==0: print("prepared",i+1,"fingerprints",flush=True)
    assert i+1==N
    sample=Chem.MolFromSmiles(smi[0]);bits=np.empty(2048,np.uint8)
    DataStructs.ConvertToNumpyArray(GEN.GetFingerprint(sample),bits)
    np.testing.assert_array_equal(bits,np.unpackbits(dest[0],bitorder="little"))
    dest.flush()
    audit={"n":N,"bits":2048,"radius":2,"chirality":True,"rdkit":rdBase.rdkitVersion,**counts}
    (out/"morgan_audit.json").write_text(json.dumps(audit))
    print(audit,flush=True)
if __name__=="__main__":
    main()
