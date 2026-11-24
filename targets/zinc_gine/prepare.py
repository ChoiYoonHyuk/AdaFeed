"""Build padded tensors from the official ZINC archive and subset indices.

Run on the experiment server, in the directory containing molecules.zip and
train.index, val.index, test.index. All molecular records stay on the server.
"""
import gc
import hashlib
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare():
    nmax = 40
    splits = {"train": 10_000, "val": 1_000, "test": 1_000}
    atom = np.zeros((sum(splits.values()), nmax), dtype=np.int16)
    bond = np.zeros((*atom.shape, nmax), dtype=np.int8)
    y = np.zeros(len(atom), dtype=np.float32)
    offset = 0
    metadata = {}
    with zipfile.ZipFile("molecules.zip") as archive:
        for split, expected in splits.items():
            index_path = f"{split}.index"
            index = [int(i) for i in Path(index_path).read_text().strip().strip(",").split(",")]
            assert len(index) == len(set(index)) == expected
            with archive.open(f"molecules/{split}.pickle") as f:
                molecules = pickle.load(f)
            assert min(index) >= 0 and max(index) < len(molecules)
            for j, idx in enumerate(index, start=offset):
                molecule = molecules[idx]
                atoms = np.asarray(molecule["atom_type"]).reshape(-1)
                edges = np.asarray(molecule["bond_type"])
                n = len(atoms)
                assert n <= nmax and edges.shape == (n, n) and np.array_equal(edges, edges.T)
                assert np.min(atoms) >= 0 and np.max(atoms) < 28
                assert np.min(edges) == 0 and np.max(edges) <= 4 and np.diag(edges).sum() == 0
                atom[j, :n] = atoms + 1
                bond[j, :n, :n] = edges
                y[j] = float(molecule["logP_SA_cycle_normalized"])
            metadata[split] = {"split_count": expected, "raw_split_count": len(molecules),
                               "index_sha256": sha256(index_path)}
            offset += expected
            del molecules
            gc.collect()
    assert np.isfinite(y).all()
    torch.save((torch.from_numpy(atom), torch.from_numpy(bond), torch.from_numpy(y)), "zinc_on_server.pt")
    metadata["archive_sha256"] = sha256("molecules.zip")
    metadata["archive_url"] = "https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1"
    metadata["indices_url"] = "https://raw.githubusercontent.com/graphdeeplearning/benchmarking-gnns/master/data/molecules/{}.index"
    metadata["target"] = "logP_SA_cycle_normalized"
    Path("split_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    prepare()
