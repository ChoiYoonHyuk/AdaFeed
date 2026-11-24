"""Construct nested connected induced subgraphs of the official ogbn-arxiv archive."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, breadth_first_order

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('edge_file', type=Path, help='Official ogbn-arxiv raw/edge.csv.gz')
raw = parser.parse_args().edge_file
out = Path(__file__).resolve().parent
sizes = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
seed = 20261122
with gzip.open(raw, "rt") as f:
    edges = np.loadtxt(f, dtype=np.int32, delimiter=",").T
n = 169343
g = sparse.coo_matrix((np.ones(len(edges[0]), dtype=np.int8), edges), shape=(n, n)).tocsr()
g.data[:] = 1
g = g.maximum(g.T)
g.setdiag(0)
g.eliminate_zeros()
g.sort_indices()
_, component = connected_components(g, directed=False)
large = np.flatnonzero(component == np.bincount(component).argmax())
root = int(np.random.default_rng(seed).choice(large))
order = breadth_first_order(g, root, directed=False, return_predecessors=False)
meta = dict(dataset="ogbn-arxiv", seed=seed, root=root, input_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), n_original=n, m_clean=g.nnz//2, largest_component=len(large), graphs={})
for size in sizes:
    idx = order[:size]
    induced = g[idx, :][:, idx].tocsr()
    induced.sort_indices()
    sparse.save_npz(out / f"graph{size}.npz", induced)
    meta["graphs"][size] = dict(edges=induced.nnz//2, ids_sha256=hashlib.sha256(idx.astype("<i4").tobytes()).hexdigest())
(out / "design.json").write_text(json.dumps(meta, indent=2))
print(meta)
