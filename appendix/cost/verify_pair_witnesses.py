"""Certify the floating-point positive decisions using exact integer residues of adjacency powers."""
from pathlib import Path
import json
import sys
import brec_stable_cost as bench
import numpy as np
from scipy.sparse import csr_matrix

prime = 1_000_003


def separated(labels):
    n = len(labels[0])
    nc = 1
    colors = np.zeros(2*n, np.int32)
    while True:
        rows = []
        for side, tok in enumerate(labels):
            local = colors[side*n:(side+1)*n]
            enc = tok.astype(np.int64)*nc + local[None,:]
            enc.sort(axis=1)
            rows.append(np.column_stack((local, enc)))
        colors, new_count = bench.ranks(np.vstack(rows))
        if not np.array_equal(np.sort(colors[:n]), np.sort(colors[n:])):
            return True
        if nc == new_count:
            return False
        nc = new_count


def certify(pair):
    n = len(pair[0])
    mats = [csr_matrix(a.astype(np.int64)) for a in pair]
    powers = [np.eye(n, dtype=np.int64) for a in pair]
    labels = np.zeros(2*n*n, dtype=np.int32)
    for k in range(2*n):
        if k:
            powers = [(a @ b) % prime for a,b in zip(mats,powers)]
        labels, _ = bench.ranks(np.column_stack((
            labels, np.concatenate([h.ravel() for h in powers]).astype(np.int32))))
        if k in (1,2,4,8,16,32,64,128,256,2*n-1) and separated(labels.reshape(2,n,n)):
            return k
    return None


data_file, bench_file, out_file = map(Path, sys.argv[1:])
assert not out_file.exists()
results = json.loads(bench_file.read_text())
graphs = bench.read_data(data_file)
targets = sorted(set(results["positives"]["epwl_A"]) | set(results["positives"]["epwl_joint"]))
certificates = {int(i):certify(graphs[i]) for i in targets}
out = {"prime": prime, "evaluated_floating_positives": len(targets),
       "certified_positive": sum(v is not None for v in certificates.values()), "degrees_by_id": certificates}
out_file.write_text(json.dumps(out,indent=2)+"\n")
print("modular positive witnesses", out["certified_positive"], "/", len(targets), flush=True)
