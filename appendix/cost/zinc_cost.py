"""Timing on the official ZINC-12k test molecules and independent relabelings."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import pickle
from zipfile import ZipFile

import brec_stable_cost as bench
import numpy as np
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("test_index", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    index = [int(s) for s in args.test_index.read_text().strip().strip(",").split(",")]
    assert len(index) == len(set(index)) == 1000
    with ZipFile(args.archive) as archive, archive.open("molecules/test.pickle") as src:
        molecules = pickle.load(src)
    assert len(molecules) == 5000
    if hasattr(os,"sched_getaffinity"):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    graphs = []
    perms = np.random.default_rng(591342)
    for idx in index:
        a = (np.asarray(molecules[idx]["bond_type"]) != 0).astype(np.float64)
        assert np.all(a.T == a) and not a.diagonal().any()
        p = perms.permutation(len(a))
        graphs.append([a, a[np.ix_(p,p)]])
    del molecules
    n = [len(g[0]) for g in graphs]
    methods = list(bench.METHODS)
    for idx in range(6):
        for m in methods:
            bench.METHODS[m](graphs[idx])
    rng = np.random.default_rng(20271105)
    repeats = []
    gc.disable()
    for rep in range(3):
        batch = [{} for _ in graphs]
        for i in rng.permutation(len(graphs)):
            for m in rng.permutation(methods):
                batch[i][m] = bench.METHODS[m](graphs[i])
        print("pass", rep+1, {m:round(sum(x[m]["full"] for x in batch),3) for m in methods},flush=True)
        repeats.append(batch)
    gc.enable()
    with args.archive.open("rb") as f:
        archive_sha = hashlib.file_digest(f, "sha256").hexdigest()
    result = dict(
        dataset="ZINC-12k, official test split, untyped bonds",
        pairs="each graph and one randomly permuted copy, full feature extraction",
        vertices_quantiles=np.quantile(n, [0,.25,.5,.75,1]).tolist(),
        dataset_archive_sha256=archive_sha,
        test_index_sha256=hashlib.sha256(args.test_index.read_bytes()).hexdigest(),
        timing_code_sha256=hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        relabel_seed=591342, ordering_seed=20271105,
        false_positive_ids={m:[int(i) for i in range(len(graphs)) if any(x[i][m]["decision"] for x in repeats)] for m in methods},
        summary=bench.summaries(repeats, {"all":list(range(len(graphs)))})
    )
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("complete", result["vertices_quantiles"], {m:len(ids) for m,ids in result["false_positive_ids"].items()}, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
