"""Finite two-query diagnostics on the entire public BREC release (server-side)."""
import argparse
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
import scipy
from scipy.sparse import csr_matrix


DEGREES = (6, 12, 24)
FIELD_SEEDS = (1729, 2718, 3141)
PRIME = 1000003
CATEGORIES = (("Basic", 60), ("Regular", 160), ("Extension", 260),
              ("CFI", 360), ("Four-vertex", 380), ("Distance-regular", 400))


def array_from_g6(blob):
    graph = nx.from_graph6_bytes(bytes(blob).strip())
    return nx.to_numpy_array(graph, dtype=np.int64)


def diagonal_bank(adjacency, potential, degree, modular):
    size = adjacency.shape[0]
    previous = np.eye(size, dtype=adjacency.dtype)
    current = adjacency.toarray()
    current[np.diag_indices(size)] += potential
    rows = [np.diag(current).copy()]
    for _ in range(2, degree + 1):
        product = adjacency @ current + potential[:, None] * current
        following = product % PRIME if modular else 2 * product - previous
        previous, current = current, following
        rows.append(np.diag(current).copy())
    return np.stack(rows, axis=1)


def sorted_rows(table):
    return table[np.lexsort(table.T[::-1])]


def exact_signatures(raw):
    adjacency = csr_matrix(raw)
    zero = np.zeros(len(raw), dtype=np.int64)
    first = diagonal_bank(adjacency, zero, 2 * max(DEGREES), True)
    profiles = {}
    for degree in (*DEGREES, 2 * max(DEGREES)):
        initial = first[:, :degree]
        profiles[(degree, 1, 0)] = sorted_rows(initial)
        if degree not in DEGREES:
            continue
        for seed in FIELD_SEEDS:
            weights = np.random.default_rng(seed).integers(1, PRIME, size=degree)
            field = (initial @ weights) % PRIME
            second = diagonal_bank(adjacency, field, degree, True)
            profiles[(degree, 2, seed)] = sorted_rows(np.column_stack((initial, second)))
    return profiles


def projected_profile(bank):
    # Each sorted column is an invariant. The last two are joint node projections.
    columns = bank.shape[1]
    weights = np.random.default_rng(6101).normal(size=(columns, 2)).astype(bank.dtype)
    weights /= np.abs(weights).sum(axis=0)
    return np.sort(np.column_stack((bank, bank @ weights)), axis=0)


def float_signatures(raw, dtype):
    n = len(raw)
    adjacency = csr_matrix(raw.astype(dtype) / max(1, int(raw.sum(axis=1).max())))
    zeros = np.zeros(n, dtype=dtype)
    first = diagonal_bank(adjacency, zeros, 2 * max(DEGREES), False)
    profiles = {}
    for degree in (*DEGREES, 2 * max(DEGREES)):
        initial = first[:, :degree]
        profiles[(degree, 1, 0)] = projected_profile(initial)
        if degree not in DEGREES:
            continue
        for seed in FIELD_SEEDS:
            weight = np.random.default_rng(seed).normal(size=degree).astype(dtype)
            weight /= np.linalg.norm(weight)
            field = np.tanh(2 * (initial @ weight))
            second = diagonal_bank(adjacency * dtype(0.5), field * dtype(0.5), degree, False)
            profiles[(degree, 2, seed)] = projected_profile(np.column_stack((initial, second)))
    return profiles


def distance(left, right):
    if left.shape != right.shape:
        return 2.0
    return float(np.max(np.abs(left - right)))


def evaluate_pair(pair_item):
    pair_id, graph_g, graph_h = pair_item
    graphs = [array_from_g6(graph_g), array_from_g6(graph_h)]
    generator = np.random.default_rng(8000 + pair_id)
    variants = []
    for graph in graphs:
        perm = generator.permutation(len(graph))
        variants.append(graph[np.ix_(perm, perm)])
    left, right = map(exact_signatures, graphs)
    row = {"id":pair_id, "category":next(c for c, end in CATEGORIES if pair_id < end),
           "observations":[]}
    for key in left:
        degree, rounds, seed = key
        row["observations"].append(dict(degree=degree, rounds=rounds, seed=seed,
                                        arithmetic="modular", separated=not np.array_equal(left[key],right[key])))
    for dtype, name, floor in [(np.float32, "FP32", 2e-5), (np.float64, "FP64", 2e-11)]:
        a, b, a_perm, b_perm = [float_signatures(g, dtype) for g in graphs + variants]
        for key in a:
            degree, rounds, seed = key
            drift = max(distance(a[key], a_perm[key]), distance(b[key], b_perm[key]))
            gap = min(distance(a[key], b[key]), distance(a_perm[key], b_perm[key]))
            row["observations"].append(dict(degree=degree, rounds=rounds, seed=seed, arithmetic=name,
                                            separated=gap > floor + 10 * drift, drift=drift, gap=gap))
    return row


def aggregate(rows):
    summary = []
    for pos, reference in enumerate(rows[0]["observations"]):
        obs = [r["observations"][pos] for r in rows]
        counts = {name:sum(o["separated"] for r,o in zip(rows,obs) if r["category"] == name)
                  for name,_ in CATEGORIES}
        record = {k:reference[k] for k in ("arithmetic","degree","rounds","seed")}
        record.update(categories=counts, total=sum(counts.values()))
        if "drift" in reference:
            record.update(maximum_null_drift=max(o["drift"] for o in obs),
                          minimum_detected_gap=min((o["gap"] for o in obs if o["separated"]),default=None))
        summary.append(record)
    # Distinguish reproducible gains from pair exchanges behind equal totals.
    for idx, entry in enumerate(summary):
        same = [i for i,e in enumerate(summary) if e["arithmetic"] == entry["arithmetic"]
                and e["degree"] == entry["degree"] and e["rounds"] == 1]
        entry["added_over_static"] = sum(r["observations"][idx]["separated"] and
                                         not r["observations"][same[0]]["separated"] for r in rows)
        entry["lost_from_static"] = sum(not r["observations"][idx]["separated"] and
                                        r["observations"][same[0]]["separated"] for r in rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path, help="server path to brec_v3.npy")
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    started = time.monotonic()
    data = np.load(args.release, mmap_mode="r", allow_pickle=False)
    assert data.shape == (51200,), data.shape
    tasks = [(p,data[64*p],data[64*p+1]) for p in range(400)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(evaluate_pair, tasks, chunksize=2):
            rows.append(row)
            if len(rows) % 40 == 0:
                print(f"{len(rows)}/400, {time.monotonic()-started:.1f} s", flush=True)
    # Pair-level diagnostics stay on the compute server.
    Path("pair_diagnostics.json").write_text(json.dumps(rows))
    with args.release.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    report = dict(dataset_sha256=digest.hexdigest(),
                  pair_count=len(rows),worker_count=args.workers,field_seeds=FIELD_SEEDS,
                  seconds=time.monotonic()-started,python=platform.python_version(),
                  scipy=scipy.__version__,numpy=np.__version__,networkx=nx.__version__,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  summary=aggregate(rows))
    temporary = Path("aggregate.json.tmp")
    temporary.write_text(json.dumps(report, indent=2))
    os.replace(temporary,"aggregate.json")


if __name__ == "__main__":
    main()
