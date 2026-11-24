"""Fit a finite field bank and graph readout separately for each BREC pair."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from evaluate import CATEGORIES, array_from_g6, diagonal_bank


K = 24
SEEDS = tuple(range(9190, 9200))
AMPLITUDES = (0.25, 0.5, 1.0)
RIDGE = 1e-7
THRESHOLD = 72.34


def preparation(blob):
    matrix = array_from_g6(blob)
    scale = max(1, matrix.sum(axis=1).max())
    adjacency = csr_matrix(matrix.astype(np.float64) / scale)
    bank = diagonal_bank(adjacency, np.zeros(len(matrix)), K, False)
    return adjacency * 0.5, bank, np.sort(bank, axis=0).ravel()


def response(graph, weights, amplitude):
    adjacency, bank, static = graph
    potential = 0.5 * amplitude * np.tanh(2 * (bank @ weights))
    second = diagonal_bank(adjacency, potential, K, False)
    return np.concatenate((static, np.sort(second, axis=0).ravel()))


def field_selection(left, right, seed):
    rng = np.random.default_rng(seed)
    best = None
    # Rank candidate fields only on the first released training pair.
    for amplitude in AMPLITUDES:
        vector = rng.normal(size=K)
        vector /= np.linalg.norm(vector)
        first = response(left, vector, amplitude)[len(left[2]):]
        second = response(right, vector, amplitude)[len(right[2]):]
        score = np.linalg.norm(first - second)
        if best is None or score > best[0]:
            best = (score, vector, amplitude)
    return best[1:]


def t_squared(embeddings):
    differences = embeddings[0::2] - embeddings[1::2]
    mean = differences.mean(axis=0)
    cov = np.cov(differences, rowvar=False) + RIDGE * np.eye(embeddings.shape[1])
    return float(mean @ np.linalg.solve(cov, mean))


def evaluate_embeddings(features):
    examples = features[:64]
    left = examples[0::2].mean(axis=0)
    right = examples[1::2].mean(axis=0)
    # Supervised feature selection and a midpoint fit, shared at evaluation.
    select = np.argsort(-np.abs(left-right), kind="stable")[:16]
    midpoint = (left[select]+right[select])/2
    embeddings = np.tanh((features[:, select] - midpoint) / 0.01)
    statistic = t_squared(embeddings[:64])
    null = t_squared(embeddings[64:])
    passed = statistic > THRESHOLD and null < THRESHOLD and not np.isclose(statistic, null, atol=1e-6)
    return dict(separated=bool(passed), null_failed=bool(null >= THRESHOLD))


def evaluate_pair(args):
    index, blobs = args
    # Reusing byte-identical inputs preserves their multiplicity in the RPC test.
    encoding = {raw:preparation(raw) for raw in dict.fromkeys(blobs)}
    graphs = [encoding[raw] for raw in blobs]
    shapes = {graph[2].shape for graph in graphs}
    assert len(shapes) == 1, (index, shapes)
    static = evaluate_embeddings(np.stack([g[2] for g in graphs]))
    fits = []
    for seed in SEEDS:
        vector, amplitude = field_selection(graphs[0], graphs[1], seed)
        encoded = {raw:response(g,vector,amplitude) for raw,g in encoding.items()}
        result = evaluate_embeddings(np.stack([encoded[raw] for raw in blobs]))
        fits.append(dict(seed=seed, amplitude=amplitude, **result))
    return dict(id=index, category=next(c for c,e in CATEGORIES if index < e), static=static, fits=fits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    data = np.load(args.release,allow_pickle=False,mmap_mode="r")
    assert data.shape == (51200,)
    tasks = ((p, [bytes(x) for x in data[p*64:p*64+64]] +
                 [bytes(x) for x in data[(p+400)*64:(p+401)*64]]) for p in range(400))
    start = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(evaluate_pair,tasks,chunksize=1):
            rows.append(row)
            if len(rows) % 40 == 0:
                print(f"{len(rows)}/400 in {time.monotonic()-start:.1f} s",flush=True)
    Path("paired_decisions.json").write_text(json.dumps(rows))
    cert_rows = json.loads(Path("pair_diagnostics.json").read_text())
    witnesses = {r["id"] for r in cert_rows if any(o["arithmetic"] == "modular" and
                 o["degree"] == K and o["rounds"] == 2 and o["seed"] == 1729 and o["separated"]
                 for o in r["observations"])}
    assert len(rows) == len(cert_rows) == 400 and len(witnesses) == 219
    diagnostics = {}
    for precision in ("FP32", "FP64"):
        diagnostics[precision] = []
        for seed in (1729,2718,3141):
            run = [next(o for o in r["observations"] if (o["arithmetic"],o["degree"],o["rounds"],o["seed"]) ==
                        (precision,K,2,seed)) for r in cert_rows]
            detected = {r["id"] for r,o in zip(cert_rows,run) if o["separated"]}
            accepted = [o["gap"]/max(o["drift"],np.finfo(float).tiny)
                        for o in run if o["separated"] and o["drift"] > 0]
            diagnostics[precision].append(dict(seed=seed,missing=len(witnesses-detected),
                          extra=len(detected-witnesses), min_gap_over_nonzero_drift=min(accepted)))
    summary = []
    for method,run_id in [("static",None)] + [("feedback",i) for i in range(len(SEEDS))]:
        outcomes = [r["static"] if run_id is None else r["fits"][run_id] for r in rows]
        positives = {r["id"] for r,o in zip(rows,outcomes) if o["separated"]}
        counts = {c:sum(o["separated"] for r,o in zip(rows,outcomes) if r["category"]==c)
                  for c,_ in CATEGORIES}
        summary.append(dict(method=method, seed=None if run_id is None else SEEDS[run_id],
                    counts=counts,total=len(positives),null_failures=sum(o["null_failed"] for o in outcomes),
                    missing_certified=len(witnesses-positives), outside_certificates=len(positives-witnesses)))
    result = dict(pairs=400,paired_relabelings=32,reliability_pairs=32,threshold=THRESHOLD,ridge=RIDGE,
                  degree=K,workers=args.workers,seconds=time.monotonic()-start,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  numerical_checks=diagnostics,summary=summary)
    Path("paired_aggregate.json").write_text(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
