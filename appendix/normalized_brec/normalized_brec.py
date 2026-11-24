"""Rational separation certificates for a fixed three-query normalized BREC encoder."""
import argparse
import hashlib
import json
from math import isqrt
import platform
from pathlib import Path
import time

import networkx as nx
import numpy as np
import scipy
from scipy.sparse import csr_matrix

PRIMES = (1_000_003, 1_000_033)
COEFFICIENTS = np.array([2,-1,3,1,-2,-3,1,-1,2,3,-1,2,-3,1,2,-2,3,-1,1,2,-3,2,-1,3], dtype=np.int64)
GATE_DENOM = int(np.abs(COEFFICIENTS).sum())
SOURCE_SHA = "6975d172d27aedcdf4eb2c747450ede7928c111caeb2689fbc67e6b62fbd5562"
GROUPS = [("basic",0,60),("regular",60,160),("extension",160,260),
          ("cfi",260,360),("four_vertex",360,380),("distance_regular",380,400)]


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def history_generator(adjacency, prime):
    deg = adjacency.sum(axis=1)
    n = len(deg)
    if max(n, GATE_DENOM) >= prime or 2 * prime**2 >= np.iinfo(np.int64).max:
        raise ArithmeticError("Degree denominator or signed int64 bound")
    inv = np.array([pow(int(d), -1, prime) if d else 0 for d in deg], dtype=np.int64)
    left = csr_matrix(adjacency)
    history = []
    last = None
    for depth in range(1, 4):
        potential = None
        if depth > 1:
            potential = ((last @ COEFFICIENTS) % prime) * pow(GATE_DENOM, -1, prime) % prime
        power = np.eye(n, dtype=np.int64)
        diag = np.empty((n, len(COEFFICIENTS)), dtype=np.int64)
        for k in range(len(COEFFICIENTS)):
            # The unweighted sparse product sums at most n terms below the prime.
            step = ((left @ power) % prime) * inv[:, None] % prime
            if depth > 1:
                step = (step + potential[:, None] * power) % prime
                step = step * pow(2, -1, prime) % prime
            power = step
            diag[:, k] = np.diag(power)
        history.append(diag)
        last = diag
        yield np.column_stack(history)


def different_multisets(a, b):
    if a.shape != b.shape:
        return True
    _, idx = np.unique(np.vstack((a, b)), axis=0, return_inverse=True)
    return not np.array_equal(np.sort(idx[:len(a)]), np.sort(idx[len(a):]))


def first_separation(graphs, prime):
    for depth, (a,b) in enumerate(zip(*(history_generator(g,prime) for g in graphs)), 1):
        if different_multisets(a, b):
            return depth
    return 0


def relabel_check(graphs, prime):
    rng = np.random.default_rng(20261122)
    histories = []
    for a in graphs:
        perm = rng.permutation(len(a))
        canonical = list(history_generator(a, prime))
        permuted = list(history_generator(a[np.ix_(perm,perm)], prime))
        for orig, moved in zip(canonical, permuted):
            if not np.array_equal(orig[perm], moved):
                raise AssertionError("Relabeling did not preserve vertex histories")
        histories.append(canonical)
    return [different_multisets(a,b) for a,b in zip(*histories)]


def counts(by_prime, subset):
    by_stage = {}
    for key, depths in by_prime.items():
        by_stage[str(key)] = [sum(0 < depths[i] <= r for i in subset) for r in (1, 2, 3)]
    return by_stage


def main(data, output):
    folder = Path(__file__).resolve().parent
    for p in PRIMES:
        assert all(p % k for k in range(2,isqrt(p)+1))
    digest = file_sha(data)
    if digest != SOURCE_SHA:
        raise ValueError("Unexpected BREC release digest")
    raw = np.load(data, mmap_mode="r", allow_pickle=False)
    assert raw.shape == (51_200,)
    start = time.perf_counter()
    by_prime = {p: [] for p in PRIMES}
    relabeled_296 = {}
    for i in range(400):
        graphs = [nx.to_numpy_array(nx.from_graph6_bytes(bytes(g).strip()),dtype=np.int64)
                  for g in raw[64*i:64*i+2]]
        for p in PRIMES:
            by_prime[p].append(first_separation(graphs, p))
            if i == 296:
                relabeled_296[str(p)] = relabel_check(graphs, p)
        if (i+1) % 50 == 0:
            print("Processed", i+1, "of 400 pairs", flush=True)
    union = [min((r for r in vals if r > 0), default=0)
             for vals in zip(*(by_prime[p] for p in PRIMES))]
    by_prime["either"] = union
    elapsed = time.perf_counter() - start
    # Read the per-pair oracle only after all normalized decisions are determined.
    oracle_file = folder / "certified_ids.json"
    oracle = set(json.loads(oracle_file.read_text())["feedback_positive_ids"])
    assert len(oracle) == 219
    seen = {i for i,d in enumerate(union) if d}
    if seen - oracle:
        raise AssertionError("A certified tie was separated; audit needed")
    result = {
        "device": "H100 server host CPU, single process, exact integer residues",
        "cpu_architecture": platform.machine(),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "networkx": nx.__version__,
        "source_sha256": digest, "protocol_sha256": file_sha(folder/"protocol.md"),
        "script_sha256": file_sha(Path(__file__)), "oracle_sha256": file_sha(oracle_file),
        "pairs": 400, "primes": PRIMES, "coefficients": COEFFICIENTS.tolist(),
        "gate_denominator": GATE_DENOM, "degrees": [1,24], "queries": 3,
        "cpu_wall_seconds_without_input_hash": elapsed,
        "cumulative_counts": counts(by_prime,range(400)),
        "family_cumulative_counts": {g:counts(by_prime,range(lo,hi)) for g,lo,hi in GROUPS},
        "certified_positive_count":len(oracle),"detected_certified_positive":len(seen & oracle),
        "undetected_certified_positive_ids":sorted(oracle-seen),
        "detected_certified_tie_ids":sorted(seen-oracle),
        "primes_disagree_first_detection_ids":[i for i in range(400)
                  if by_prime[PRIMES[0]][i] != by_prime[PRIMES[1]][i]],
        "pair_296_relabel_vertex_histories_agree":True,
        "pair_296_unequal_multisets_at_each_depth":relabeled_296,
        "separation_ids_at_exact_first_depth":{str(d):[i for i,r in enumerate(union) if r==d]
                                               for d in (1,2,3)},
    }
    output.write_text(json.dumps(result,indent=2)+"\n")
    print("Normalized cumulative separations:", result["cumulative_counts"])
    print("Undetected certified positives:",result["undetected_certified_positive_ids"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    main(args.data, args.output)

