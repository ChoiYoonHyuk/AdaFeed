"""Exact stabilized adjacency, Laplacian and normalized-Laplacian EPWL upper checks.

Usage: python epwl_upper.py brec_v3.npy brec_results.json epwl_upper.json --workers 4
brec_results.json is the per-pair certificate route report from brec_verify.py.
No floating-point eigenvalues, rounded projectors or tolerance are used.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from math import lcm
from pathlib import Path

import networkx as nx
import numpy as np

RELEASE_SHA256 = "6975d172d27aedcdf4eb2c747450ede7928c111caeb2689fbc67e6b62fbd5562"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse(raw):
    graph = nx.from_graph6_bytes(raw.strip())
    return [list(sorted(graph[i])) for i in range(len(graph))]


def intern(keys):
    indices = {}
    return [indices.setdefault(key, len(indices)) for key in keys]


def advance(x, adj, operator, scale):
    out = np.empty_like(x)
    for i, neighbors in enumerate(adj):
        row = x[neighbors].sum(axis=0)
        if operator == "L":
            row = len(neighbors) * x[i] - row
        elif operator == "walk":
            row = (scale // len(neighbors)) * row if neighbors else np.zeros_like(row)
        out[i] = row
    return out


def stationary_histograms(pair, n):
    colors = [[0] * n, [0] * n]
    count = 1
    sizes = [1]
    for _ in range(2 * n):
        signatures = []
        for r in range(2):
            for i in range(n):
                offset = (r*n+i)*n
                incident = tuple(sorted(zip(pair[offset:offset+n], colors[r])))
                signatures.append((colors[r][i], incident))
        labels = intern(signatures)
        colors = [labels[:n], labels[n:]]
        next_count = max(labels)+1
        sizes.append(next_count)
        if Counter(colors[0]) != Counter(colors[1]):
            return False, sizes
        if next_count == count:
            return True, sizes
        count = next_count
    raise AssertionError("vertex partition did not stabilize")


def verify(task):
    pair_id, left, right = task
    adj = parse(left), parse(right)
    n = len(adj[0])
    assert n == len(adj[1])
    degrees = [[len(row) for row in a] for a in adj]
    scale = lcm(*(d for deg in degrees for d in deg if d)) or 1
    labels = intern((deg[i], deg[j]) for deg in degrees for i in range(n) for j in range(n))
    checked = []
    for name in ("A", "L", "walk"):
        powers = [np.eye(n, dtype=object) for _ in adj]
        for k in range(n+1):
            if powers[0].trace() != powers[1].trace():
                return {"id":pair_id, "n":n, "upper_certified":False,
                        "mismatch":"trace", "operator":name, "power":k}
            if k < n:
                entries = (value for mat in powers for value in mat.flat)
                labels = intern(zip(labels, entries))
                powers = [advance(mat, a, name, scale) for mat,a in zip(powers,adj)]
        checked.append(max(labels)+1)
    tied, vertex_counts = stationary_histograms(labels,n)
    return {"id":pair_id,"n":n,"walk_lcm":scale,"max_trace_power":n,
            "max_pair_entry_power":n-1,"pair_classes_after_each_operator":checked,
            "joint_vertex_classes_by_round":vertex_counts,"upper_certified":tied}


def write_summary(path, pairs, records):
    obj = {"release_sha256":RELEASE_SHA256,"target_pair_ids":pairs,
           "algorithm":"exact integer power traces and joint pair-entry sequences, followed by common vertex-color refinement",
           "target_count":len(pairs),"checked_count":len(records),
           "upper_certified_count":sum(r["upper_certified"] for r in records),
           "rows":sorted(records,key=lambda r:r["id"])}
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("release",type=Path)
    ap.add_argument("routes",type=Path)
    ap.add_argument("output",type=Path)
    ap.add_argument("--workers",type=int,default=1)
    args=ap.parse_args()
    assert file_hash(args.release)==RELEASE_SHA256
    rows=json.loads(args.routes.read_text(encoding="utf8"))["decisions"]
    ids=sorted(r["id"] for r in rows if r["epwl_positive_degree"] is None and r["feedback_route"]=="common_colored_words")
    assert len(ids)==34 and len(set(ids))==len(ids)
    release=np.load(args.release,mmap_mode="r",allow_pickle=False)
    assert release.shape==(51200,)
    jobs=[(p,bytes(release[64*p]),bytes(release[64*p+1])) for p in ids]
    done=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(verify,job) for job in jobs]
        for future in as_completed(futures):
            row=future.result()
            done.append(row)
            write_summary(args.output,ids,done)
            if len(done)%8==0 or not row["upper_certified"]:
                print(len(done),row["id"],row["upper_certified"],flush=True)
    print("Upper certificates",sum(r["upper_certified"] for r in done),"of",len(ids))
if __name__=="__main__":
    main()
