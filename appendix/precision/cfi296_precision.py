"""Numerical audit of the prespecified BREC-296 two-query witness on one CUDA GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path

# Deterministic cuBLAS path. Set before CUDA initialization.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import networkx as nx
import numpy as np
import torch


INPUT_SHA = "6975d172d27aedcdf4eb2c747450ede7928c111caeb2689fbc67e6b62fbd5562"
PAIR = 296
SEED = 20261122
RELABELINGS = 16


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def exact_graph(x):
    a = x.astype(object)
    a2 = np.linalg.matrix_power(a, 2).diagonal()
    a9 = np.linalg.matrix_power(a, 9).diagonal()
    mask = (a2 == 4) & (a9 == 4)
    if len(a) != 62 or mask.sum() != 8:
        raise ValueError("The integer selector differs from the reported benchmark pair")
    spatial = 2 * a + np.diag(mask.astype(np.int64))
    local = 2 * a + np.eye(len(a), dtype=object)
    scaled = {
        "spatial": np.linalg.matrix_power(spatial, 10).diagonal().copy(),
        "local": np.linalg.matrix_power(local, 10).diagonal().copy(),
    }
    return dict(a2=a2, a9=a9, mask=mask, scaled=scaled,
                degrees=x.sum(axis=1), max_spatial_row_sum=max(spatial.sum(axis=1)))


def query(x, dtype, reference, perm):
    a = torch.as_tensor(x[np.ix_(perm, perm)], device="cuda", dtype=dtype)
    # Sequential matrix multiplication implements the two requested power banks.
    power = a
    for k in range(2, 10):
        power = power @ a
        if k == 2:
            a2 = torch.diagonal(power).clone()
    a9 = torch.diagonal(power)
    mask = (a2 == 4) & (a9 == 4)
    exact_mask = reference["mask"][perm]
    stats = {
        "mask_mismatches": int(np.count_nonzero(mask.cpu().numpy() != exact_mask)),
        "max_a2_error": float(np.max(np.abs(a2.cpu().numpy().astype(np.float64) - reference["a2"][perm].astype(np.float64)))),
        "max_a9_error": float(np.max(np.abs(a9.cpu().numpy().astype(np.float64) - reference["a9"][perm].astype(np.float64)))),
    }
    for name, op in [
        ("spatial", a + torch.diag(mask.to(dtype)*0.5)),
        # Only the selected roots enter the Local score, and all have p_i=1/2.
        ("local", a + torch.eye(len(a), device="cuda", dtype=dtype)*0.5),
    ]:
        power = op
        for _ in range(2, 11):
            power = power @ op
        diagonal = torch.diagonal(power)
        computed = diagonal.cpu().numpy().astype(np.float64)
        exact = reference["scaled"][name][perm][exact_mask].astype(np.float64)/1024
        stats[name] = {
            "score": float(diagonal[mask].sum().item()),
            "max_selected_diagonal_error": float(np.max(np.abs(computed[exact_mask]-exact))),
        }
    return stats


def summarize(rows):
    return {
        "total_mask_mismatches": sum(r["mask_mismatches"] for pair in rows for r in pair),
        "max_a2_error": max(r["max_a2_error"] for pair in rows for r in pair),
        "max_a9_error": max(r["max_a9_error"] for pair in rows for r in pair),
        "readouts": {
            name: {
                "identity_graph_scores": [r[name]["score"] for r in rows[0]],
                "score_ranges_by_graph": [
                    [min(pair[j][name]["score"] for pair in rows), max(pair[j][name]["score"] for pair in rows)]
                    for j in range(2)
                ],
                "gap_by_ordering": [pair[0][name]["score"]-pair[1][name]["score"] for pair in rows],
                "max_selected_diagonal_error": max(r[name]["max_selected_diagonal_error"] for pair in rows for r in pair),
            } for name in ("spatial", "local")
        },
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("input", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    if digest(args.input) != INPUT_SHA:
        raise ValueError("Official BREC input checksum mismatch")
    records = np.load(args.input, mmap_mode="r", allow_pickle=False)
    raw = records[64*PAIR:64*PAIR+2]
    graphs = [nx.to_numpy_array(nx.from_graph6_bytes(bytes(row).strip()),dtype=np.int64) for row in raw]
    refs = [exact_graph(x) for x in graphs]
    assert [set(ref["scaled"]["spatial"][ref["mask"]]) for ref in refs] == [{199319057}, {199317009}]
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.cuda.get_device_name(0)
    if "H100" not in device:
        raise ValueError(f"H100 requested, found {device}")
    rng = np.random.default_rng(SEED)
    n = len(graphs[0])
    permutations = [[np.arange(n) for _ in graphs]]
    permutations += [[rng.permutation(n) for _ in graphs] for _ in range(RELABELINGS)]
    result = dict(
        source_sha256=INPUT_SHA, pair_zero_based=PAIR, n=n, independent_relabelings=RELABELINGS,
        permutation_seed=SEED, device=device, torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda), allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        exact=[{
            "maximum_degree":int(ref["degrees"].max()),
            "maximum_spatial_scaled_row_sum":int(ref["max_spatial_row_sum"]),
            "selected_roots":int(ref["mask"].sum()),
            "denominator":1024,
            "selected_scaled_diagonal_by_mode": {name:sorted(set(map(int,vals[ref["mask"]]))) for name,vals in ref["scaled"].items()},
            "score_by_mode":{name:int(vals[ref["mask"]].sum())/1024 for name,vals in ref["scaled"].items()},
        } for ref in refs],
        results={},
    )
    with torch.no_grad():
        for dtype in (torch.float32, torch.float64):
            rows=[[query(x,dtype,ref,p) for x,ref,p in zip(graphs,refs,pair)] for pair in permutations]
            result["results"][str(dtype)] = summarize(rows)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for k,v in result["results"].items():
        print(k, "mask mismatches", v["total_mask_mismatches"], "d9 abs error",v["max_a9_error"],
              "spatial gap range", min(v["readouts"]["spatial"]["gap_by_ordering"]), max(v["readouts"]["spatial"]["gap_by_ordering"]),
              "local gap range",min(v["readouts"]["local"]["gap_by_ordering"]),max(v["readouts"]["local"]["gap_by_ordering"]))
    print("max degrees", [x["maximum_degree"] for x in result["exact"]])


if __name__ == "__main__":
    main()
