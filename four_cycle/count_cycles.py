"""Evaluate the official I2-GNN rooted four-cycle target on an NVIDIA GPU."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

import numpy as np
import scipy.io
import torch


COMMIT = "f514487625b42406dcd85ca55ac66436606f9aac"
URL = f"https://raw.githubusercontent.com/GraphPKU/I2GNN/{COMMIT}/data/count_cycle/raw/data.mat"
DATA_SHA = "758ae12bd05ced3c1ba609d81f0d0dbdc1d227d486e3f2c1c517b47fe7dad9b4"
PARTS = {"train": "train_idx", "validation": "val_idx", "test": "test_idx"}
ARMS = ("root_local", "spatial")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def prepare_file(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as response:
            data = response.read()
        if sha256(data).hexdigest() != DATA_SHA:
            raise ValueError("Data checksum mismatch")
        path.write_bytes(data)
    if digest(path) != DATA_SHA:
        raise ValueError("Data checksum mismatch")


def measure(adjacencies, device):
    """Use float64 products. All sums/products fit exactly into the 53-bit mantissa."""
    by_order = defaultdict(list)
    for index, item in enumerate(adjacencies):
        a = np.asarray(item, dtype=np.float64)
        assert a.ndim == 2 and a.shape[0] == a.shape[1]
        assert np.array_equal(a, a.T) and not a.diagonal().any()
        assert np.all((a == 0) | (a == 1))
        by_order[a.shape[0]].append((index, a))
    records = {}
    for n, group in by_order.items():
        # Conservative bound for matrix powers, degree weights and graph sums.
        assert 16 * n**6 < 2**53
        for start in range(0, len(group), 128):
            batch = group[start:start + 128]
            a = torch.tensor(np.stack([m for _, m in batch]), device=device, dtype=torch.float64)
            d = a.sum(-1)
            two = a @ a
            three = two @ a
            tri = three.diagonal(dim1=-2, dim2=-1)
            four = (two @ two).diagonal(dim1=-2, dim2=-1)
            shifted = a + torch.diag_embed(d)
            local = tri + 3*d.square() + d**3
            spatial = (shifted @ shifted @ shifted).diagonal(dim1=-2, dim2=-1)
            neighbor = (a @ d.unsqueeze(-1)).squeeze(-1)
            delta = spatial - local
            assert torch.equal(delta, neighbor - d.square())
            energy = torch.triu(a*(d.unsqueeze(-1) - d.unsqueeze(-2)).square(), diagonal=1).sum((-2,-1))
            assert torch.equal(delta.sum(-1), torch.zeros_like(energy))
            assert torch.equal((d*delta).sum(-1), -energy)
            common = torch.stack([torch.ones_like(d), d, d.square(), d**3, tri, four], dim=-1)
            all_values = torch.cat([common, local.unsqueeze(-1), spatial.unsqueeze(-1), neighbor.unsqueeze(-1)], -1)
            assert torch.equal(all_values, all_values.round())
            rows = all_values.to(torch.int64).cpu().numpy()
            for (idx, _), row, value in zip(batch, rows, energy.tolist()):
                records[idx] = (row, int(value))
    return records


def features(x, arm):
    index = 6 if arm == "root_local" else 7
    return np.column_stack([x[:, :6], x[:, index]])


def finite_bound(x, y):
    groups = defaultdict(list)
    for row, label in zip(x, y):
        groups[tuple(row)].append(int(label))
    errors = [np.abs(np.asarray(values)-np.median(values)).sum()
              for values in groups.values()]
    mixed = [v for v in groups.values() if min(v) != max(v)]
    total_l1 = int(sum(errors))
    assert total_l1 == sum(errors)
    out = {"groups": len(groups), "ambiguous_groups": len(mixed),
           "roots_in_ambiguous_groups": sum(map(len, mixed)),
           "min_sum_absolute_error": total_l1,
           "within_tuple_min_mae": total_l1 / len(y)}
    if mixed:
        key, values = min(((k, v) for k, v in groups.items() if min(v) != max(v)),
                          key=lambda kv: (-len(kv[1]), kv[0]))
        target, freq = np.unique(values, return_counts=True)
        out["largest_ambiguous_group"] = {
            "features": list(map(int, key)),
            "label_counts": {str(k): int(v) for k, v in zip(target, freq)}}
    return out


def evaluate(records, labels, split_ids):
    results = {}
    aggregate = {}
    trains = np.concatenate([records[i][0] for i in split_ids["train"]])
    train_y = np.concatenate([labels[i] for i in split_ids["train"]])
    val_y = np.concatenate([labels[i] for i in split_ids["validation"]])
    ref_sd = np.concatenate([train_y, val_y]).std(ddof=1)
    fits = {}
    for arm in ARMS:
        beta, _, rank, _ = np.linalg.lstsq(features(trains, arm), train_y, rcond=1.e-12)
        fits[arm] = beta
        results[arm] = {"train_rank": int(rank), "coefficients": beta.tolist()}
    for split, ids in split_ids.items():
        blocks = [records[i][0] for i in ids]
        counts = np.array([len(x) for x in blocks])
        offsets = np.r_[0, counts.cumsum()[:-1]]
        x = np.concatenate(blocks)
        y = np.concatenate([labels[i] for i in ids])
        d, d2, four, local, spatial, neighbor = x[:,1], x[:,2], x[:,5], x[:,6], x[:,7], x[:,8]
        baseline2 = four - 2*d2 + d
        exact2 = baseline2 - (spatial-local)
        explicit2 = baseline2 - (neighbor-d2)
        errors = {}
        part = {
            "graphs": int(len(ids)), "roots": int(len(y)), "target_sum": int(y.sum()),
            "positive_roots": int(np.count_nonzero(y)), "max_target": int(y.max()),
            "order_min": int(counts.min()), "order_max": int(counts.max()),
            "formula_exact_roots": int(np.count_nonzero(exact2 == 2*y)),
            "formula_mae": float(np.abs(exact2/2-y).mean()),
            "neighbor_control_mae": float(np.abs(explicit2/2-y).mean()),
            "omit_delta_node_mae": float(np.abs(baseline2/2-y).mean()),
            "omit_delta_graph_total_mae": float(np.abs(np.add.reduceat(baseline2-2*y, offsets)/8).mean()),
            "nonzero_contrast_roots": int(np.count_nonzero(spatial-local)),
            "nonzero_edge_energy_graphs": int(sum(records[i][1]>0 for i in ids)),
            "uniform_contrast_absmax": int(np.max(np.abs(np.add.reduceat(spatial-local, offsets)))),
            "weighted_identity_absmax": int(np.max(np.abs(np.add.reduceat(d*(spatial-local), offsets)+[records[i][1] for i in ids])))}
        for arm in ARMS:
            bank = features(x, arm)
            error = np.abs(bank @ fits[arm]-y)
            errors[arm] = np.add.reduceat(error, offsets)
            part[arm] = {"ols_mae": float(error.mean()), "normalized_mae": float(error.mean()/ref_sd)}
            if split == "test":
                part[arm].update(finite_bound(bank, y))
        if split == "test":
            per_graph = errors["root_local"]-errors["spatial"]
            rng = np.random.default_rng(2901)
            estimates = []
            for _ in range(2000):
                sample = rng.integers(len(ids), size=len(ids))
                estimates.append(per_graph[sample].sum()/counts[sample].sum())
            part["paired_node_mae_improvement"] = float(per_graph.sum()/counts.sum())
            part["paired_graph_interval"] = np.quantile(estimates, [.025,.975]).tolist()
        aggregate[split] = part
    return {"training_plus_validation_target_sd": float(ref_sd), "fits": results, "splits": aggregate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare_file(args.data)
    raw = scipy.io.loadmat(args.data)
    ids = {name: raw[key].reshape(-1) for name, key in PARTS.items()}
    aa, ff = raw["A"].reshape(-1), raw["F"].reshape(-1)
    assert sorted(np.concatenate(list(ids.values())).tolist()) == list(range(len(aa))) and len(ff)==len(aa)
    labels = []
    for a, targets in zip(aa, ff):
        y = targets[:, 1]
        assert len(y) == len(a) and np.array_equal(y, y.astype("int64"))
        labels.append(y.astype("int64"))
    for idx in ids["train"][:3]:
        a = aa[idx].astype("int64")
        common_neighbors = a @ a
        np.fill_diagonal(common_neighbors,0)
        cycle_counts = (common_neighbors*(common_neighbors-1)//2).sum(1)
        assert np.array_equal(cycle_counts, labels[idx])
    assert torch.cuda.is_available()
    records = measure(aa, "cuda")
    out = evaluate(records, labels, ids)
    out.update(commit=COMMIT, dataset_sha256=DATA_SHA, target_index=1,
               protocol_sha256=digest(Path(__file__).with_name("protocol.md")),
               program_sha256=digest(Path(__file__)), gpu=torch.cuda.get_device_name(),
               torch_version=torch.__version__, numpy_version=np.__version__)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
