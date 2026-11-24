"""Verify the ZINC readout results using the released fit-level aggregates.

Usage: python verify_readout.py
The whole-molecule resampling interval is produced by adaspec_zinc.py on the
compute host. It requires the paired errors for individual molecules.
"""
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

DIR = Path(__file__).resolve().parent
SEEDS = (103, 211, 307, 401, 503)
ARMS = ("uniform", "weighted_root_local", "weighted_spatial")
SPLITS = {"train": 10000, "val": 1000, "test": 1000}

def verify():
    summary = json.loads((DIR / "summary.json").read_text("utf8"))
    assert summary["dataset"] == "ZINC-12k" and summary["metric"] == "MAE"
    assert tuple(summary["seeds"]) == SEEDS and tuple(summary["arms"]) == ARMS
    assert "h100" in summary["h100"].lower()
    assert summary["commit"] == "d76aa64d3075a73b014345548473d672f103045a"
    for filename, key in (("adaspec_zinc.py", "code_sha256"), ("protocol.md", "protocol_sha256")):
        assert hashlib.sha256((DIR / filename).read_bytes()).hexdigest() == summary[key], filename
    assert set(summary["split_sha256"]) == set(SPLITS)
    assert set(summary["source_sha256"]) == {"gpr_model.py", "training.py"}
    assert all(re.fullmatch("[0-9a-f]{64}", value) for value in list(summary["split_sha256"].values()) + list(summary["source_sha256"].values()))
    stats = summary["pooled_coordinate_checks"]
    assert {s: stats[s]["n"] for s in stats} == SPLITS
    for split, count in SPLITS.items():
        entry = stats[split]
        assert max(entry["uniform_max_abs_error"], entry["weighted_energy_max_abs_error"]) < 1e-10
        assert 0 <= entry["positive_energy_graphs"] <= count and entry["mean_edge_energy"] >= 0
    assert stats["train"]["published_matrix_check_n"] == 32
    assert stats["train"]["published_matrix_max_abs_error_fp32"] < 1e-7
    records = {(row["seed"], row["arm"]): row for row in summary["records"]}
    assert len(summary["records"]) == 15 and set(records) == {(s, a) for s in SEEDS for a in ARMS}
    assert len({row["parameters"] for row in records.values()}) == 1
    for row in records.values():
        assert 1 <= row["best_epoch"] <= row["epochs"] <= 160
        assert row["best_epoch"] + 25 == row["epochs"] or row["epochs"] == 160
        for k in ("validation_mae", "test_mae", "test_max_grouped_forward_abs_diff", "train_max_grouped_forward_abs_diff"):
            assert math.isfinite(row[k]) and row[k] >= 0
    losses = {arm: [records[seed, arm]["test_mae"] for seed in SEEDS] for arm in ARMS}
    for arm in ARMS:
        assert math.isclose(statistics.mean(losses[arm]), summary["mean_mae"][arm], abs_tol=1e-12)
    for control in ARMS[:2]:
        delta = [a - b for a, b in zip(losses[control], losses["weighted_spatial"])]
        mean = statistics.mean(delta)
        half = 2.776445105197799 * statistics.stdev(delta) / math.sqrt(len(delta))
        obs = summary["paired_control_minus_spatial"][control]
        for reported, rebuilt in zip(obs["per_seed"] + [obs["mean"]] + obs["seed_ci"], delta + [mean, mean-half, mean+half], strict=True):
            assert math.isclose(reported, rebuilt, abs_tol=1e-12), (control, reported, rebuilt)
        lo, hi = obs["molecule_ci"]
        assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi
    print("Checked source and protocol hashes, split sizes, 15 fits, graph coordinates, mean MAEs and paired seed intervals.")

if __name__ == "__main__":
    verify()
