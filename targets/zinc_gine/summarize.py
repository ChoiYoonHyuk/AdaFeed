"""Summarize fixed, paired ZINC runs using only aggregate metric files."""
import argparse
import json
import math
import statistics as stats
from pathlib import Path


def summarize(directory):
    args = json.loads((directory / "protocol.json").read_text())["arguments"]
    seeds = args["seeds"]
    expected_modes = args["modes"]
    if len(seeds) != 4:
        raise ValueError("The paired Student interval below uses three degrees of freedom.")
    records = {mode: [json.loads((directory / f"{mode}_{s}.json").read_text()) for s in seeds]
               for mode in expected_modes}
    summary = {}
    for mode, rows in records.items():
        assert all(r["seed"] == seed and r["mode"] == mode for r, seed in zip(rows, seeds))
        test = [r["test_mae"] for r in rows]
        summary[mode] = dict(test_mean=stats.mean(test), test_sd=stats.stdev(test),
                             validation_mean=stats.mean(r["val_mae"] for r in rows),
                             test_by_seed=test, best_epochs=[r["best_val_epoch"] for r in rows])
    contrasts = {}
    for mode in expected_modes:
        if mode == "adaptive":
            continue
        deltas = [a["test_mae"] - b["test_mae"] for a, b in zip(records["adaptive"], records[mode])]
        mean = stats.mean(deltas)
        half_width = 3.182446305 * stats.stdev(deltas) / math.sqrt(len(deltas))
        contrasts[f"adaptive_minus_{mode}"] = dict(mean=mean, ci95_t=[mean-half_width, mean+half_width],
                                                    adaptive_wins=sum(x < 0 for x in deltas), by_seed=deltas)
    return dict(variant=args["variant"], epochs=args["epochs"], seeds=seeds, methods=summary,
                differences_in_test_mae=contrasts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs="+")
    arguments = parser.parse_args()
    print(json.dumps([summarize(d) for d in arguments.directories], indent=2))
