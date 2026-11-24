"""Check the original per-family records or compare an offline rerun with them."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
SEEDS = [103, 211, 307, 401, 503]
MODES = ["local_sum", "local_gated", "spatial_sum", "spatial_gated", "frozen_spatial_gated"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_candidate(receipt, reference, summary, cache):
    require(receipt["seeds"] == reference["seeds"] == SEEDS, "Seed list differs")
    require(receipt["epochs"] == 180, "Training duration differs")
    require(receipt["script_sha256"] == sha256((ROOT / "learned_catalogue.py").read_bytes()).hexdigest(),
            "Training script differs from the recorded script")
    families = receipt["families"]
    require(len(families) == len(reference["families"]) == 6, "Missing family")
    for family, ref in zip(families, reference["families"]):
        params = family["parameters"]
        n, d, _, _ = params
        name = "sr" + "".join(map(str, params)) + ".g6"
        for k in ("parameters", "source_sha256", "split_sha256", "cores", "roots",
                  "split_seed", "best_constant_on_test", "test_target_range"):
            require(family[k] == ref[k], f"{name}: {k} differs")
        if cache is not None:
            require(sha256((cache / name).read_bytes()).hexdigest() == family["source_sha256"],
                    f"{name}: input bytes differ")
        count = sum(family["cores"])
        nt, nv = int(.6 * count), max(1, int(.2 * count))
        require(family["cores"] == [nt, nv, count - nt - nv] and
                family["roots"] == [v*n for v in family["cores"]], f"{name}: partition")
        require(family["split_seed"] == 20271107 + 100*n + d, f"{name}: split seed")
        require(set(family["modes"]) == set(MODES), f"{name}: arm list")
        for mode in MODES:
            results = family["modes"][mode]
            require([v["seed"] for v in results] == SEEDS, f"{name}: {mode} seeds")
            for fit, archived in zip(results, ref["modes"][mode]):
                require(fit["n"] == family["roots"][2], f"{name}: {mode} test size")
                require(fit["correct"] == archived["correct"], f"{name}: {mode} rounded counts")
                require(isclose(fit["mae"], archived["mae"], rel_tol=1e-4, abs_tol=1e-7),
                        f"{name}: {mode} MAE differs at seed {fit['seed']}")
        print(f"{name}: {family['cores'][2]} test cores, {family['roots'][2]} roots,"
              f" {family['modes']['spatial_gated'][0]['correct']} correct with weighted spatial"
              f" and {family['modes']['local_sum'][0]['correct']} with uniform local (first seed).")
    for key in ("cores", "roots"):
        require([sum(f[key][i] for f in families) for i in range(3)] == summary[key], key)
    require(sum(f["best_constant_on_test"] for f in families) ==
            summary["oracle_constant_correct"], "Oracle total")
    n_test = summary["roots"][2]
    for mode in MODES:
        rounded, errors = [], []
        for i, seed in enumerate(SEEDS):
            fits = [f["modes"][mode][i] for f in families]
            correct = sum(x["correct"] for x in fits)
            mae = sum(x["mae"]*x["n"] for x in fits) / n_test
            expected = summary["modes"][mode]["per_seed"][i]
            require(expected["seed"] == seed and correct == expected["correct"] and
                    isclose(mae, expected["mae"], rel_tol=1e-4, abs_tol=1e-7),
                    f"{mode}: aggregate at seed {seed}")
            rounded.append(correct / n_test)
            errors.append(mae)
        require(isclose(mean(rounded), summary["modes"][mode]["accuracy_mean"]) and
                isclose(mean(errors), summary["modes"][mode]["mae_mean"], rel_tol=1e-4, abs_tol=1e-7),
                f"{mode}: seed average")
        print(f"{mode}: test MAE {mean(errors):.8g}, rounded accuracy {100*mean(rounded):.4f}%")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, nargs="?", default=ROOT / "learned-catalogue-results.json")
    parser.add_argument("--cache", type=Path, help="Check the six graph6 files against their recorded SHA-256 values.")
    args = parser.parse_args()
    reference = json.loads((ROOT / "learned-catalogue-results.json").read_text(encoding="utf8"))
    summary = json.loads((ROOT / "learned-catalogue-summary.json").read_text(encoding="utf8"))
    verify_candidate(json.loads(args.receipt.read_text(encoding="utf8")), reference, summary, args.cache)


if __name__ == "__main__":
    main()
