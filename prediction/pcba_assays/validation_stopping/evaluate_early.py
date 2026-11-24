"""Evaluate all validation-selected fits and compare budgets with paired, synchronized deletions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy.stats as stats


def contrast(curves, base_eval):
    seed_effects = curves[:, 0]
    center = float(seed_effects.mean())
    half = float(stats.t.ppf(.975, len(seed_effects)-1) * seed_effects.std(ddof=1) / np.sqrt(len(seed_effects)))
    g = base_eval.G
    return {"mean": center, "per_seed": seed_effects.tolist(),
            "positive_fits": int((seed_effects > 0).sum()), "seed95": [center-half, center+half],
            "molecule95": base_eval.ci_groups(center, curves[:, 1:1+g].mean(axis=0)),
            "scaffold95": base_eval.ci_groups(center, curves[:, 1+g:].mean(axis=0))}


def deleted_curves(fits_dir, fit_summary, observed, y, parts, base, common_support):
    scores = {}
    for arm in base.ARMS:
        curves = []
        for i, seed in enumerate(base.SEEDS):
            pred = np.load(fits_dir / f"{arm}-{seed}-test.npy", mmap_mode="r")
            per_assay = []
            for j in range(y.shape[1]):
                keep = observed[:, j]
                per_assay.append(base.deletion_ap(y[keep, j], pred[keep, j],
                                  [groups[keep] for groups in parts], do_check=(i == 0 and j % 43 == 0)))
            per_assay = np.array(per_assay)
            np.testing.assert_array_equal(np.isfinite(per_assay).sum(0), common_support)
            curve = np.nanmean(per_assay, axis=0)
            assert abs(curve[0]-fit_summary["official_seed_ap"][arm][i]) < 1e-11
            curves.append(curve)
        scores[arm] = np.array(curves)
        print("verified deletion curves", arm, flush=True)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("baseline_code", "dataset", "features", "baseline_fits", "work", "output"):
        parser.add_argument(arg, type=Path)
    a = parser.parse_args()
    assert not a.output.exists()
    sys.path.insert(0, str(a.baseline_code.resolve()))
    import evaluate_pcba as base
    import sklearn, ogb
    from prepare_pcba import digest
    tr, va, te = base.split_arrays(a.dataset)
    y = base.labels(a.dataset)[te]
    parts, structural = base.groupings(a.dataset, y, te)
    pos = (y == 1).sum(axis=0)
    eligible = (pos > 0) & ((y == 0).sum(axis=0) > 0)
    observed = np.isfinite(y)
    # The baseline experiment verified the same eligible assays across every deletion.
    support = np.full(1+2*base.G, eligible.sum())
    short_fit = json.loads((a.baseline_fits/"fit_summary.json").read_text())
    long_fit = json.loads((a.work/"fit_summary.json").read_text())
    assert long_fit["split_sha256"] == short_fit["split_sha256"]
    assert all(q["prefix_bitwise_match"] for v in long_fit["checkpoints"].values() for q in v)
    short = deleted_curves(a.baseline_fits, short_fit, observed, y, parts, base, support)
    long = deleted_curves(a.work, long_fit, observed, y, parts, base, support)
    primary = {c: contrast(long["Spatial"] - long[c], base) for c in ("Local","Common")}
    short_primary = {c: contrast(short["Spatial"] - short[c], base) for c in ("Local","Common")}
    paired = {arm: long[arm] - short[arm] for arm in base.ARMS}
    budget = {arm: contrast(values, base) for arm, values in paired.items()}
    difference_of_gains = {c: contrast(paired["Spatial"] - paired[c], base) for c in ("Local","Common")}
    out = {"benchmark":"ogbg-molpcba", "official_metric":"macro average precision on eligible assays",
           "fit":long_fit, "feature_audit":json.loads((a.features/"feature_audit.json").read_text()),
           "test_structure":structural, "eligible_assays":int(eligible.sum()),
           "observed_test_labels":int(observed.sum()), "positive_test_labels":int(pos.sum()),
           "test_positive_support_min_median_max":[int(pos[eligible].min()),float(np.median(pos[eligible])),int(pos[eligible].max())],
           "eligible_after_delete":{"molecule":[int(eligible.sum())]*2,"scaffold":[int(eligible.sum())]*2},
           "short_seed_ap":short_fit["official_seed_ap"],
           "short_test_ap":{k:float(v[:,0].mean()) for k,v in short.items()},
           "test_ap":{k:float(v[:,0].mean()) for k,v in long.items()},
           "spatial_minus":primary,"short_spatial_minus":short_primary,
           "long_minus_short":budget,"change_in_spatial_minus":difference_of_gains,
           "joint_superiority":all(v[ci][0]>0 for v in primary.values() for ci in ("seed95","molecule95")),
           "source_sha256":digest(__file__), "baseline_eval_sha256":digest(a.baseline_code/"evaluate_pcba.py"),
           "baseline_prepare_sha256":digest(a.baseline_code/"prepare_pcba.py"),
           "ogb_version":ogb.__version__, "sklearn_version":sklearn.__version__}
    with a.output.open("x") as file:
        json.dump(out, file, allow_nan=False, indent=2)
        file.write("\n")
    print("Aggregate budget and model comparisons completed.",flush=True)


if __name__ == "__main__":
    main()
