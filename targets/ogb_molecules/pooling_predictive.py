"""Reproduce the matched all-atom quartic ablation on three official OGB splits.

Run on the benchmark server:
  python pooling_predictive.py /path/to/secondary /path/to/new/output
The first directory supplies the original benchmark.py, regression.py and
server_only_regression/{esol,freesolv,lipophilicity}. No dataset is downloaded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import sys
import time

import numpy as np
import scipy.stats
import torch

MODES = ("root_local", "global_quartic", "full_feedback")
PAIRS = ((1, 0), (2, 1), (2, 0))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_seed(seed):
    outputs = {}
    errors = {}
    for name in MODES:
        mode, values, stats = INPUTS[name]
        b.MODES = (mode,)
        b.ARRAYS = values
        b.STATS = {mode: stats}
        rows, loss = original.fit(seed)
        row, = rows
        row["mode"] = name
        outputs[name] = row
        errors[name] = loss[mode]
    return outputs, errors


def evaluate(original_results):
    rows = [out for out, _ in original_results]
    # Shape: fit, arm, molecule. Both resampling schemes preserve all pairings.
    squares = np.array([[err[name] for name in MODES] for _, err in original_results])
    fit_rmse = np.sqrt(squares.mean(axis=2))
    sizes = {row[name]["trainable_parameters"] for row in rows for name in MODES}
    assert len(sizes) == 1
    rng = np.random.default_rng(7001)
    by_molecule = []
    n = squares.shape[-1]
    for _ in range(20):
        weights = rng.multinomial(n, np.full(n,1/n), size=100)/n
        by_molecule.append(np.sqrt(squares @ weights.T).mean(axis=0))
    by_molecule = np.concatenate(by_molecule, axis=1)
    result = {}
    for i,j in PAIRS:
        seed_diff = fit_rmse[:, i]-fit_rmse[:, j]
        estimate = seed_diff.mean()
        se = scipy.stats.sem(seed_diff)
        margin = scipy.stats.t.ppf(.975,len(seed_diff)-1)*se
        mol_interval = np.quantile(by_molecule[i]-by_molecule[j], [.025,.975])
        result[f"{MODES[i]}_minus_{MODES[j]}"] = {
            "mean": float(estimate),
            "seed_t_95": [float(estimate-margin),float(estimate+margin)],
            "molecule_bootstrap_95": mol_interval.tolist(),
            "fit_differences":seed_diff.tolist(),
            "wins_of_first":int(np.count_nonzero(seed_diff<0)),
        }
    return dict(arms={name:{"mean_test_rmse":float(fit_rmse[:,i].mean()),
                             "sd_test_rmse":float(fit_rmse[:,i].std(ddof=1))}
                      for i,name in enumerate(MODES)}, paired=result,
                fits=rows, parameter_count=sizes.pop())


def execute(name,stem,size,args):
    global INPUTS
    import torch as pt
    root = args.project / "server_only_regression" / stem
    b.ROOT = root
    b.SPLITS = [pt.tensor(b.load(f"split/scaffold/{split}.csv.gz").ravel())
                for split in ("train","valid","test")]
    b.LABELS = pt.tensor(b.load("raw/graph-label.csv.gz").ravel(), dtype=pt.float32)
    assert len(b.LABELS)==size and pt.isfinite(b.LABELS).all()
    assert np.array_equal(np.sort(np.concatenate(b.SPLITS)), np.arange(size))
    raw = [pt.tensor(x) for x in b.featurize()]
    bits, coeff, moments = raw
    # Dimension order: graph, (feedback/local), atom group, moment 2..4, alpha power.
    # For the all-atom sum the single differing coefficient is k=4, degree in alpha=2.
    # Cast-to-FP32 differences in otherwise equal coefficients are negligible.
    gap = coeff[:, 0, -1]-coeff[:, 1, -1]
    changed = gap[:,2,2].clone()
    gap[:,2,2] = 0
    assert gap.abs().max().item()<2e-5
    assert changed.max().item()<2e-5
    hybrid = coeff.clone()
    hybrid[:,1,-1,2,2] = coeff[:,0,-1,2,2]
    inputs = {"root_local":("local",raw),
              "global_quartic":("local",[bits,hybrid,moments]),
              "full_feedback":("feedback",raw)}
    INPUTS = {}
    for label,(mode,values) in inputs.items():
        b.ARRAYS = values
        INPUTS[label] = mode,values,b.training_stats(mode)
    t0=time.monotonic()
    with mp.get_context("fork").Pool(len(original.SEEDS)) as pool:
        results=pool.map(run_seed,original.SEEDS)
    out=evaluate(results)
    out.update(seconds_training=time.monotonic()-t0, split_counts=list(map(len,b.SPLITS)),
               largest_other_coeff_gap=gap.abs().max().item(),
               split_file_sha256={k:sha(root/f"split/scaffold/{k}.csv.gz")
                                  for k in ("train","valid","test")})
    # External test of reproducibility for both reference arms, regardless of direction.
    prior={
        (x["seed"],x["mode"]):x["test_rmse"] for x in
        json.loads((args.project/"regression_aggregates.json").read_text())["datasets"][name]["rows"]
    }
    out["max_rmse_difference_to_original_arms"]=max(
        abs(row[m]["test_rmse"]-prior[(row[m]["seed"],original_mode)])
        for row in out["fits"] for m,original_mode in
        (("root_local","local"),("full_feedback","feedback"))
    )
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    sys.path.insert(0,str(args.project))
    import benchmark as b
    import regression as original
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    b.ARGS.update(original.SETUP)
    assert args.output.is_dir() and not (args.output/"results.json").exists()
    report=dict(protocol="All three official OGB molecule regression benchmarks; official scaffold splits; "
                "matched 8 original seeds, earliest minimum validation RMSE per fit. All arms same trainable "
                "architecture and features except replacement of designated polynomial coefficients. No tuning.",
                modes=MODES, seeds=original.SEEDS,training=original.SETUP,
                software={"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__},
                scripts={p.name:sha(p) for p in (Path(__file__),args.project/"benchmark.py",args.project/"regression.py")},
                datasets={})
    start=time.monotonic()
    for name,(stem,size) in original.RELEASES.items():
        print("Starting",name,flush=True)
        report["datasets"][name]=execute(name,stem,size,args)
        (args.output/"progress.json").write_text(json.dumps(report, indent=2))
        print(name,report["datasets"][name]["arms"],flush=True)
    report["elapsed_seconds"]=time.monotonic()-start
    (args.output/"results.new").write_text(json.dumps(report,indent=2))
    (args.output/"results.new").replace(args.output/"results.json")
    print("COMPLETE",flush=True)
