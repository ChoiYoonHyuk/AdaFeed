"""Official OGB molecular regression family; execute on the benchmark host."""
import copy
import hashlib
import http.client
import json
import multiprocessing as mp
import os
import platform
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
import torch
from torch import nn

import benchmark as b

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("OGB_REGRESSION_ROOT", HERE / "server_only_regression"))
RELEASES = {"ogbg-molesol":("esol",1128), "ogbg-molfreesolv":("freesolv",642),
            "ogbg-mollipo":("lipophilicity",4200)}
SEEDS = (103,211,307,401,503,601,701,809)
SETUP = dict(epochs=120, patience=20, lr=.001, weight_decay=.001, batch=64,
             width=128, dropout=.22, grad_clip=5.)


def release(stem):
    archive = DATA / (stem + ".zip")
    urls = [
        f"https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/{stem}.zip",
        f"https://data.dgl.ai/dataset/ogb/ogbg_mol{stem}.zip",
        f"https://data.dgl.ai/dataset/ogb/{stem}.zip",
    ]
    metadata = DATA / (stem + "_source.json")
    if not archive.is_file():
        tmp = archive.with_suffix(".part")
        failures = []
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=65) as response, tmp.open("wb") as out:
                    shutil.copyfileobj(response, out)
                with zipfile.ZipFile(tmp) as z:
                    if z.testzip() is not None:
                        raise ValueError("Zip CRC failed")
                tmp.replace(archive)
                metadata.write_text(json.dumps({"url":url, "sha256":hashlib.sha256(archive.read_bytes()).hexdigest()}))
                break
            except (OSError, ValueError, zipfile.BadZipFile, http.client.HTTPException) as exc:
                failures.append(str(exc))
        else:
            raise RuntimeError(f"{stem}: {failures}")
    dest = DATA / stem
    with zipfile.ZipFile(archive) as z:
        for subdir, names in (("raw",("num-node-list","num-edge-list","edge","edge-feat","node-feat","graph-label")),
                              ("split/scaffold",("train","valid","test"))):
            for name in names:
                rel = f"{subdir}/{name}.csv.gz"
                member, = [p for p in z.namelist() if p == rel or p.endswith("/" + rel)]
                path = dest / rel
                path.parent.mkdir(exist_ok=True, parents=True)
                with z.open(member) as src, path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    return dest, json.loads(metadata.read_text())


def fit(seed):
    tr, va, te = b.SPLITS
    ymean = b.LABELS[tr].mean()
    ysd = b.LABELS[tr].std(unbiased=False)
    assert ysd > 0
    outcomes, squared_errors = [], {}
    for mode in b.MODES:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = b.Predictor(mode, *b.STATS[mode])
        optimizer = torch.optim.AdamW(model.parameters(), lr=SETUP["lr"], weight_decay=SETUP["weight_decay"])
        best, state, best_epoch = float("inf"), None, 0
        for epoch in range(1, SETUP["epochs"]+1):
            model.train()
            perm = tr[torch.randperm(len(tr), generator=generator)]
            for ids in perm.split(SETUP["batch"]):
                optimizer.zero_grad(set_to_none=True)
                preds = model(ids)
                loss = nn.functional.mse_loss(preds, (b.LABELS[ids]-ymean)/ysd)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), SETUP["grad_clip"])
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_preds = model(va)*ysd+ymean
                val = ((val_preds-b.LABELS[va]).square().mean().sqrt()).item()
            if val < best:
                best, state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
            if epoch-best_epoch >= SETUP["patience"]:
                break
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            pred = model(te)*ysd+ymean
            sq = (pred.double() - b.LABELS[te].double()).square().numpy()
        squared_errors[mode] = sq
        outcomes.append(dict(seed=seed, mode=mode, epochs_run=epoch, best_epoch=best_epoch, val_rmse=best,
                             test_rmse=float(np.sqrt(sq.mean())), test_mae=float(np.sqrt(sq).mean()),
                             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                             alpha=float(model.alpha_raw.detach().tanh()) if mode in ("feedback","local") else None))
    print("finished seed", seed, flush=True)
    return outcomes, squared_errors


def intervals(results):
    scores = np.stack([[r[1][m] for m in b.MODES] for r in results])
    n = scores.shape[-1]
    rng = np.random.default_rng(7001)
    diffs = []
    for j in range(20):
        weights = rng.multinomial(n, np.full(n, 1/n), size=100)/n
        rmse = np.sqrt(scores @ weights.T).mean(0)
        diffs.append(rmse[-1:]-rmse[:-1])
    diffs = np.concatenate(diffs, axis=1)
    raw = np.sqrt(scores.mean(axis=2)).mean(0)
    return {mode:dict(point=float(raw[-1]-raw[i]), lo=float(np.quantile(diffs[i],.025)),
                      hi=float(np.quantile(diffs[i],.975))) for i, mode in enumerate(b.MODES[:-1])}


def execute(name, stem, size):
    b.ROOT, source = release(stem)
    print("release", name, source, flush=True)
    b.SPLITS = [torch.tensor(b.load(f"split/scaffold/{s}.csv.gz").ravel()) for s in ("train","valid","test")]
    b.LABELS = torch.tensor(b.load("raw/graph-label.csv.gz").ravel(), dtype=torch.float32)
    assert len(b.LABELS)==size and np.array_equal(np.sort(np.concatenate(b.SPLITS)), np.arange(size))
    assert torch.isfinite(b.LABELS).all()
    b.ARRAYS = [torch.tensor(x) for x in b.featurize()]
    b.STATS = {m:b.training_stats(m) for m in b.MODES}
    with mp.get_context("fork").Pool(len(SEEDS)) as pool:
        results = pool.map(fit, SEEDS)
    source["input_sha256"] = {str(p.relative_to(b.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(b.ROOT.glob("**/*.csv.gz"))}
    return dict(source=source, split_counts=[len(s) for s in b.SPLITS], rows=[r for rows,_ in results for r in rows],
                feedback_minus_paired_reference=intervals(results))


if __name__=="__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    b.ARGS.update(SETUP)
    DATA.mkdir(parents=True, exist_ok=True)
    report = dict(protocol="Official scaffold, all molecules. Training-target standardized MSE. "
                  "Earliest minimum validation RMSE sets checkpoint. Test RMSE in native units.",
                  bootstrap="Paired molecule bootstrap, 2000 draws, mean RMSE over fixed eight fits.",
                  config=SETUP, seeds=SEEDS, python=platform.python_version(), torch=torch.__version__,
                  prediction_source_sha256=hashlib.sha256((HERE/"benchmark.py").read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), datasets={})
    for name, (stem, size) in RELEASES.items():
        start=time.monotonic()
        report["datasets"][name] = execute(name, stem, size)
        report["datasets"][name]["elapsed_seconds"] = time.monotonic()-start
        print(name, "complete", flush=True)
    (HERE / "regression_aggregates.tmp").write_text(json.dumps(report, indent=2))
    (HERE / "regression_aggregates.tmp").replace(HERE / "regression_aggregates.json")
    print("COMPLETE", flush=True)
