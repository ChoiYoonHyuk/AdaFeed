"""Replay the original prefix, stop each matched PCBA model on validation AP, then evaluate all."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def fit_model(base, x_train, x_valid, targets, observed, positive_weights, val_labels, seed, reference):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = base.net(x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best = -1.0
    best_epoch = 0
    history = []
    prefix_match = False
    for epoch in range(1, 401):
        model.train()
        for indices in torch.randperm(len(x_train), device="cuda").split(2048):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x_train[indices]), targets[indices],
                                                       pos_weight=positive_weights, reduction="none")
            known = observed[indices]
            objective = (loss * known).sum() / known.sum()
            objective.backward()
            optimizer.step()
        if epoch % 4:
            continue
        score = base.fast_ap(val_labels, base.predict(model, x_valid))
        history.append(float(score))
        if epoch == 20:
            assert reference["epoch"] == 20 and reference["seed"] == seed
            assert abs(score - reference["validation_ap"]) < 1e-12, (score, reference["validation_ap"])
            assert all(torch.equal(w, reference["state"][k]) for k, w in model.state_dict().items())
            prefix_match = True
        if score > best:
            best, best_epoch = score, epoch
            state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 32:
            break
    assert prefix_match and best_epoch >= 20
    return {"state": state, "epoch": best_epoch, "stop_epoch": epoch,
            "reason": "validation_patience" if epoch - best_epoch >= 32 else "hard_cap",
            "validation_ap": float(best), "validation_aps_every_four_epochs": history, "seed": seed,
            "prefix_bitwise_match": prefix_match, "parameters": sum(p.numel() for p in model.parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("baseline_code", "dataset", "features", "baseline_fits", "work"):
        parser.add_argument(arg, type=Path)
    a = parser.parse_args()
    sys.path.insert(0, str(a.baseline_code.resolve()))
    import train_pcba as base
    from prepare_pcba import digest
    from ogb.graphproppred import Evaluator
    a.work.mkdir(parents=True, exist_ok=True)
    assert "H100" in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    tr, va, te = base.split_arrays(a.dataset)
    y = base.labels(a.dataset)
    assert y.shape == (437929, 128) and np.array_equal(np.sort(np.r_[tr, va, te]), np.arange(len(y)))
    old = json.loads((a.baseline_fits / "fit_summary.json").read_text())
    protocol_sha = digest(Path(__file__).with_name("protocol.md"))
    source_sha = digest(__file__)
    assert old["source_sha256"] == digest(a.baseline_code / "train_pcba.py")
    assert old["protocol_sha256"] == digest(a.baseline_code / "protocol.md")
    known = torch.isfinite(torch.as_tensor(y[tr], device="cuda"))
    target = torch.as_tensor(y[tr], device="cuda").nan_to_num()
    positives = target.sum(0)
    weights = ((known.sum(0) - positives) / positives).sqrt().clamp(1, 20)
    def load_bank(name):
        return base.stdize(np.load(a.features / (name + ".npy"), mmap_mode="r"), tr)
    common = load_bank("common")
    local = load_bank("local")
    highs = (torch.zeros_like(local), local, load_bank("spatial"))
    checkpoints = {}
    for arm, high in zip(base.ARMS, highs):
        x = torch.cat([common, high], 1)
        xt, xv = x[tr], x[va]
        checks = []
        for seed in base.SEEDS:
            name = f"{arm}-{seed}.pt"
            path = a.work / name
            if path.exists():
                record = torch.load(path, map_location="cuda", weights_only=True)
                assert (record["protocol_sha256"], record["source_sha256"]) == (protocol_sha, source_sha)
            else:
                reference = torch.load(a.baseline_fits / name, map_location="cuda", weights_only=True)
                record = fit_model(base, xt, xv, target, known, weights, y[va], seed, reference)
                record.update(protocol_sha256=protocol_sha, source_sha256=source_sha)
                tmp = path.with_suffix(".tmp")
                torch.save(record, tmp)
                os.replace(tmp, path)
            checks.append({k: v for k,v in record.items() if k != "state"})
            print("validation stopped", arm, seed, record["epoch"], record["stop_epoch"], record["reason"], flush=True)
        checkpoints[arm] = checks
        del x, xt, xv
    assert sum(map(len, checkpoints.values())) == 15
    print("All 15 validation selections are fixed.", flush=True)
    evaluator = Evaluator(name="ogbg-molpcba")
    scores = {}
    dim = common.shape[1] + local.shape[1]
    for arm, high in zip(base.ARMS, highs):
        x_test = torch.cat([common[te], high[te]], 1)
        model = base.net(dim)
        scores[arm] = []
        for seed in base.SEEDS:
            record = torch.load(a.work / f"{arm}-{seed}.pt", map_location="cuda", weights_only=True)
            model.load_state_dict(record["state"])
            pred = base.predict(model, x_test)
            official = float(evaluator.eval({"y_true": y[te], "y_pred": pred})["ap"])
            assert abs(official - base.fast_ap(y[te], pred)) < 1e-11
            np.save(a.work / f"{arm}-{seed}-test.npy", pred)
            scores[arm].append(official)
    out = {"source_sha256": source_sha, "protocol_sha256": protocol_sha,
           "base_train_sha256": digest(a.baseline_code / "train_pcba.py"),
           "baseline_fit_summary_sha256": digest(a.baseline_fits / "fit_summary.json"),
           "device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
           "seeds": list(base.SEEDS), "checkpoints": checkpoints, "official_seed_ap": scores,
           "split_n": [len(tr), len(va), len(te)], "common_dim": int(common.shape[1]), "input_dim": int(dim),
           "split_sha256": {s:digest(a.dataset / "split/scaffold" / (s+".csv.gz")) for s in ("train","valid","test")}}
    (a.work / "fit_summary.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    print("All selected models evaluated.", flush=True)


if __name__ == "__main__":
    main()
