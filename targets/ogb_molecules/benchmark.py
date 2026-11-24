"""Paired official OGB scaffold evaluation. Keep the feature cache on the data host."""
import copy
import hashlib
import json
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import average_precision_score, roc_auc_score
import sklearn
import torch
from torch import nn

ROOT = Path(os.environ.get("MOLHIV_ROOT", Path(__file__).resolve().parent / "server_only_molhiv"))
HERE = Path(__file__).resolve().parent
SEEDS = (103, 211, 307, 401)
MODES = ("chemical", "static", "local", "feedback")
ATOM_GROUPS = (5, 6, 7, 8, 14, 15, 16, 34, 52)  # zero-based OGB atomic number
GROUPS = len(ATOM_GROUPS) + 2  # one other-atom channel and one all-atom channel
BITS = 1024
ARGS = dict(epochs=36, patience=8, lr=0.001, batch=384, width=128, dropout=.22)


def load(name):
    return pd.read_csv(ROOT / name, header=None).to_numpy()


def mix(payload):
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def fingerprint(nodes, edges, bonds):
    adj = [dict() for _ in nodes]
    for (i, j), bond in zip(edges, bonds):
        value = bytes(bond)
        adj[i][j] = value
        adj[j][i] = value
    colors = [mix(bytes(atom)) for atom in nodes]
    bits = np.zeros(BITS, dtype=np.uint8)
    for level in range(3):
        bits[[c % BITS for c in colors]] = 1
        if level < 2:
            digests = [c.to_bytes(8, "little") for c in colors]
            colors = [mix(digests[i] + b"".join(sorted(b + digests[j] for j, b in ns.items())))
                      for i, ns in enumerate(adj)]
    return bits


def coefficients(adj, verify=False):
    # D_ii = alpha*z_i. Coefficient order is 1, alpha, ..., alpha^4.
    adj2 = adj @ adj
    adj3 = adj2 @ adj
    s2 = np.diag(adj2)
    s3 = np.diag(adj3)
    s4 = np.sum(adj2 * adj2, axis=1)
    z = 2 * s2 - 1
    a = np.zeros((len(adj), 3, 5))
    a[:, :, 0] = np.stack([s2, s3, s4], axis=1)
    a[:, 0, 2] = z ** 2
    a[:, 1, 1] = 2 * z * s2 + (adj ** 2) @ z
    a[:, 1, 3] = z ** 3
    a[:, 2, 1] = 2 * z * s3 + 2 * (adj * adj2) @ z
    a[:, 2, 2] = 3 * z ** 2 * s2 + 2 * z * ((adj ** 2) @ z) + (adj ** 2) @ (z ** 2)
    a[:, 2, 4] = z ** 4
    local = a.copy()
    local[:, 1, 1] = 3 * z * s2
    local[:, 2, 1] = 4 * z * s3
    local[:, 2, 2] = 6 * z ** 2 * s2
    s5 = np.sum(adj2 * adj3, axis=1)
    s6 = np.sum(adj3 ** 2, axis=1)
    s7 = np.sum((adj3 @ adj) * adj3, axis=1)
    if verify:
        for alpha in (-.37, .61):
            field = adj + np.diag(alpha * z)
            value = a @ (alpha ** np.arange(5))
            for k in range(2, 5):
                assert np.allclose(value[:, k-2], np.diag(np.linalg.matrix_power(field, k)), atol=2e-11)
                expected = [np.linalg.matrix_power(adj + np.eye(len(adj)) * (alpha*z[i]), k)[i, i]
                            for i in range(len(adj))]
                assert np.allclose(local[:, k-2, :] @ (alpha ** np.arange(5)), expected, atol=2e-11)
    return a, local, np.stack((s5, s6, s7), axis=1)


def featurize():
    nlist = load("raw/num-node-list.csv.gz").ravel()
    elist = load("raw/num-edge-list.csv.gz").ravel()
    nodes = load("raw/node-feat.csv.gz").astype("uint8")
    edges = load("raw/edge.csv.gz")
    bonds = load("raw/edge-feat.csv.gz").astype("uint8")
    assert sum(nlist) == len(nodes) and sum(elist) == len(edges) == len(bonds)
    fp = np.empty((len(nlist), BITS), dtype="uint8")
    coeff = np.empty((len(nlist), 2, GROUPS, 3, 5), dtype="float32")
    higher = np.empty((len(nlist), GROUPS, 3), dtype="float32")
    lookup = {x:i for i,x in enumerate(ATOM_GROUPS)}
    offset = eoffset = 0
    for g, (n, m) in enumerate(zip(nlist, elist)):
        ns = nodes[offset:offset+n]
        es = edges[eoffset:eoffset+m]
        bs = bonds[eoffset:eoffset+m]
        assert not len(es) or (es.min() >= 0 and es.max() < n)
        adj = np.zeros((n, n), dtype=np.float64)
        adj[es[:, 0], es[:, 1]] = adj[es[:, 1], es[:, 0]] = 1
        assert not adj.diagonal().any()
        deg = adj.sum(1)
        norm = np.maximum(1, deg)**(-.5)
        adj *= norm[:, None] * norm[None, :]
        a, local, s567 = coefficients(adj, verify=(g < 3))
        fp[g] = fingerprint(ns, es, bs)
        pooling = np.zeros((GROUPS, n))
        pooling[[lookup.get(x, GROUPS-2) for x in ns[:, 0]], np.arange(n)] = 1
        pooling[-1] = 1
        pooling /= max(1, n)**.5
        coeff[g, 0] = (pooling @ a.reshape(n, -1)).reshape(GROUPS, 3, 5)
        coeff[g, 1] = (pooling @ local.reshape(n, -1)).reshape(GROUPS, 3, 5)
        higher[g] = pooling @ s567
        offset += n
        eoffset += m
        if (g+1) % 10000 == 0:
            print("features", g+1, flush=True)
    return fp, coeff, higher


class Predictor(nn.Module):
    def __init__(self, mode, shift, sd):
        super().__init__()
        self.mode = mode
        self.alpha_raw = nn.Parameter(torch.tensor(float(np.arctanh(.5))), requires_grad=mode in ("local", "feedback"))
        self.register_buffer("shift", shift)
        self.register_buffer("sd", sd)
        self.layers = nn.Sequential(nn.Linear(BITS + GROUPS * 6, ARGS["width"]), nn.LayerNorm(ARGS["width"]),
                                    nn.ReLU(), nn.Dropout(ARGS["dropout"]), nn.Linear(ARGS["width"], 1))

    def forward(self, ids):
        chem, coeff, high = ARRAYS
        coeff = coeff[ids, int(self.mode == "local")]
        if self.mode == "chemical":
            pooled = torch.zeros((len(ids), GROUPS * 6))
        else:
            first = coeff[..., 0]
            if self.mode == "static":
                second = high[ids]
            else:
                alpha = self.alpha_raw.tanh()
                powers = torch.stack([alpha**k for k in range(5)])
                second = coeff @ powers
            pooled = (torch.cat([first, second], -1).flatten(1) - self.shift) / self.sd
        return self.layers(torch.cat((chem[ids].float(), pooled), -1)).flatten()


def training_stats(mode):
    _, coeff, high = ARRAYS
    train_ids = SPLITS[0]
    c = coeff[train_ids, int(mode == "local")]
    second = high[train_ids] if mode == "static" else c @ torch.tensor([.5**k for k in range(5)])
    x = torch.cat((c[..., 0], second), dim=-1).flatten(1)
    return x.mean(0), torch.clamp(x.std(0), min=.05)


def train(seed):
    summaries = []
    probabilities = {}
    tr, va, te = SPLITS
    for mode in MODES:
        start = time.monotonic()
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = Predictor(mode, *STATS[mode])
        optimizer = torch.optim.AdamW(model.parameters(), lr=ARGS["lr"], weight_decay=.001)
        positive = LABELS[tr].mean()
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=((1-positive)/positive).sqrt())
        best, state, epoch_best = -1., None, 0
        for epoch in range(1, ARGS["epochs"]+1):
            model.train()
            for ids in tr[torch.randperm(len(tr), generator=generator)].split(ARGS["batch"]):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(ids), LABELS[ids])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val = roc_auc_score(LABELS[va], torch.cat([model(ids) for ids in va.split(1024)]))
            if val > best:
                best, state, epoch_best = val, copy.deepcopy(model.state_dict()), epoch
            if epoch-epoch_best >= ARGS["patience"]:
                break
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(ids) for ids in te.split(1024)]).numpy()
        probabilities[mode] = logits
        row = dict(seed=seed, mode=mode, best_epoch=epoch_best, epochs_run=epoch, val_auc=best,
                   test_auc=roc_auc_score(LABELS[te], logits), test_ap=average_precision_score(LABELS[te], logits),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   alpha=float(model.alpha_raw.detach().tanh()) if mode in ("local","feedback") else None,
                   elapsed_seconds=time.monotonic()-start)
        summaries.append(row)
        print("fit", seed, mode, epoch_best, flush=True)
    return summaries, probabilities


def paired_uncertainty(predictions):
    # Conditional test-set uncertainty on one public split. Resample positive and negative molecules separately.
    y = LABELS[SPLITS[2]].numpy()
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    order = {mode:np.stack([p[mode] for p in predictions]) for mode in MODES}
    wins = {}
    for mode in MODES:
        margins = order[mode][:,pos,None] - order[mode][:,None,neg]
        wins[mode] = ((margins > 0) + .5*(margins == 0)).mean(0).astype("float32")
    delta = np.stack([wins["feedback"]-wins[mode] for mode in ("static","local")])
    rng = np.random.default_rng(7001)
    out = np.zeros((2000, 2))
    for j in range(0, len(out), 100):
        pw = rng.multinomial(len(pos), np.full(len(pos), 1/len(pos)), size=100).astype("float32")/len(pos)
        nw = rng.multinomial(len(neg), np.full(len(neg), 1/len(neg)), size=100).astype("float32")/len(neg)
        for i in range(2):
            out[j:j+100,i] = ((pw @ delta[i]) * nw).sum(1)
    return {mode:dict(point=float(delta[i].mean()),lo=float(np.quantile(out[:,i],.025)),
                      hi=float(np.quantile(out[:,i],.975))) for i, mode in enumerate(("static","local"))}


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    t0 = time.monotonic()
    src = [*ROOT.glob("raw/*.csv.gz"), *ROOT.glob("split/scaffold/*.csv.gz")]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(src)}
    SPLITS = [torch.tensor(load(f"split/scaffold/{s}.csv.gz").ravel()) for s in ("train","valid","test")]
    LABELS = torch.tensor(load("raw/graph-label.csv.gz").ravel(), dtype=torch.float32)
    assert len(LABELS) == 41127 and np.array_equal(np.sort(np.concatenate(SPLITS)), np.arange(len(LABELS)))
    print("Official scaffold sizes", [len(x) for x in SPLITS], flush=True)
    feature_file = HERE / "server_only_features.npz"
    if not feature_file.is_file():
        features = featurize()
        with (HERE / "feature_build.tmp").open("wb") as f:
            np.savez(f, fp=features[0], coeff=features[1], high=features[2])
        (HERE / "feature_build.tmp").replace(feature_file)
    with np.load(feature_file) as archive:
        ARRAYS = [torch.tensor(archive[k]) for k in ("fp", "coeff", "high")]
    STATS = {m:training_stats(m) for m in MODES}
    with mp.get_context("fork").Pool(len(SEEDS)) as pool:
        results = pool.map(train, SEEDS)
    rows = [r for rr,_ in results for r in rr]
    intervals = paired_uncertainty([pr for _,pr in results])
    report = dict(dataset="ogbg-molhiv", scaffold_counts=[len(x) for x in SPLITS],
                  positives=[int(LABELS[x].sum()) for x in SPLITS], args=ARGS, seeds=SEEDS,
                  versions=dict(torch=torch.__version__, sklearn=sklearn.__version__, numpy=np.__version__,
                                scipy=scipy.__version__, python=platform.python_version()), hashes=hashes,
                  split_protocol="Official scaffold split, validation ROC-AUC maximization; earliest tie.",
                  formula_check="Dense matrix powers checked on first three molecules at two signed amplitudes.",
                  rows=rows, feedback_minus_paired_reference=intervals, duration_seconds=time.monotonic()-t0)
    (HERE / "aggregates.tmp").write_text(json.dumps(report, indent=2))
    (HERE / "aggregates.tmp").replace(HERE / "aggregates.json")
    print("COMPLETE", flush=True)
