"""Gradient-trained diagonal-potential networks on each of 100 BREC CFI pairs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch import nn

from evaluate import array_from_g6, diagonal_bank
from paired import t_squared, RIDGE, THRESHOLD

K, STEPS = 24, 24
SEEDS = (117,283,461)
LEARNING_RATE = .012
EMBEDDING_SCALE = 100.


def prepare(raw):
    A = raw.astype("float64") / max(1, int(raw.sum(1).max()))
    first = diagonal_bank(csr_matrix(A), np.zeros(len(A)), K, False)
    return torch.tensor(A), torch.tensor(first)


def batch(graphs):
    return tuple(torch.stack(parts) for parts in zip(*graphs))


class Encoder(nn.Module):
    def __init__(self, rounds, training):
        super().__init__()
        self.rounds = rounds
        moments = training[1].flatten(0,1)
        self.register_buffer("center", moments.mean(0))
        self.register_buffer("sd", moments.std(0,unbiased=False).clamp_min(.05))
        self.field = nn.Sequential(nn.Linear(K,16), nn.Tanh(), nn.Linear(16,1), nn.Tanh())
        self.vertex = nn.Sequential(nn.Linear(2*K,32), nn.Tanh(), nn.Linear(32,16), nn.Tanh())
        self.readout = nn.Linear(16,16)

    def forward(self, x):
        adj, bank = x
        first = (bank-self.center)/self.sd
        if self.rounds==2:
            potential = .45*self.field(first)
            half_adj = .5*adj
            previous = torch.eye(adj.shape[1]).expand_as(adj)
            current = half_adj + torch.diag_embed(potential[:,:,0])
            second = [current.diagonal(dim1=1,dim2=2)]
            for _ in range(2,K+1):
                following = 2*(torch.bmm(half_adj,current)+potential*current)-previous
                previous, current = current, following
                second.append(current.diagonal(dim1=1,dim2=2))
            second = torch.stack(second, dim=-1)
        else:
            second = torch.zeros_like(first)
        nodes = self.vertex(torch.cat((first,second), dim=-1))
        return self.readout(nodes.sum(1)/np.sqrt(adj.shape[1]))


def score(model, graphs, assignment, permutations):
    with torch.no_grad():
        outputs = torch.cat([model(batch(graphs[j:j+4])) for j in range(0,len(graphs),4)])
        outputs = EMBEDDING_SCALE * outputs
        raw = outputs[assignment].numpy()
        perm = (EMBEDDING_SCALE * model(batch(permutations))).numpy()
    statistic, null = t_squared(raw[:64]), t_squared(raw[64:])
    heldout = t_squared(raw[2:64])
    gap = float(np.linalg.norm(raw[0]-raw[1]))
    drift = float(np.linalg.norm(perm-raw[:2],axis=1).max())
    reliable = null < THRESHOLD and not np.isclose(statistic,null,atol=1e-6)
    return dict(separated=bool(reliable and statistic>THRESHOLD), null_failed=bool(null>=THRESHOLD),
                heldout_passed=bool(reliable and heldout>THRESHOLD), gap=gap, drift=drift,
                drift_guard_passed=bool(gap>1e-10+10*drift))


def evaluate(task):
    idx, blobs = task
    torch.set_default_dtype(torch.float64)
    known = list(dict.fromkeys(blobs))
    lookup = {g:i for i,g in enumerate(known)}
    ids = [lookup[g] for g in blobs]
    adjacencies = [array_from_g6(raw) for raw in known]
    graphs = [prepare(adj) for adj in adjacencies]
    training = batch([graphs[i] for i in ids[:2]])
    rng = np.random.default_rng(434+idx)
    permutations = []
    for i in ids[:2]:
        adj = adjacencies[i]
        perm = rng.permutation(len(adj))
        permutations.append(prepare(adj[np.ix_(perm,perm)]))
    out = []
    for seed in SEEDS:
        for rounds in (1,2):
            torch.manual_seed(seed)
            model = Encoder(rounds, training)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            initial_field = torch.cat([p.detach().flatten() for p in model.field.parameters()])
            before = score(model,graphs,ids,permutations)
            for _ in range(STEPS):
                optimizer.zero_grad(set_to_none=True)
                emb = model(training)
                dist = (emb[0]-emb[1]).square().mean()
                loss = -torch.log(dist + 1e-12)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            result = score(model,graphs,ids,permutations)
            current_field = torch.cat([p.detach().flatten() for p in model.field.parameters()])
            out.append(dict(seed=seed, rounds=rounds, before=before, after=result,
                            field_parameter_movement=float((current_field-initial_field).norm())))
    return idx,len(set(blobs[:64])),len(set(blobs[64:])),out


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data=np.load(args.release, mmap_mode="r", allow_pickle=False)
    assert data.shape==(51200,)
    tasks=((idx,[bytes(s) for s in data[64*idx:64*(idx+1)]]+
            [bytes(s) for s in data[64*(idx+400):64*(idx+401)]]) for idx in range(260,360))
    started=time.monotonic()
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(evaluate,tasks,chunksize=1):
            rows.append(row)
            if len(rows)%10==0:
                print(len(rows),"CFI pairs",round(time.monotonic()-started,2),flush=True)
    assert sorted(r[0] for r in rows)==list(range(260,360))
    exact=json.loads(Path("pair_diagnostics.json").read_text())
    expected={r["id"] for r in exact if 260<=r["id"]<360 and any(
        x["arithmetic"]=="modular" and x["rounds"]==2 and x["degree"]==K and x["seed"]==1729 and x["separated"]
        for x in r["observations"])}
    summary=[]
    for seed in SEEDS:
        for rounds in (1,2):
            selected=[(idx,next(f for f in fits if (f["seed"],f["rounds"])==(seed,rounds)))
                      for idx,_,_,fits in rows]
            for stage in ("before","after"):
                separated={i for i,f in selected if f[stage]["separated"]}
                success=[f for i,f in selected if i in separated]
                summary.append(dict(seed=seed,rounds=rounds,stage=stage,
                    accepted_CFI_ids=sorted(separated),missed_modular_ids=sorted(expected-separated),
                    outside_modular_ids=sorted(separated-expected),
                    null_failures=sum(f[stage]["null_failed"] for _,f in selected),
                    passed_on_last_31=sum(f[stage]["heldout_passed"] for _,f in selected),
                    permutation_guard_failures_among_accepted=sum(not f[stage]["drift_guard_passed"] for f in success),
                    minimum_accepted_gap=min((f[stage]["gap"] for f in success), default=None),
                    max_permutation_drift=max(f[stage]["drift"] for _,f in selected),
                    field_movement_on_accepted=[f["field_parameter_movement"] for f in success] if stage=="after" else []))
    payload=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset_sha256=hashlib.file_digest(args.release.open("rb"), "sha256").hexdigest(),
        pairs=100,seeds=SEEDS,steps=STEPS,degree=K,lr=LEARNING_RATE,embedding_scale=EMBEDDING_SCALE,
        ridge=RIDGE,threshold=THRESHOLD,modular_upper_reference_ids=sorted(expected),
        unique_alternative_encodings_range=[min(r[1] for r in rows),max(r[1] for r in rows)],
        unique_reliability_encodings_range=[min(r[2] for r in rows),max(r[2] for r in rows)],
        official_alternative_pairs=32,official_reliability_pairs=32,
        training="First alternative pair. Adam optimizes negative log mean squared embedding difference + "
                 "epsilon inside log (1e-12); gradient norm cap=5. Smooth tanh field and vertex MLP; sum/sqrt(n).",
        secondary_checks="Also remove first alternative pair for 31-pair score with same cutoff; generate fresh "
                 "independent random relabelings of first pair to measure embedding drift.",
        software=dict(torch=torch.__version__, numpy=np.__version__), seconds=time.monotonic()-started,rows=summary)
    Path("cfi_neural.json").write_text(json.dumps(payload,indent=2))
    print("complete",flush=True)
