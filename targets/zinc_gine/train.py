"""Matched GINE positional-encoding experiments on the official ZINC-12k split."""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import torch
from torch import nn


class Molecules:
    def __init__(self, structure_only=False):
        atoms, bonds, y = torch.load("zinc_on_server.pt", map_location="cuda", weights_only=True)
        self.atoms = atoms.long()
        self.y = y
        self.mask = atoms.ne(0).float().unsqueeze(-1)
        self.bonds = bonds
        if structure_only:
            self.atoms = atoms.ne(0).long()
            self.bonds = bonds.ne(0).to(bonds.dtype)
        self.norm = bonds.ne(0).float()
        inv = self.norm.sum(-1).clamp_min(1).rsqrt()
        self.norm *= inv.unsqueeze(-1) * inv.unsqueeze(-2)
        self.raw = torch.empty((*atoms.shape, 5), device="cuda")
        for i in range(0, len(atoms), 250):
            a = self.norm[i:i + 250]
            self.raw[i:i + 250] = diagonal_powers(a)
        real = self.raw[:10000][self.mask[:10000, :, 0].bool()]
        self.rw = (self.raw - real.mean(0)) / real.std(0).clamp_min(0.05)
        self.rw *= self.mask


def diagonal_powers(a):
    a2 = torch.bmm(a, a)
    a3 = torch.bmm(a2, a)
    return torch.stack([(a * a).sum(-1), (a * a2).sum(-1), (a2 * a2).sum(-1),
                        (a2 * a3).sum(-1), (a3 * a3).sum(-1)], dim=-1)


def uniform_shift(raw, p):
    moments = [torch.ones_like(p), torch.zeros_like(p), *raw.unbind(-1)]
    powers = [torch.ones_like(p), p, p.square(), p.pow(3), p.pow(4), p.pow(5), p.pow(6)]
    return torch.stack([sum(math.comb(k, j) * moments[j] * powers[k - j]
                            for j in range(k + 1)) for k in range(2, 7)], dim=-1)


class GINEBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.edge = nn.Parameter(torch.empty(1, 4, 1, width))
        nn.init.normal_(self.edge, std=0.25)
        self.epsilon = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, x, adjacency, mask):
        messages = torch.relu(x.unsqueeze(1) + self.edge).flatten(1, 2)
        aggregated = torch.bmm(adjacency, messages)
        update = self.mlp((1 + self.epsilon) * x + aggregated)
        return (x + self.norm(update)) * mask


class GINE(nn.Module):
    def __init__(self, mode, width=64, depth=4):
        super().__init__()
        self.mode = mode
        self.atom = nn.Embedding(29, width, padding_idx=0)
        self.potential = nn.Sequential(nn.Linear(5, 16), nn.SiLU(), nn.Linear(16, 1))
        self.pe = nn.Sequential(nn.Linear(11, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([GINEBlock(width) for _ in range(depth)])
        self.node_score = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        self.graph_score = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, d, indices):
        mask = d.mask[indices]
        rw = d.rw[indices]
        raw = d.raw[indices]
        no_p = torch.zeros_like(rw[..., 0])
        if self.mode in ("adaptive", "static"):
            p = 0.5 * self.potential(rw).squeeze(-1).tanh() * mask[..., 0]
        elif self.mode == "fixed":
            p = 0.5 * rw[..., 0].tanh() * mask[..., 0]
        else:
            p = no_p
        if self.mode in ("adaptive", "fixed"):
            response = diagonal_powers(d.norm[indices] + torch.diag_embed(p)) - raw
        elif self.mode == "static":
            response = uniform_shift(raw, p) - raw
        else:
            response = torch.zeros_like(raw)
        if self.mode == "none":
            rw = torch.zeros_like(rw)
        pe = torch.cat((rw, p.unsqueeze(-1), response * 4), dim=-1)
        bonds = d.bonds[indices]
        bond_types = torch.arange(1, 5, device="cuda", dtype=bonds.dtype)
        adj = (bonds.unsqueeze(2) == bond_types[None, None, :, None]).flatten(2).to(torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = (self.atom(d.atoms[indices]) + self.pe(pe)) * mask
            for block in self.blocks:
                x = block(x, adj, mask)
            node_score = (self.node_score(x).float() * mask).sum(1) / 4
            mean_pool = x.sum(1) / mask.sum(1).clamp_min(1)
            graph_score = self.graph_score(mean_pool).float()
        return (node_score + graph_score).squeeze(-1)


@torch.inference_mode()
def evaluate(model, d, start, batch):
    chunks = []
    for lo in range(start, start + 1000, batch):
        ix = torch.arange(lo, min(lo + batch, start + 1000), device="cuda")
        chunks.append((model(d, ix) - d.y[ix]).abs())
    return float(torch.cat(chunks).mean())


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def train_one(d, seed, mode, epochs, batch, variant, out):
    torch.manual_seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed + 100000)
    model = GINE(mode).cuda()
    indices = torch.empty(batch, device="cuda", dtype=torch.long)
    lr = torch.tensor(0.0015, device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, fused=True, capturable=True)

    def step():
        opt.zero_grad(set_to_none=True)
        err = (model(d, indices) - d.y[indices]).abs().mean()
        err.backward()
        opt.step()

    cg = None
    it = 0
    start = time.monotonic()
    best = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        lr.fill_(0.00008 + 0.5 * (0.0015 - 0.00008) * (1 + math.cos(math.pi * (epoch-1) / epochs)))
        perm = torch.randperm(10000, generator=generator, device="cuda")
        for ids in perm.split(batch):
            indices.copy_(ids)
            if it < 3:
                step()
            else:
                if cg is None:
                    torch.cuda.synchronize()
                    cg = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(cg):
                        step()
                cg.replay()
            it += 1
        val = evaluate(model, d, 10000, batch)
        assert math.isfinite(val)
        history.append(val)
        if val < best:
            best, best_epoch = val, epoch
            weights = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            print(seed, mode, epoch, round(val, 5), round(time.monotonic()-start, 1), flush=True)
    step_count = int(opt.state[model.atom.weight]["step"].item())
    assert step_count == epochs * (10000 // batch), (step_count, it)
    model.load_state_dict(weights)
    val = evaluate(model, d, 10000, batch)
    assert abs(val - best) < 1e-6
    test = evaluate(model, d, 11000, batch)
    result = dict(variant=variant, mode=mode, seed=seed, epochs=epochs, batch=batch,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  best_val_epoch=best_epoch, val_mae=val, test_mae=test, val_history=history,
                  optimizer_steps=step_count, seconds=time.monotonic()-start)
    write_json(out, result)
    print("FINISHED", variant, seed, mode, best_epoch, best, test, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["molecular", "structure"], default="molecular")
    parser.add_argument("--modes", nargs="+", default=["static", "adaptive", "rwse", "fixed", "none"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1241, 1242, 1243, 1244])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=200)
    args = parser.parse_args()
    assert 10000 % args.batch == 0 and 1000 % args.batch == 0
    assert set(args.modes) <= {"static", "adaptive", "rwse", "fixed", "none"}
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    d = Molecules(args.variant == "structure")
    protocol = dict(arguments=vars(args), gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                    cuda=str(torch.version.cuda), python=platform.python_version(),
                    train_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    data_metadata=json.loads(Path("split_metadata.json").read_text()))
    output = Path(f"metrics_{args.variant}_{args.epochs}")
    output.mkdir(exist_ok=True)
    write_json(output / "protocol.json", protocol)
    for seed in args.seeds:
        for mode in args.modes:
            path = output / f"{mode}_{seed}.json"
            if not path.exists():
                train_one(d, seed, mode, args.epochs, args.batch, args.variant, path)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
