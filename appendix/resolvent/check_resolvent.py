"""Reproduce Appendix E.4 from the published equations on CPU."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brec"))
from constructions import base_graph


# FP64 score gaps and their derivatives, rounded in the original sensitivity table.
REFERENCE = {
    (1, 1): (7.464e-9, 2.239e-8), (1, 10): (7.326e-6, 2.170e-6),
    (1, 100): (1.238e-3, -5.426e-6), (2, 1): (5.141e-8, 1.542e-7),
    (2, 10): (4.891e-5, 1.418e-5), (2, 100): (6.379e-4, -3.290e-5),
    (5, 1): (4.439e-7, 1.330e-6), (5, 10): (3.892e-4, 1.065e-4),
    (5, 100): (-2.877e-5, -6.561e-6),
}


def decorated_graph(kind, leaves, dtype):
    graph = torch.zeros((16 + leaves, 16 + leaves), dtype=dtype)
    graph[:16, :16] = torch.from_numpy(base_graph(kind).astype(np.float64))
    graph[0, 16:] = 1
    graph[16:, 0] = 1
    return graph


def resolvents(h):
    imaginary_poles = 1j * torch.arange(1, 22, dtype=h.dtype)
    identity = torch.eye(len(h), dtype=h.dtype)
    return torch.linalg.inv(imaginary_poles[:, None, None] * identity - h)


def model(a, theta):
    first = resolvents(a).diagonal(dim1=-2, dim2=-1).T
    centered = first[:, 0].real - first[:, 0].real.mean()
    p = torch.tanh(theta * centered)
    h = a + torch.diag(p)
    second = resolvents(h)
    observed = second.diagonal(dim1=-2, dim2=-1).T
    state = torch.cat((first.real, first.imag, p[:, None], observed.real, observed.imag), dim=1)
    score = torch.dot(state[:, 42], state[:, 43])
    return score, centered, p, second[0], h


def evaluate(graph, strength):
    a = graph.clone().requires_grad_()
    theta = torch.tensor(float(strength), dtype=a.dtype, requires_grad=True)
    score, centered, p, r, h = model(a, theta)
    reverse_derivative, adjacency_gradient = torch.autograd.grad(score, (theta, a))
    if not torch.isfinite(adjacency_gradient).all():
        raise ArithmeticError("Nonfinite adjacency derivative")
    with torch.no_grad():
        dp = centered * (1 - p.square())
        dr = (r * dp[None, :]) @ r
        analytic = dp @ r.diagonal().real + p @ dr.diagonal().real
        tol = 1e-6 if graph.dtype == torch.float32 else 1e-13
        torch.testing.assert_close(reverse_derivative, analytic, atol=tol, rtol=tol)
        eig = torch.linalg.eigvalsh(h)
        sv = torch.sqrt(1 + eig.square())
        condition = sv.max() / sv.min()
    return float(score.detach()), float(reverse_derivative), float(analytic), float(condition)


@torch.no_grad()
def score_pair(graphs, theta):
    return model(graphs[0], theta)[0].item() - model(graphs[1], theta)[0].item()


def run(seed):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(seed)
    conditions = []
    worst_fd = 0.0
    print("m theta gap64 gap32 derivative64 derivative32 analytic_error64 "
          "analytic_error32 max_drift32 max_drift64 fd_error64")
    for (m, theta), reference in REFERENCE.items():
        # The same permutations are used in both precisions.
        perms = [[torch.randperm(16 + m, generator=generator) for _ in range(8)]
                 for _ in range(2)]
        rows = {}
        for dtype in (torch.float32, torch.float64):
            graphs = [decorated_graph(kind, m, dtype) for kind in ("R", "S")]
            results = [evaluate(graph, theta) for graph in graphs]
            score, grad, analytic, condition = zip(*results)
            gap, derivative = score[0] - score[1], grad[0] - grad[1]
            err = max(abs(x - y) for x, y in zip(grad, analytic))
            drift = 0.0
            with torch.no_grad():
                for graph, original, permutations in zip(graphs, score, perms):
                    for order in permutations:
                        new_score = model(graph[order][:, order], theta)[0].item()
                        drift = max(drift, abs(new_score - original))
            rows[dtype] = gap, derivative, err, drift
            if dtype == torch.float64:
                step = 1e-4 * theta
                fd = (score_pair(graphs, theta + step) -
                      score_pair(graphs, theta - step)) / (2 * step)
                fd_error = abs(fd - derivative)
                if fd_error > 3e-11:
                    raise ArithmeticError(f"Finite difference disagrees at m={m}, theta={theta}")
                np.testing.assert_allclose([gap, derivative], reference, rtol=5e-4, atol=0)
                conditions.extend(condition)
                worst_fd = max(worst_fd, fd_error)
        a, b = rows[torch.float64], rows[torch.float32]
        print(m, theta, *(f"{v:.6g}" for v in
                          (a[0], b[0], a[1], b[1], a[2], b[2], b[3], a[3], fd_error)))
    interval = min(conditions), max(conditions)
    np.testing.assert_allclose(interval, (5.427, 6.916), rtol=0, atol=0.0005)
    print(f"First-pole condition range: {interval[0]:.6f} to {interval[1]:.6f}")
    print(f"Maximum FP64 centered-difference error: {worst_fd:.6g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutation-seed", type=int, default=1543)
    run(parser.parse_args().permutation_seed)
