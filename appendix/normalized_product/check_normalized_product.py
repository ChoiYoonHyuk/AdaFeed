"""Exact certificate for the two-query normalized direct-product obstruction.

Run python -B check_normalized_product.py. Only the standard library is needed.
Read p = diag(S^2) at zero shift, then sum diag((S+diag(p))^6) at shift p.
All witness graphs have positive degrees. With P = D^{-1} A, diagonal similarity
gives S + diag(p) = D^{1/2} (P + diag(p)) D^{-1/2} and diag(S^2) = diag(P^2).
"""
import argparse
from collections import Counter
from fractions import Fraction
from itertools import permutations, product
import json
from math import lcm
from pathlib import Path


CORE_EDGES = ((0, 2, 0), (0, 3, 0), (1, 2, 0), (1, 3, 0), (2, 4, 0),
              (4, 5, 0), (5, 2, 0), (2, 3, 1))
LEAF_PARENTS = (1, 3, 3, 3, 5, 5, 5, 5, 5)


def graph_from_edges(size, edges):
    graph = [set() for _ in range(size)]
    for i, j in edges:
        if i == j or j in graph[i]:
            raise ValueError("Edge list must define a simple graph")
        graph[i].add(j)
        graph[j].add(i)
    return [tuple(sorted(row)) for row in graph]


def cut_graph(sign):
    edges = []
    for r in range(3):
        for u, v, shift in (*CORE_EDGES, (0, 1, sign)):
            edges.append((15 * r + u, 15 * ((r + shift) % 3) + v))
        for slot, parent in enumerate(LEAF_PARENTS, start=6):
            edges.append((15 * r + parent, 15 * r + slot))
    return graph_from_edges(45, edges)


def direct_product(base, factor):
    n = len(factor)
    return [tuple(j * n + b for j in base[i] for b in factor[a])
            for i in range(len(base)) for a in range(n)]


def second_returns(graph):
    return [sum(Fraction(1, len(row) * len(graph[j])) for j in row) for row in graph]


def direct_trace(graph):
    """Propagate six times on the actual graph, clearing all denominators first."""
    field = second_returns(graph)
    scale = lcm(*(len(row) for row in graph), *(x.denominator for x in field))
    stays = [int(scale * x) for x in field]
    edges = [scale // len(row) for row in graph]
    total = 0
    for root in range(len(graph)):
        state = [0] * len(graph)
        state[root] = 1
        for _ in range(6):
            updated = [state[i] * stays[i] for i in range(len(graph))]
            for i, value in enumerate(state):
                if value:
                    increment = value * edges[i]
                    for j in graph[i]:
                        updated[j] += increment
            state = updated
        total += state[root]
    return Fraction(total, scale ** 6)


def word_trace(graph, field, word):
    """Evaluate an ordered word using exact rational matrix-vector products."""
    inverses = [Fraction(1, len(row)) for row in graph]
    total = 0
    for root in range(len(graph)):
        state = [0] * len(graph)
        state[root] = 1
        for letter in reversed(word):
            if letter == "P":
                state = [inv * sum(state[j] for j in row) for inv, row in zip(inverses, graph)]
            else:
                state = [p * x for p, x in zip(field, state)]
        total += state[root]
    return total


def word_check(plus, minus, factor):
    """Here P and diag(p) both factor, so each word trace is multiplicative."""
    classes = Counter()
    for letters in product("PQ", repeat=6):
        word = "".join(letters)
        classes[min(word[i:] + word[:i] for i in range(6))] += 1
    graphs = (plus, minus, factor)
    fields = tuple(second_returns(g) for g in graphs)
    scores = [0, 0]
    differing = {}
    for word, multiplicity in sorted(classes.items()):
        a, b, c = (word_trace(g, f, word) for g, f in zip(graphs, fields))
        scores[0] += multiplicity * a * c
        scores[1] += multiplicity * b * c
        if a != b:
            differing[word] = {"multiplicity": multiplicity, "base_plus_minus": str(a - b),
                               "factor_trace": str(c)}
    return scores, differing, len(classes)


def core_minor(vertices, sign, degrees):
    """Expand a normalized Fourier-core principal minor over Q[x_0,...,x_5,q,q^-1]."""
    shifts = {}
    for i, j, shift in (*CORE_EDGES, (0, 1, sign)):
        shifts[i, j], shifts[j, i] = shift, -shift
    coefficients = Counter()
    for perm in permutations(vertices):
        diagonal_mask = voltage = 0
        inversions = sum(i > j for t, i in enumerate(perm) for j in perm[t + 1:])
        weight = Fraction((-1) ** inversions)
        for i, j in zip(vertices, perm):
            if i == j:
                diagonal_mask |= 1 << i
            elif (i, j) in shifts:
                weight *= Fraction(-1, degrees[i])
                voltage += shifts[i, j]
            else:
                break
        else:
            coefficients[diagonal_mask, voltage] += weight
    return {term: val for term, val in coefficients.items() if val}


def verify_base_transfer(plus, minus):
    """The formal pencil has arbitrary diagonal variables, including pendant corrections.

    Eliminating a selected pendant with diagonal x_l multiplies its determinant by
    x_l and subtracts 1/(d_u x_l) from the parent's diagonal if that parent is present.
    The factors and corrections match across signs for every selected pendant set.
    Thus equality of all six-core principal minors below gives equality for all
    fifteen-slot principal minors, including diagonal cofactors. The result is a
    polynomial identity even where this elimination temporarily divides by zero.
    """
    degrees = [len(row) for row in plus]
    assert degrees == list(map(len, minus))
    for mask in range(64):
        vertices = [i for i in range(6) if mask & (1 << i)]
        assert core_minor(vertices, 1, degrees) == core_minor(vertices, -1, degrees)
    return {"equal_normalized_laurent_principal_minors": 64,
            "core_degrees": degrees[:6], "base_orders": [len(plus), len(minus)]}


def calculate():
    plus, minus = cut_graph(1), cut_graph(-1)
    factor = graph_from_edges(3, [(0, 1), (1, 2)])
    transfer = verify_base_transfer(plus, minus)
    scores, differing, necklaces = word_check(plus, minus, factor)
    products = [direct_product(base, factor) for base in (plus, minus)]
    direct = [direct_trace(graph) for graph in products]
    assert direct == scores
    gap = direct[0] - direct[1]
    assert necklaces == 14
    assert gap == Fraction(-70073, 36288000)
    assert set(differing) == {"PPPQPQ", "PPQPPQ"}
    return {"query_1": "p = diag(S^2)", "query_2": "trace((S+diag(p))^6)",
            "base_transfer": transfer, "factor": "P3", "product_orders": list(map(len, products)),
            "direct_trace_plus": str(direct[0]), "direct_trace_minus": str(direct[1]),
            "signed_gap_plus_minus": str(gap), "factorized_word_traces_match_direct": True,
            "sixth_word_cyclic_classes": necklaces, "differing_base_traces": differing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the same JSON to a new file")
    args = parser.parse_args()
    result = json.dumps(calculate(), indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(result)
    print(result, end="")


if __name__ == "__main__":
    main()
