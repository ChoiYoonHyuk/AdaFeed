"""Exact certificates for the effect of a common direct-product factor.

Place next to the released check_product.py and run python check_direct_factors.py.
All graph entries and matrix products are Python integers, with no random probes.
"""
from collections import Counter
from itertools import product as tuples
import argparse
import json
from pathlib import Path

from check_product import cut_graph, identity, multiply, product


def rotate_class(word):
    return min(word[i:] + word[:i] for i in range(len(word)))


def length_classes(length):
    return Counter(rotate_class("".join(word)) for word in tuples("AD", repeat=length))


def word_trace(graph, word):
    degrees = list(map(len, graph))
    zero = [0] * len(graph)
    mat = identity(len(graph))
    for letter in reversed(word):
        if letter == "A":
            mat = multiply(graph, zero, mat)
        else:
            mat = [[degree * v for v in row] for degree, row in zip(degrees, mat)]
    return sum(row[i] for i, row in enumerate(mat))


def direct_trace(graph, z):
    field = [z * len(nbrs) for nbrs in graph]
    cube = identity(len(graph))
    for _ in range(3):
        cube = multiply(graph, field, cube)
    return sum(v * v for row in cube for v in row)


def from_edges(n, edges):
    g = [set() for _ in range(n)]
    for i, j in edges:
        assert 0 <= i < j < n and j not in g[i]
        g[i].add(j)
        g[j].add(i)
    return g


def path(m):
    return from_edges(m, [(i, i+1) for i in range(m-1)])


def cycle(m):
    return from_edges(m, [(i, i+1) for i in range(m-1)] + [(0, m-1)])


def complete_bipartite(a, b):
    return from_edges(a+b, [(i, a+j) for i in range(a) for j in range(b)])


def subdivided_k4():
    base_edges = [(i, j) for i in range(4) for j in range(i+1, 4)]
    return from_edges(10, [(u, 4+k) for k, edge in enumerate(base_edges) for u in edge])


def predicted_gap(factor, z):
    return 36 * z * z * (word_trace(factor, "AAADAD") - word_trace(factor, "AADAAD"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plus, minus = cut_graph(1), cut_graph(-1)
    zeros_through_five = 0
    for length in range(1, 6):
        for word in length_classes(length):
            assert word_trace(plus, word) == word_trace(minus, word)
            zeros_through_five += 1
    words = {}
    for word, mult in sorted(length_classes(6).items()):
        a, b = word_trace(plus, word), word_trace(minus, word)
        words[word] = {"cyclic_orbit": mult, "plus": a, "minus": b, "delta": a-b}
    assert len(words) == 14 and sum(v["cyclic_orbit"] for v in words.values()) == 64
    assert {w: (v["delta"], v["cyclic_orbit"]) for w, v in words.items() if v["delta"]} == {
        "AAADAD": (6, 6), "AADAAD": (-12, 3)
    }
    path_gaps = {}
    cycle_gaps = {}
    for m in range(3, 21):
        path_gaps[m] = predicted_gap(path(m), 1)
        cycle_gaps[m] = predicted_gap(cycle(m), 1)
        assert path_gaps[m] == (-144 if m == 3 else -72)
        assert cycle_gaps[m] == 0
    for a in range(1, 6):
        for b in range(1, 6):
            assert predicted_gap(complete_bipartite(a, b), 1) == -36 * a*a * b*b * (a-b)**2
    sub = subdivided_k4()
    assert predicted_gap(sub, 1) == -18*(3-2)**2*word_trace(sub, "AAAA")
    unstructured = from_edges(5, [(0,1), (0,2), (1,2), (2,3), (3,4)])
    direct = {}
    for name, f, z in [
        ("P3", path(3), 1), ("P4", path(4), 1), ("P5_negative_shift", path(5), -2),
        ("C4", cycle(4), 1), ("K2_3", complete_bipartite(2,3), 1),
        ("triangle_tail", unstructured, 3), ("subdivided_K4", sub, 1)
    ]:
        a, b = (direct_trace(product(g, f), z) for g in (plus, minus))
        assert a-b == predicted_gap(f, z)
        direct[name] = {"plus": a, "minus": b, "z": z, "delta": a-b}
    assert direct["P3"]["plus"] == 64530216 and direct["P3"]["minus"] == 64530360
    summary = {"equal_necklaces_through_degree_five": zeros_through_five,
               "length_six_necklaces": words, "direct_product_traces": direct,
               "path_gaps_at_z1": path_gaps, "cycle_gaps_at_z1": cycle_gaps}
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf8") as f:
            f.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
