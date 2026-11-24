"""Independent checks of modular arithmetic and complete normalized static profiles."""
import argparse
from fractions import Fraction as F
from math import lcm
from pathlib import Path
import json

import networkx as nx
import numpy as np

from normalized_brec import (COEFFICIENTS, GATE_DENOM, PRIMES, SOURCE_SHA, file_sha,
                             history_generator)


def fraction_rows(adjacency):
    n = len(adjacency)
    matrix = np.array([
        [F(int(v), max(int(sum(row)), 1)) for v in row]
        for row in adjacency
    ], dtype=object)
    current = matrix
    histories = []
    for depth in range(3):
        power = np.array([[F(i == j) for j in range(n)] for i in range(n)],dtype=object)
        columns = []
        for _ in range(24):
            power = power @ current
            columns.append(np.diag(power).copy())
        last = np.column_stack(columns)
        histories.append(last)
        yield np.column_stack(histories)
        if depth < 2:
            field = last @ COEFFICIENTS / GATE_DENOM
            current = (matrix + np.diag(field)) / F(2)


def modular_check():
    # A three-vertex path and a triangle with a pendant edge and an isolate.
    cases = [(3, [(0,1),(1,2)]), (5,[(0,1),(1,2),(0,2),(2,3)])]
    total = 0
    for n, edges in cases:
        a = np.zeros((n,n),dtype=np.int64)
        for i,j in edges:
            a[i,j] = a[j,i] = 1
        checks = {p:iter(history_generator(a,p)) for p in PRIMES}
        for fractions in fraction_rows(a):
            for p in PRIMES:
                values = np.array([
                    [(v.numerator % p) * pow(v.denominator,-1,p) % p for v in row]
                    for row in fractions
                ],dtype=np.int64)
                assert np.array_equal(values,next(checks[p]))
                total += values.size
    return total


def cleared_tuples(adjacency, scale, max_order):
    n = len(adjacency)
    neighbors = [np.flatnonzero(row) for row in adjacency]
    factors = [scale // len(nbr) if len(nbr) else 0 for nbr in neighbors]
    powers = np.eye(n, dtype=object)
    cols = [np.diag(powers).copy()]
    for _ in range(max_order):
        next_power = np.zeros((n,n),dtype=object)
        for i, nbr in enumerate(neighbors):
            if len(nbr):
                next_power[i] = powers[nbr].sum(axis=0) * factors[i]
        powers = next_power
        cols.append(np.diag(powers).copy())
    return sorted(map(tuple,np.column_stack(cols).tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    digest = file_sha(args.data)
    assert digest == SOURCE_SHA
    raw = np.load(args.data,mmap_mode="r",allow_pickle=False)
    assert raw.shape == (51200,)
    report = {"source_sha256":digest,"script_sha256":file_sha(Path(__file__)),
              "coordinates_checked_against_exact_fractions":modular_check(),"static_checks":[]}
    for key in (273,296,307):
        graphs = [nx.to_numpy_array(nx.from_graph6_bytes(bytes(c).strip()),dtype=np.int64)
                  for c in raw[64*key:64*key+2]]
        degrees = {int(d) for a in graphs for d in a.sum(1) if int(d)>0}
        scale = lcm(*degrees)
        cutoff = sum(map(len,graphs))-1
        left,right = (cleared_tuples(a,scale,cutoff) for a in graphs)
        report["static_checks"].append({
            "pair":key, "orders":[len(g) for g in graphs], "common_lcm":scale,
            "max_power":cutoff,"distinct_root_profiles":[len(set(left)),len(set(right))],
            "equal_integer_profile_multisets":left==right,
        })
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

