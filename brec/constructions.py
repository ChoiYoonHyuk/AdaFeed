"""Exact finite witnesses and symbolic core minors, using only Python and NumPy."""
from collections import Counter
from itertools import combinations, permutations

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def adjacency(n, edges):
    # Python integers throughout avoid any dependence on machine integer overflow.
    a = np.zeros((n, n), dtype=object)
    for i, j in edges:
        require(i != j, "Loop in constructed graph")
        a[i, j] = a[j, i] = 1
    return a


def power(a, k):
    return np.linalg.matrix_power(a, k)


def base_graph(kind):
    edges = []
    for i, j in combinations(range(16), 2):
        a, b = divmod(i, 4)
        c, d = divmod(j, 4)
        move = ((c-a) % 4, (d-b) % 4)
        adjacent = (a == c or b == d) if kind == "R" else move in {
            (0,1),(1,0),(0,3),(3,0),(1,1),(3,3)
        }
        if adjacent:
            edges.append((i, j))
    return adjacency(16, edges)


def decorated(kind, count):
    base = base_graph(kind)
    vertices = len(base)+count
    a = np.zeros((vertices, vertices), dtype=object)
    a[:16, :16] = base
    a[0, 16:] = a[16:, 0] = 1
    q = np.zeros(vertices, dtype=object)
    q[:16] = base[0]
    return int(q @ power(a+np.diag(q), 5).diagonal())


def arrange(sequence):
    a = adjacency(68, [(i, (i+1)%4) for i in range(4)])
    for j, kind in enumerate(sequence):
        start = 4+16*j
        a[start:start+16, start:start+16] = base_graph(kind)
        a[start, j] = a[j, start] = 1
    d, s = a.sum(1), power(a, 4).diagonal()
    numerator = -(d-3)*(d-7)*(s-96)
    require(all(int(x)%3 == 0 for x in numerator), "Nonintegral selector")
    chi = numerator // 3
    require(set(chi) == {0,1} and sum(chi) == 24, "Selector failed")
    o = power(a+np.diag(chi),5).diagonal()
    require(Counter(o[chi==1]) == {897:12,895:12}, "Unexpected selected fifth returns")
    shift = chi * (o-896)
    h = a+np.diag(shift)
    return int((power(h,5).diagonal()*power(h,7).diagonal())[d==3].sum())


def voltage_edges(sign):
    return [(0,1,sign),(2,3,1),(0,2,0),(0,3,0),(1,2,0),(1,3,0),(2,4,0),(4,5,0),(5,2,0)]


def cut_graph(sign,k):
    next_vertex = 6*k
    edges = []
    for r in range(k):
        for u,v,s in voltage_edges(sign):
            edges.append((u*k+r,v*k+(r+s)%k))
        for u,count in enumerate([0,1,0,3,0,5]):
            for j in range(count):
                edges.append((u*k+r,next_vertex))
                next_vertex += 1
    require(next_vertex == 15*k, "Incorrect cover size")
    return adjacency(next_vertex,edges)


def minor(sign, vertices):
    # Each permutation produces a Laurent monomial in z times a squarefree
    # product of independent formal diagonal variables. Its key stores both.
    arc = {(u,v):s for u,v,s in voltage_edges(sign)}
    arc.update({(v,u):-s for u,v,s in voltage_edges(sign)})
    poly = Counter()
    for permutation in permutations(vertices):
        inversions = sum(permutation[i]>permutation[j]
                         for i,j in combinations(range(len(vertices)),2))
        coeff = (-1)**inversions
        exponent = 0
        diagonal = []
        for u,v in zip(vertices, permutation):
            if u == v:
                diagonal.append(u)
            elif (u,v) in arc:
                exponent += arc[u,v]
                coeff *= -1
            else:
                break
        else:
            poly[tuple(diagonal), exponent] += coeff
    return {k:v for k,v in poly.items() if v}


def integer_certificates():
    decorated_pairs = []
    for m in (1,2,5):
        scores = [decorated(kind,m) for kind in ("R","S")]
        require(scores == [5334+36*m+12,5334+36*m], "Decorated scores disagree")
        decorated_pairs.append(scores)
    cycle = [arrange(sequence) for sequence in ("RRSS","RSRS")]
    require(cycle[0]-cycle[1] == 288, "Third-query gap disagrees")
    graph_gaps = {}
    product_traces = []
    for k in (3,4,5,7):
        scores = []
        for a in (1,-1):
            g = cut_graph(a,k)
            h = g@g
            scores.append(int(g.sum(1) @ ((h*h) @ (h@g).diagonal())))
            if k == 3:
                x = np.kron(g,adjacency(3,[(0,1),(1,2)]))
                q = power(x+np.diag(x.sum(1)),3)
                product_traces.append(int((q*q).sum()))
        require(scores == [1180*k,1176*k], "Midpoint formula failed")
        graph_gaps[str(k)] = scores
    require(product_traces == [64530216,64530360], "Direct-product trace mismatch")
    checked = 0
    for t in range(7):
        for subset in combinations(range(6),t):
            require(minor(1,subset)==minor(-1,subset), f"Minor mismatch at {subset}")
            checked += 1
    require(checked==64, "Missing principal minor")
    return {"decorated_scores_m_1_2_5":decorated_pairs, "third_query_68_vertices":cycle,
            "midpoint_scores_by_cover_order":graph_gaps,"product_traces":product_traces,
            "principal_minors_verified":checked}


if __name__ == "__main__":
    import json
    print(json.dumps(integer_certificates(),indent=2))