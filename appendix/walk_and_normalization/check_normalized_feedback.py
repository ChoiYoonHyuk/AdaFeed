"""Check the normalization equivalence by rational arithmetic on small graphs.

Run with Python 3.9+ and no third-party dependencies:
    python check_normalized_feedback.py
The script prints aggregate JSON and downloads no data.
"""

from collections import Counter, deque
from fractions import Fraction as F
from itertools import combinations
import json


def identity(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def add_diagonal(matrix, values):
    result = [row[:] for row in matrix]
    for i, a in enumerate(values):
        result[i][i] += a
    return result


def dot_product(left, right):
    n = len(left)
    result = [[F(0)] * n for _ in range(n)]
    for i, row in enumerate(left):
        for k, coefficient in enumerate(row):
            if coefficient:
                for j, value in enumerate(right[k]):
                    if value:
                        result[i][j] += coefficient * value
    return result


def diagonal(matrix):
    return [row[i] for i, row in enumerate(matrix)]


def unit_projector(matrix):
    """Characteristic polynomial, removal of all unit roots, then polynomial evaluation."""
    n = len(matrix)
    coefficients = [F(1)]
    step = identity(n)
    for k in range(1, n + 1):
        step = dot_product(matrix, step)
        coefficient = -sum(diagonal(step)) / k
        coefficients.append(coefficient)
        step = add_diagonal(step, [coefficient] * n)
    if sum(coefficients):
        return [[F(0)] * n for _ in range(n)]
    while len(coefficients) > 1 and not sum(coefficients):
        quotient = [coefficients[0]]
        for a in coefficients[1:-1]:
            quotient.append(a + quotient[-1])
        coefficients = quotient
    normalizer = sum(coefficients)
    result = [[F(0)] * n for _ in range(n)]
    for a in coefficients:
        result = add_diagonal(dot_product(matrix, result), [a] * n)
    return [[value / normalizer for value in row] for row in result]


def inverse_diagonal(matrix):
    """Independent Gauss-Jordan evaluation of all diagonal inverse entries."""
    n = len(matrix)
    right = identity(n)
    augmented = [row[:] + other for row, other in zip(matrix, right)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if augmented[i][j] != 0)
        augmented[pivot], augmented[j] = augmented[j], augmented[pivot]
        scale = augmented[j][j]
        augmented[j] = [value / scale for value in augmented[j]]
        for i in range(n):
            coefficient = augmented[i][j]
            if i != j and coefficient:
                augmented[i] = [
                    x - coefficient * y for x, y in zip(augmented[i], augmented[j])
                ]
    return [augmented[i][n + i] for i in range(n)]


def inverse_shifted_target(operator, z):
    n = len(operator)
    negative = [[-value for value in row] for row in operator]
    return inverse_diagonal(add_diagonal(negative, [z] * n))


def check_reduction(base, target, t, u, poles):
    """Return diagonals of target and class-field source agree at both test poles."""
    n = len(base)
    negative = [[-value for value in row] for row in base]
    for z in poles:
        lhs = inverse_shifted_target(target, z)
        rhs_field = [z * t[i] - u[i] for i in range(n)]
        base_diagonal = inverse_diagonal(add_diagonal(negative, rhs_field))
        rhs = [x * y for x, y in zip(t, base_diagonal)]
        if lhs != rhs:
            raise AssertionError((z, lhs, rhs))


def components(adjacency):
    todo = set(range(len(adjacency)))
    result = []
    while todo:
        first = min(todo)
        todo.remove(first)
        queue = deque([first])
        visit = [first]
        while queue:
            node = queue.popleft()
            for other, edge in enumerate(adjacency[node]):
                if edge and other in todo:
                    visit.append(other)
                    queue.append(other)
                    todo.remove(other)
        result.append(visit)
    return result


def check_graph(n, edges):
    adjacency = [[F(0)] * n for _ in range(n)]
    for i, j in edges:
        adjacency[i][j] = adjacency[j][i] = F(1)
    degree = list(map(sum, adjacency))
    scale = [max(x, 1) for x in degree]
    walk = [[x / scale[i] for x in row] for i, row in enumerate(adjacency)]
    w = diagonal(unit_projector(walk))
    square = dot_product(walk, walk)
    cubic = dot_product(square, walk)
    shifted = add_diagonal(walk, w)
    shifted_cubic = dot_product(dot_product(shifted, shifted), shifted)
    r = [
        shifted_cubic[i][i] - cubic[i][i] - 2 * w[i] * square[i][i] - w[i] ** 3
        for i in range(n)
    ]
    recovered = [x / y if y else F(0) for x, y in zip(w, r)]
    if recovered != degree:
        raise AssertionError((edges, recovered, degree))
    for comp in components(adjacency):
        volume = sum(degree[i] for i in comp)
        for i in comp:
            assert w[i] == (degree[i] / volume if volume else F(0))
            assert r[i] == (1 / volume if volume else F(0))
    field = [(d - 1) / 3 for d in degree]
    poles = [F(n + 5), F(2 * n + 7)]
    # Fixed normalized adjacency with added history-derived diagonal.
    check_reduction(adjacency, add_diagonal(walk, field), scale,
                    [a * b for a, b in zip(scale, field)], poles)
    # Ordinary adjacency with added diagonal, simulated on the normalized base.
    inverse_degree = [F(1) / d for d in scale]
    check_reduction(walk, add_diagonal(adjacency, field), inverse_degree,
                    [a * b for a, b in zip(inverse_degree, field)], poles)
    # Degree-dependent positive self-loop weights with normalized loop and edge weights.
    strength = [(d + 2) / 3 for d in degree]
    mass = [d + b for d, b in zip(degree, strength)]
    operator = [
        [x / mass[i] for x in row] for i, row in enumerate(add_diagonal(adjacency, strength))
    ]
    check_reduction(adjacency, operator, mass, strength, poles)
    return {"degree": list(map(int, degree)), "unit_projector_diagonal": list(map(str, w)),
            "cubic_residual": list(map(str, r)), "recovered_degree": list(map(int, recovered))}


def six_vertex_static_tie():
    """Static normalized profiles can tie when unnormalized degree histories differ."""
    graph_edges = [
        [(0, 1), (2, 3), (3, 4), (4, 5), (5, 2)],
        [(0, 1), (1, 2), (3, 4), (4, 5)],
    ]
    profiles = []
    degree_sums = []
    for edges in graph_edges:
        n = 6
        a = [[F(0)] * n for _ in range(n)]
        for i, j in edges:
            a[i][j] = a[j][i] = F(1)
        degrees = [sum(row) for row in a]
        degree_sums.append(int(sum(degrees)))
        p = [[v / degrees[i] for v in row] for i, row in enumerate(a)]
        power = identity(n)
        coords = [diagonal(power)]
        for _ in range(11):
            power = dot_product(p, power)
            coords.append(diagonal(power))
        profiles.append(sorted(zip(*coords)))
        assert dot_product(dot_product(p,p),p) == p
    if profiles[0] != profiles[1] or degree_sums != [10, 8]:
        raise AssertionError("Six-vertex example failed")
    return {"graphs": ["K2 + C4", "P3 + P3"], "ordinary_two_step_sums": degree_sums,
            "normalized_even_profile_multiplicities": {"1": 2, "1/2": 4},
            "normalized_odd_profile": "0", "verified_P_cubed_equals_P": True}



if __name__ == "__main__":
    orders = Counter()
    roots = 0
    for order in range(1, 6):
        candidates = list(combinations(range(order), 2))
        for mask in range(2 ** len(candidates)):
            edges = [e for k, e in enumerate(candidates) if (mask >> k) & 1]
            check_graph(order, edges)
            orders[order] += 1
            roots += order
    mixed = [(0, 1), (1, 2), (3, 4), (3, 5), (4, 5), (6, 7)]
    record = check_graph(9, mixed)
    graphs = sum(orders.values()) + 1
    print(json.dumps({
        "arithmetic": "exact rational",
        "static_normalization_example": six_vertex_static_tie(),
        "exhaustive_labeled_graphs_by_order": orders,
        "additional_nine_vertex_example": record,
        "graphs": graphs,
        "root_degree_recoveries": roots + 9,
        "diagonal_resolvent_equalities": (roots + 9) * 3 * 2,
        "poles_per_target": 2,
        "targets_per_graph": 3,
    }, indent=2))
