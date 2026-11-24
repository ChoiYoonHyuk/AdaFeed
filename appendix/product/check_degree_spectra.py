"""Integer checks for degree-weighted spectra under the direct product.

Run python -B check_degree_spectra.py in this directory.
"""

from check_product import cut_graph, identity, multiply, product
from check_direct_factors import direct_trace, from_edges, path


def characteristic(graph, edge_sign):
    """Coefficients in descending order for det(tI - (D + edge_sign*A))."""
    assert edge_sign in (-1, 1)
    n = len(graph)
    signed_degree = [edge_sign * len(row) for row in graph]
    powers = identity(n)
    traces = []
    coeff = [1]
    for k in range(1, n + 1):
        # multiply() uses (A + diag(signed_degree)). Multiply its trace by
        # edge_sign**k to get the kth trace of (D + edge_sign*A).
        powers = multiply(graph, signed_degree, powers)
        traces.append(edge_sign**k * sum(powers[i][i] for i in range(n)))
        # Newton identities applied to the integer power sums.
        numerator = -sum(coeff[k - j] * traces[j - 1] for j in range(1, k + 1))
        quotient, remainder = divmod(numerator, k)
        assert remainder == 0
        coeff.append(quotient)
    return coeff


def zero_multiplicity(coeff):
    return next(i for i, c in enumerate(reversed(coeff)) if c)


def main():
    factor = path(3)
    bases = [cut_graph(1), cut_graph(-1)]
    products = [product(g, factor) for g in bases]
    for sign, name, reference in ((1, "Q", 1668990), (-1, "L", 1567326)):
        assert characteristic(bases[0], sign) == characteristic(bases[1], sign)
        base_traces = [direct_trace(g, sign) for g in bases]
        assert base_traces == [reference, reference]
        # At sign = -1 the measured operator is -L. Its sixth power is L^6.
        values = [direct_trace(g, sign) for g in products]
        assert values == [64530216, 64530360]
        print(f"{name}: equal base characteristic polynomials; sixth traces")
        print(f"  original {base_traces}, products {values}, gap {values[0] - values[1]}")

    # Connected L-cospectral controls: a triangle and a square sharing one
    # vertex, and K(2,3) with a leaf incident to a vertex on the size-three side.
    triangle_square = from_edges(6, [(0,2),(0,3),(2,3),(0,4),(0,5),(1,4),(1,5)])
    bipartite_leaf = from_edges(6, [(0,2),(0,4),(0,5),(1,2),(1,4),(1,5),(2,3)])
    # Q-cospectral controls: the claw and the disjoint union of a triangle and K1.
    claw = from_edges(4, [(0,1),(0,2),(0,3)])
    triangle_isolate = from_edges(4, [(0,1),(0,2),(1,2)])
    for sign, name, graphs, expected in (
        (-1, "L", [triangle_square, bipartite_leaf], [1, 2]),
        (1, "Q", [claw, triangle_isolate], [2, 4]),
    ):
        coeff = characteristic(graphs[0], sign)
        assert coeff == characteristic(graphs[1], sign)
        nullities = [zero_multiplicity(characteristic(product(g, factor), sign)) for g in graphs]
        assert nullities == expected
        print(f"{name} control: base characteristic coefficients {coeff}; product nullities {nullities}")


if __name__ == "__main__":
    main()
