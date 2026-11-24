"""Check the fourth-trace identity for the 45-vertex cut-transpose pair.

No input files or dependencies. The operator uses exact constants b=1/45,
the scalar shift 1/10, and a raw-adjacency coefficient c=1/4500 or 0.
"""
from fractions import Fraction
from decimal import Decimal, localcontext
from math import isqrt, prod

P = 19423
B = Fraction(1, 45)
ALPHA = Fraction(1, 10)
C = Fraction(1, 4500)
SIZE = 45


def factors(value):
    result = []
    for factor in range(2, isqrt(value) + 1):
        while value % factor == 0:
            result.append(factor)
            value //= factor
    if value != 1:
        result.append(value)
    return result


def adjacency(sign):
    rules = [(0, 1, sign), (2, 3, 1), (0, 2, 0), (0, 3, 0),
             (1, 2, 0), (1, 3, 0), (2, 4, 0), (4, 5, 0), (5, 2, 0)]
    matrix = [[0] * SIZE for _ in range(SIZE)]
    for u, v, shift in rules:
        for r in range(3):
            i, j = 3 * u + r, 3 * v + (r + shift) % 3
            matrix[i][j] = matrix[j][i] = 1
    leaf = 18
    for u, number in enumerate((0, 1, 0, 3, 0, 5)):
        for r in range(3):
            for _ in range(number):
                i = 3 * u + r
                matrix[i][leaf] = matrix[leaf][i] = 1
                leaf += 1
    assert leaf == SIZE and sum(map(sum, matrix)) == 108
    return matrix


def residue(value):
    return value.numerator * pow(value.denominator, -1, P) % P


def weights(degrees):
    # Let q=45d+1. A normalized edge has weight 45/sqrt(q_i q_j).
    # Every prime generator uses the same image in both graphs. The source
    # radicals are independent (their square classes are distinct primes).
    assert factors(P) == [P] and P % 4 == 3
    radicands = {f for d in degrees for f in factors(45 * d + 1)}
    root = {f: pow(f, (P + 1) // 4, P) for f in sorted(radicands)}
    assert all(r * r % P == f for f, r in root.items())
    print("images of positive prime radicals mod", P, root)
    return {d: pow(prod(root[f] for f in factors(45 * d + 1)), -1, P)
            for d in degrees}


def trace_four(matrix, t, c):
    degrees = [sum(row) for row in matrix]
    operator = [[edge * ((45 * t[degrees[i]] * t[degrees[j]] + c) % P)
                 for j, edge in enumerate(row)] for i, row in enumerate(matrix)]
    for i, d in enumerate(degrees):
        operator[i][i] = (residue(B / (d + B)) + residue(ALPHA)) % P
    # Symmetry yields tr(H^4) = sum_ij (H^2_ij)^2.
    return sum(sum(x * y for x, y in zip(row, col))**2
               for row in operator for col in operator) % P


def run():
    graphs = [adjacency(1), adjacency(-1)]
    t = weights({sum(row) for row in graphs[0]})
    for coef, expected in ((C, [13816, 4869]), (Fraction(0), [14652, 14652])):
        c = residue(coef)
        traces = [trace_four(matrix, t, c) for matrix in graphs]
        print("c =", coef, "fourth traces mod", P, traces)
        assert traces == expected
        # The exceptional 4-cycles use 01 and 23. There are three fibers,
        # four roots, and two orientations. Factoring their cross-edge
        # determinant leaves c*(x0-x1)*(x2-x3), x_u = 1/sqrt(d_u+b).
        w01 = (45 * t[3] * t[4] + c) % P
        w23 = (45 * t[5] * t[6] + c) % P
        gap = 24 * w01 * w23 * c * 45 * (t[3] - t[4]) * (t[5] - t[6])
        assert (traces[0] - traces[1] - gap) % P == 0
    with localcontext() as context:
        context.prec = 60
        b = Decimal(B.numerator) / Decimal(B.denominator)
        c = Decimal(C.numerator) / Decimal(C.denominator)
        x0, x1, x2, x3 = (1 / (Decimal(d) + b).sqrt() for d in (3, 4, 5, 6))
        gap = 24 * c * (x0*x1+c) * (x2*x3+c) * (x0-x1) * (x2-x3)
        print("real signed fourth-trace difference", gap)


if __name__ == "__main__":
    run()

