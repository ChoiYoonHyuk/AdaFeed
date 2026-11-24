"""Integer certificate for the direct-product homomorphism obstruction."""
import argparse
import json
from pathlib import Path


def cut_graph(sign):
    order = 45
    graph = [set() for _ in range(order)]

    def edge(i, j):
        assert i != j and j not in graph[i]
        graph[i].add(j)
        graph[j].add(i)

    rules = [(0, 1, sign), (2, 3, 1), (0, 2, 0), (0, 3, 0), (1, 2, 0),
             (1, 3, 0), (2, 4, 0), (4, 5, 0), (5, 2, 0)]
    for u, w, s in rules:
        for r in range(3):
            edge(3 * u + r, 3 * w + (r + s) % 3)
    leaf = 18
    for u, count in enumerate([0, 1, 0, 3, 0, 5]):
        for r in range(3):
            for _ in range(count):
                edge(3 * u + r, leaf)
                leaf += 1
    assert leaf == order and sum(map(len, graph)) == 108
    return graph


def product(graph, factor):
    m = len(factor)
    return [{j * m + b for j in neighbors for b in factor[a]}
            for neighbors in graph for a in range(m)]


def identity(n):
    return [[int(i == j) for j in range(n)] for i in range(n)]


def multiply(graph, diagonal, matrix):
    # The left matrix has the graph's unit edges and the supplied diagonal.
    rows = []
    for i, neighbors in enumerate(graph):
        row = [diagonal[i] * value for value in matrix[i]]
        for neighbor in neighbors:
            row = [a + b for a, b in zip(row, matrix[neighbor])]
        rows.append(row)
    return rows


def profile(graph):
    n = len(graph)
    powers = [identity(n)]
    for _ in range(4):
        powers.append(multiply(graph, [0] * n, powers[-1]))
    d = [powers[2][i][i] for i in range(n)]
    h = [sum(d[i] * d[j] * powers[k][i][j] * powers[4-k][j][i]
             for i in range(n) for j in range(n)) for k in range(3)]
    field = list(d)
    cube = identity(n)
    for _ in range(3):
        cube = multiply(graph, field, cube)
    trace6 = sum(x*x for row in cube for x in row)
    return {"order": n, "edges": sum(map(len, graph)) // 2,
            "hom": h, "quadratic_coefficient": 3 * (2*h[0] + 2*h[1] + h[2]),
            "signless_sixth_trace": trace6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path3 = [{1}, {0, 2}, {1}]
    graphs = {"path3": path3, "plus": cut_graph(1), "minus": cut_graph(-1)}
    graphs["plus_times_path3"] = product(graphs["plus"], path3)
    graphs["minus_times_path3"] = product(graphs["minus"], path3)
    results = {name: profile(graph) for name, graph in graphs.items()}
    assert results["path3"]["hom"] == [20, 16, 20]
    for name in ("plus", "minus"):
        assert results[name+"_times_path3"]["hom"] == [
            x*y for x, y in zip(results[name]["hom"], results["path3"]["hom"])]
    for suffix, expected in (("", 0), ("_times_path3", -144)):
        a = results["plus"+suffix]
        b = results["minus"+suffix]
        assert a["quadratic_coefficient"] - b["quadratic_coefficient"] == expected
    assert results["plus_times_path3"]["signless_sixth_trace"] == 64530216
    assert results["minus_times_path3"]["signless_sixth_trace"] == 64530360
    encoded = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

