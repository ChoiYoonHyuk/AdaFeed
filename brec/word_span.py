"""Prove equality of rooted colored-word diagonals by an exact finite span."""
from collections import defaultdict, deque
from math import gcd
from functools import reduce

import numpy as np


def rank_rows(x):
    a = np.ascontiguousarray(x)
    keys = a.view(np.dtype((np.void,a.dtype.itemsize*a.shape[1]))).ravel()
    return np.unique(keys,return_inverse=True)[1].ravel().astype(np.int64)


def candidate_classes(graphs):
    from brec_verify import diagonal_bank, P
    from scipy.sparse import csr_matrix
    sparse = [csr_matrix(g) for g in graphs]
    n = len(graphs[0])
    colors = np.zeros((2,n),dtype=np.int64)
    rng = np.random.default_rng(20261122)
    columns = []
    for iteration in range(8):
        old_count = int(colors.max())+1
        weights = rng.integers(0,P,old_count)
        banks = [diagonal_bank(a,weights[c]) for a,c in zip(sparse,colors)]
        columns.append(np.vstack(banks))
        following = rank_rows(np.hstack(columns)).reshape(2,n)
        if not np.array_equal(np.sort(following[0]),np.sort(following[1])):
            raise ValueError("Unequal candidate class sizes")
        colors = following
        if iteration>=2 and int(colors.max())+1 == old_count:
            break
    return colors


def rooted_quotient(graph, partition, root):
    """Root-marked equitable cells; every letter preserves cell-constant vectors."""
    n = len(graph)
    marked = (np.arange(n)==root).astype(np.int64)
    cells = rank_rows(np.column_stack((partition,marked)))
    while True:
        k = int(cells.max())+1
        histogram = graph @ np.eye(k,dtype=np.int64)[cells]
        next_cells = rank_rows(np.column_stack((cells,histogram)))
        if int(next_cells.max())+1==k:
            break
        cells = next_cells
    _,representatives = np.unique(cells,return_index=True)
    quotient = histogram[representatives]
    cell_class = partition[representatives]
    r = int(cells[root])
    if (np.count_nonzero(cells==r)!=1 or not np.array_equal(quotient[cells],histogram)
            or not np.array_equal(cell_class[cells],partition)):
        raise ValueError("Invalid rooted equitable quotient")
    return quotient,cell_class,r


def normalize(v):
    div = reduce(gcd,v,0)
    return v if div<=1 else [x//div for x in v]


def insert(basis, candidate):
    """Fraction-free row elimination over Q; coordinates are arbitrary-precision integers."""
    v = list(map(int,candidate))
    for j, entry in sorted(basis.items()):
        if v[j]:
            d = gcd(v[j],entry[j])
            a,b = entry[j]//d,v[j]//d
            v = normalize([a*x-b*y for x,y in zip(v,entry)])
    pivot = next((j for j,x in enumerate(v) if x),None)
    if pivot is None:
        return None
    v = normalize(v)
    basis[pivot] = v
    return np.array(v,dtype=object)


def compare(left,right):
    a,ca,ra = left
    b,cb,rb = right
    if ca[ra]!=cb[rb]:
        raise ValueError("Different root colors")
    colors = np.r_[ca,cb]
    classes = np.unique(colors)
    where = {int(c):np.flatnonzero(colors==c) for c in classes}
    move = np.zeros((len(colors),len(colors)),dtype=np.int64)
    move[:len(ca),:len(ca)] = a
    move[len(ca):,len(ca):] = b
    transitions = defaultdict(list)
    for source,src in where.items():
        for target,dst in where.items():
            m = move[np.ix_(dst,src)]
            if m.any():
                transitions[source].append((target,m.astype(object)))
    root_color = int(ca[ra])
    ids = where[root_color]
    ia = int(np.flatnonzero(ids==ra)[0])
    ib = int(np.flatnonzero(ids==len(ca)+rb)[0])
    v = np.zeros(len(ids),dtype=object)
    v[ia] = v[ib] = 1
    bases = {int(c):{} for c in classes}
    queue = deque([(root_color,insert(bases[root_color],v))])
    while queue:
        source,vector = queue.popleft()
        if source==root_color and vector[ia]!=vector[ib]:
            return False,sum(len(v) for v in bases.values())
        for target,block in transitions[source]:
            independent = insert(bases[target],block@vector)
            if independent is not None:
                queue.append((target,independent))
    return True,sum(len(v) for v in bases.values())


def verify_word_pair(graphs):
    coloring = candidate_classes(graphs)
    classes = int(coloring.max())+1
    anchors = {}
    largest = 0
    comparisons = 0
    for graph,partition in zip(graphs,coloring):
        for root,cls in enumerate(partition):
            q = rooted_quotient(graph,partition,root)
            # These bytes contain the entire exact quotient and its root and
            # initial colors. Hash tables compare full byte strings on a collision.
            key = (q[0].tobytes(),q[1].tobytes(),q[2])
            if cls not in anchors:
                anchors[cls] = (q,{key})
                continue
            first,seen = anchors[cls]
            if key in seen:
                continue
            ok,dimension = compare(first,q)
            comparisons += 1
            largest = max(largest,dimension)
            if not ok:
                return {"verified":False,"classes":classes,"root":int(root),"class":int(cls)}
            seen.add(key)
    return {"verified":True,"classes":classes,"unequal_quotient_comparisons":comparisons,
            "max_paired_span_dimension":largest}
