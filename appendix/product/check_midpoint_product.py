"""Recompute two integer certificates directly from the paper's edge rules."""
import numpy as np

def witness(sign, k):
    edges = []
    next_leaf = 6 * k
    for j in range(k):
        for v, w, offset in [(0,1,sign),(2,3,1),(0,2,0),(0,3,0),(1,2,0),(1,3,0),(2,4,0),(4,5,0),(5,2,0)]:
            edges.append((v*k+j,w*k+(j+offset)%k))
        for v, count in enumerate([0,1,0,3,0,5]):
            for _ in range(count):
                edges.append((v*k+j,next_leaf))
                next_leaf += 1
    A = np.zeros((next_leaf,next_leaf), dtype=np.int64)
    for i,j in edges:
        A[i,j] = A[j,i] = 1
    assert next_leaf == 15*k and A.sum() == 36*k
    return A

path = np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=np.int64)
for k in (3,4,5,7):
    counts, traces = [], []
    for sign in (1,-1):
        A = witness(sign,k)
        B = A @ A
        q = np.diag(B @ A)
        counts.append(int(A.sum(1) @ (B**2 @ q)))
        if k == 3:
            M = np.kron(A,path)
            M += np.diag(M.sum(1))
            traces.append(int(((M @ M @ M)**2).sum()))
    assert counts == [1180*k,1176*k]
    if k == 3:
        assert traces == [64530216,64530360]
        print("45-vertex midpoint count:", counts, "135-vertex product sixth trace:", traces)
print("Verified midpoint gaps at k=3,4,5,7.")