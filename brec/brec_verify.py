"""Check BREC positives by residues and feedback-equivalence by exact relations."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


SOURCE_SHA = "6975d172d27aedcdf4eb2c747450ede7928c111caeb2689fbc67e6b62fbd5562"
P = 1_000_003


def rank(rows):
    """Intern full tuples jointly on both sides."""
    rows = np.ascontiguousarray(rows)
    key = rows.view(np.dtype((np.void,rows.dtype.itemsize*rows.shape[1]))).ravel()
    return np.unique(key, return_inverse=True)[1].ravel().astype(np.int64)


def aligned_multisets(x,y):
    return np.array_equal(np.sort(x.ravel()), np.sort(y.ravel()))


def diagonal_bank(a, field):
    n = a.shape[0]
    if n * (P-1)**2 >= np.iinfo(np.int64).max:
        raise ValueError("Potential overflow with input size")
    x = np.eye(n,dtype=np.int64)
    responses = []
    for _ in range(24):
        x = (a @ x + field[:,None]*x) % P
        responses.append(x.diagonal().copy())
    return np.stack(responses,axis=1)


def profile_mismatch(l,r):
    keys = rank(np.vstack((l,r)))
    return not aligned_multisets(*np.split(keys,2))


def finite_queries(graphs):
    weights = np.random.default_rng(1729).integers(1,P,24)
    sparse = [csr_matrix(a) for a in graphs]
    first = [diagonal_bank(a,np.zeros(a.shape[0],dtype=np.int64)) for a in sparse]
    static = profile_mismatch(*first)
    if static:
        return True,True
    full = []
    for a,initial in zip(sparse,first):
        field = initial@weights % P
        full.append(np.column_stack((initial,diagonal_bank(a,field))))
    return False,profile_mismatch(*full)


def pair_spectral_positive(graphs, max_degree):
    # Modular power-entry tuples are coarser than complete joint projector tuples.
    size = graphs[0].shape[0]
    powers = [np.eye(size,dtype=np.int64) for a in graphs]
    sparse = [csr_matrix(a) for a in graphs]
    pair = np.zeros(2*size*size,dtype=np.int64)
    for order in range(max_degree+1):
        if order:
            powers = [(a@h) % P for a,h in zip(sparse,powers)]
        pair = rank(np.column_stack((pair,np.concatenate([h.ravel() for h in powers]))))
    pair = pair.reshape(2,size,size)
    node = np.zeros((2,size),dtype=np.int64)
    while True:
        count = int(node.max())+1
        merged = np.sort(pair*(2*size+1)+node[:,None,:],axis=2)
        node = rank(np.column_stack((node.ravel(),merged.reshape(2*size,size)))).reshape(2,size)
        if not aligned_multisets(*node):
            return True
        if int(node.max())+1 == count:
            return False


def common_srg(graphs):
    def params(a):
        n = len(a)
        degrees = a.sum(1)
        d = int(degrees[0])
        if not (0 < d < n-1 and np.all(degrees==d)):
            return None
        twostep = a@a
        adjacent = np.unique(twostep[a==1])
        far = np.unique(twostep[(a==0)&~np.eye(n,dtype=bool)])
        if len(adjacent)==len(far)==1:
            return n,d,int(adjacent[0]),int(far[0])
    x,y = map(params,graphs)
    return x is not None and x==y


def verifies_intersections(colors):
    # For each (i,j) compare its entire multiset of (c_ik,c_kj) to the
    # representative of its relation. The shared reference table spans both graphs.
    k,n = int(colors.max())+1,colors.shape[1]
    representatives = np.full((k,n),-1,dtype=np.int64)
    for c in colors:
        for i in range(n):
            actual = np.sort(c[i,:,None]*k+c,axis=0).T
            ids, first = np.unique(c[i],return_index=True)
            missing = representatives[ids,0] == -1
            representatives[ids[missing]] = actual[first[missing]]
            if not np.array_equal(actual,representatives[c[i]]):
                return False
    return True


def common_coherent(graphs):
    # Random codes only propose a partition. A positive upper certificate always
    # receives a deterministic full-integer intersection-number check.
    n = len(graphs[0])
    c = np.array([1+a- np.eye(n,dtype=np.int64) for a in graphs])
    random = np.random.default_rng(581)
    while aligned_multisets(*c):
        k = int(c.max())+1
        f = random.integers(0,2**62,k,dtype=np.uint64)
        g = random.integers(0,2**62,k,dtype=np.uint64)
        hashed = f[c] @ g[c]
        # Include transpose to obtain a scheme closed under matrix transposition.
        table = np.column_stack((c.ravel().astype(np.uint64),c.transpose(0,2,1).ravel().astype(np.uint64),hashed.ravel()))
        following = rank(table).reshape(2,n,n)
        if int(following.max())+1 == k:
            return aligned_multisets(*following) and verifies_intersections(following)
        c = following
    return False


def check(task):
    pair_id, raw, spectral_degree = task
    graphs = [nx.to_numpy_array(nx.from_graph6_bytes(bytes(g).strip()),dtype=np.int64) for g in raw]
    adjacency,feedback = finite_queries(graphs)
    word_check = None
    if feedback:
        route = "static" if adjacency else "spatial"
    elif common_srg(graphs):
        route = "common_srg"
    elif common_coherent(graphs):
        route = "common_coherent"
    else:
        from word_span import verify_word_pair
        word_check = verify_word_pair(graphs)
        if not word_check["verified"]:
            raise ValueError(f"Colored-word certificate failed at pair {pair_id}")
        route = "common_colored_words"
    spectral = pair_spectral_positive(graphs,spectral_degree) if spectral_degree is not None else None
    if spectral is False:
        raise ValueError(f"EPWL positive certificate failed at input pair {pair_id}")
    row = {"id":pair_id, "feedback_route":route,"epwl_positive_degree":spectral_degree}
    if word_check is not None:
        row["word_check"] = word_check
    return row


def verify(input_file, workers):
    with input_file.open("rb") as f:
        digest = hashlib.file_digest(f,"sha256").hexdigest()
    if digest != SOURCE_SHA:
        raise ValueError("The graph6 array hash differs from the official input used in this study")
    data = np.load(input_file,mmap_mode="r",allow_pickle=False)
    if data.shape!=(51200,):
        raise ValueError("Unexpected number of graph records")
    expected_file = Path(__file__).with_name("pair_reference.json")
    reference = json.loads(expected_file.read_text(encoding="utf8"))
    degrees = reference["epwl_positive_degrees"]
    tasks = [(i,tuple(data[64*i:64*i+2]),degrees.get(str(i))) for i in range(400)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(check,tasks,chunksize=2))
    positives = [row["id"] for row in rows if row["feedback_route"] in ("static","spatial")]
    if positives != sorted(reference["feedback_positive_ids"]):
        raise ValueError("The modular feedback results do not match the archived pair IDs")
    counts = dict(sorted(Counter(row["feedback_route"] for row in rows).items()))
    if counts != {"static":216,"spatial":3,"common_srg":70,"common_coherent":60,
                  "common_colored_words":51} or len(degrees)!=236:
        raise ValueError(f"Counts disagree: {counts}")
    return {"data_sha256":digest, "pair_count":len(rows),"epwl_positive_ids_verified":len(degrees),
            "feedback_routes":counts, "decisions":rows}


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("input",type=Path)
    cli.add_argument("--workers",type=int,default=2)
    cli.add_argument("--output",type=Path)
    args = cli.parse_args()
    result = verify(args.input,args.workers)
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in result.items() if k!="decisions"},indent=2))
