"""Audit the graph-level union of stable 1-WL and complete adjacency returns on BREC.

Run on the compute host: python wl_bench.py brec_v3.npy result.json
The static and feedback upper bounds used for matching are the 216 and 219
separations already certified in the paper. Unequal modular moments establish
lower witnesses; equality of residues is never used to establish equivalence.
"""
import argparse
from collections import Counter
from itertools import combinations
import hashlib
import json
from pathlib import Path
import random

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, diags


EXPECTED_SHA = "6975d172d27aedcdf4eb2c747450ede7928c111caeb2689fbc67e6b62fbd5562"
PRIME = 1000003
CFI_EXTRA = {273, 296, 307}
FAMILIES = (
    (0, 60, "Basic"),
    (60, 160, "Regular"),
    (160, 260, "Extension"),
    (260, 360, "CFI"),
    (360, 380, "Four-vertex"),
    (380, 400, "Distance-regular"),
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def adjacency(graph):
    return [tuple(sorted(graph[i])) for i in range(len(graph))]


def permute(adj, rng):
    mapping = list(range(len(adj)))
    rng.shuffle(mapping)
    out = [None] * len(adj)
    for v, neighbors in enumerate(adj):
        out[mapping[v]] = tuple(sorted(mapping[w] for w in neighbors))
    return out


def refine(a, b, seed=None):
    colors = [[0] * len(a), [0] * len(b)] if seed is None else seed
    classes = len(set(colors[0]) | set(colors[1]))
    class_trace = [classes]
    first_difference = 0 if Counter(colors[0])!=Counter(colors[1]) else None
    for step in range(1, len(a) + len(b) + 1):
        signatures = [
            [(col[i], tuple(sorted(col[j] for j in neighbors)))
             for i, neighbors in enumerate(adj)]
            for col, adj in zip(colors, (a,b))
        ]
        unique = sorted(set(signatures[0]) | set(signatures[1]))
        palette = {signature: idx for idx, signature in enumerate(unique)}
        colors = [[palette[key] for key in keys] for keys in signatures]
        if first_difference is None and Counter(colors[0])!=Counter(colors[1]):
            first_difference=step
        class_trace.append(len(unique))
        if len(unique) == classes:
            return colors, class_trace, first_difference
        classes = len(unique)
    raise RuntimeError("color refinement failed to stabilize")


def wl(a,b):
    colors, _, _ = refine(a,b)
    return Counter(colors[0]) != Counter(colors[1])


def integer_returns(adj):
    n=len(adj)
    x=np.eye(n,dtype=object)
    profiles=[[] for _ in range(n)]
    for k in range(2*n):
        for seq,value in zip(profiles,x.diagonal()):
            seq.append(int(value))
        if k+1 < 2*n:
            nxt=np.empty_like(x)
            for i,neighbors in enumerate(adj):
                nxt[i]=x[list(neighbors)].sum(axis=0)
            x=nxt
    return [tuple(seq) for seq in profiles]


def stronger_controls(a,b):
    # Up to 2n-1 suffices over two graphs of common order n.
    pa,pb=integer_returns(a),integer_returns(b)
    lookup={key:i for i,key in enumerate(sorted(set(pa)|set(pb)))}
    ea=[[lookup[key] for key in hist] for hist in (pa,pb)]
    unmarked, trace0, first0=refine(a,b)
    pair=[Counter(zip(colors,spectral)) for colors,spectral in zip(unmarked,ea)]
    initialized,trace1,first1=refine(a,b,ea)
    return {"n":len(a), "through_power":2*len(a)-1,
            "static": Counter(ea[0])!=Counter(ea[1]),
            "wl_first_separating_round": first0, "ea_initialized_wl_first_separating_round": first1,
            "nodewise_wl_and_ea":pair[0]!=pair[1],
            "ea_initialized_wl":Counter(initialized[0])!=Counter(initialized[1]),
            "wl_class_count_by_round_including_first_stationary_check":trace0,
            "ea_initialized_wl_class_count_by_round_including_first_stationary_check":trace1}

def sparse_adj(adj):
    edges = [(v,j) for v,neighbors in enumerate(adj) for j in neighbors]
    row, col = zip(*edges) if edges else ([],[])
    return csr_matrix((np.ones(len(row),dtype=np.int64), (row,col)), shape=(len(adj),len(adj)))


def profile_pair(a,b,degree,initial=None,early=True):
    # Multiplication below is safe in signed int64: pre-reduce each entry.
    matrices = (a,b)
    powers = [np.eye(a.shape[0],dtype=np.int64) for a in matrices]
    histories = [[() for _ in range(m.shape[0])] for m in matrices] if initial is None else initial
    for k in range(1,degree+1):
        powers = [(m @ x) % PRIME for m,x in zip(matrices,powers)]
        histories = [[old+(int(v),) for old,v in zip(hist,x.diagonal())] for hist,x in zip(histories,powers)]
        if early and Counter(histories[0]) != Counter(histories[1]):
            return k, histories
    return None, histories


def adaptive_witness(a,b,history):
    keys = sorted(set(history[0]) | set(history[1]))
    palette = {key:i+1 for i,key in enumerate(keys)}
    codes = [np.array([palette[v] for v in h],dtype=np.int64) for h in history]
    for r in (1,2,3):
        scale = pow(len(keys)+1,r,PRIME)
        h = [scale*m+diags((p**r) % PRIME, dtype=np.int64, format="csr") for m,p in zip((a,b),codes)]
        k,_ = profile_pair(*h,min(2*a.shape[0]-1,64),initial=history)
        if k is not None:
            return {"field_exponent":r, "shifted_degree":k}
    raise AssertionError("adaptive lower witness did not separate a previously certified CFI pair")


def decorate(kind,leaves):
    graph=nx.Graph()
    graph.add_nodes_from(range(16+leaves))
    for i in range(16):
        x,y=divmod(i,4)
        for j in range(i+1,16):
            u,v=divmod(j,4)
            diff=((u-x)%4,(v-y)%4)
            if (kind=="rook" and (u==x or v==y)) or (kind=="shrikhande" and diff in {(0,1),(0,3),(1,0),(3,0),(1,1),(3,3)}):
                graph.add_edge(i,j)
    graph.add_edges_from((0,k) for k in range(16,16+leaves))
    return adjacency(graph)


def exact_decorated(adj):
    a=sparse_adj(adj).toarray().astype(object)
    x=np.eye(len(adj),dtype=object)
    vectors=[]
    for k in range(2*len(adj)):
        vectors.append(x.diagonal().copy())
        x=a@x
    ea=Counter(tuple(row) for row in zip(*vectors))
    q=np.array([(d==6 and t>96) for d,t in zip(vectors[2],vectors[4])],dtype=int)
    assert q.sum()==6
    h=a+np.diag(q)
    return ea,int(q@np.linalg.matrix_power(h,5).diagonal())



def single_class_witness(data):
    graphs=[adjacency(nx.from_graph6_bytes(bytes(line).strip())) for line in data[64*296:64*296+2]]
    returns=[integer_returns(g) for g in graphs]
    profile_set=sorted(set(returns[0]) | set(returns[1]))
    width=next(k for k in range(1,len(profile_set[0])+1) if len({r[:k] for r in profile_set})==len(profile_set))
    lookup={key:i for i,key in enumerate(profile_set)}
    labels=[np.array([lookup[r] for r in rows]) for rows in returns]
    exact=[]
    for selected in range(len(lookup)):
        powers=[np.eye(len(g),dtype=np.int64) for g in graphs]
        matrices=[2*sparse_adj(g)+diags((colors==selected).astype(np.int64),format="csr",dtype=np.int64) for g,colors in zip(graphs,labels)]
        assert all(m.dtype==np.int64 for m in matrices)
        assert all((2*max(map(len,g))+1)**12<2**63 for g in graphs)
        for k in range(1,13):
            powers=[h@x for h,x in zip(matrices,powers)]
            diagonals=[x.diagonal() for x in powers]
            for root in range(len(lookup)):
                values=[d[colors==root].tolist() for d,colors in zip(diagonals,labels)]
                hist=[Counter(v) for v in values]
                if hist[0]!=hist[1]:
                    exact.append((k,selected,root,hist))
            if any(row[1]==selected for row in exact):
                break
    if not exact:
        return {"mark_classes_tested":len(lookup),"max_degree":12,"separated":False}
    k,marked,root,hist=min(exact,key=lambda row:row[:3])
    def signature(cls):
        r=profile_set[cls]
        cut=next(k for k in range(1,width+1) if sum(v[:k]==r[:k] for v in profile_set)==1)
        return list(r[:cut])
    def identifying_moments(cls):
        for size in range(1,width):
            for idx in combinations(range(1,width),size):
                vals=tuple(profile_set[cls][j] for j in idx)
                if sum(tuple(row[j] for j in idx)==vals for row in profile_set)==1:
                    return dict(zip(idx,vals))
        raise AssertionError("failed to identify class")
    return {"degree":k,"EA_class_count":len(lookup),"marked_class_rank_0based":marked,
            "marked_sparse_static_moments":identifying_moments(marked),
            "readout_sparse_static_moments":identifying_moments(root),
            "readout_class_rank_0based":root,"marked_return_signature_from_power0":signature(marked),
            "readout_return_signature_from_power0":signature(root),"scale":2,"mark_weight_in_query":"1/2",
            "readout_distribution_integers_for_scaled_operator":[dict(sorted(x.items())) for x in hist],
            "aligned_class_codes_from_static_powers_through":width-1}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    actual=sha(args.release)
    assert actual==EXPECTED_SHA
    data=np.load(args.release,mmap_mode="r",allow_pickle=False)
    assert data.shape==(51200,)
    fields={"wl":set(),"static":set(),"feedback":set()}
    adaptive={}
    hard_controls={}
    certified_control_count=0
    rng=random.Random(20261123)
    for index in range(400):
        a,b=[adjacency(nx.from_graph6_bytes(bytes(line).strip())) for line in data[64*index:64*index+2]]
        assert len(a)==len(b)
        wr=wl(a,b)
        ar,br=permute(a,rng),permute(b,rng)
        assert not wl(a,ar) and not wl(b,br)
        assert wr==wl(ar,br)
        certified_control_count+=2
        if wr:
            fields["wl"].add(index)
        ga,gb=sparse_adj(a),sparse_adj(b)
        degree,history=profile_pair(ga,gb,min(48,2*len(a)-1))
        if degree is not None:
            fields["static"].add(index)
        if index in CFI_EXTRA:
            assert degree is None
            adaptive[index]=adaptive_witness(ga,gb,history)
            hard_controls[index]=stronger_controls(a,b)
            assert hard_controls[index]==stronger_controls(ar,br) and not hard_controls[index]["static"]
            print("exact seeded controls",index,hard_controls[index],flush=True)
    assert len(fields["static"])==216
    fields["feedback"]=fields["static"] | set(adaptive)
    assert len(fields["feedback"])==219
    fields["joint"]=fields["static"]|fields["wl"]
    # Every static positive is positive for both stronger refinements. The
    # certified 181 feedback ties imply ties for these two weaker controls.
    for name in ("nodewise_wl_and_ea","ea_initialized_wl"):
        fields[name]=fields["static"] | {idx for idx,row in hard_controls.items() if row[name]}
    def counts(ids):
        f={name:decisions&ids for name,decisions in fields.items()}
        return {"pairs":len(ids),**{name:len(value) for name,value in f.items()},
                "wl_only_over_static":len(f["wl"]-f["static"]),
                "feedback_only_over_joint":len(f["feedback"]-f["joint"]),
                "joint_only_over_feedback":len(f["joint"]-f["feedback"])}
    result={"release_sha256":actual,"range":[0,399],"index_rule":"64*p, 64*p+1",
            "permutation_seed":20261123,"isomorphic_controls":certified_control_count,
            "independent_pair_relabelings":400,"relabel_failures":0,
            "stronger_control_CFI":hard_controls,"modular_prime":PRIME,"static_power_limit":48,"CFI_feedback_additions":adaptive,
            "all":counts(set(range(400))),
            "families":{name:counts(set(range(lo,hi))) for lo,hi,name in FAMILIES}}
    dec={}
    for m in (1,2,5):
        a,b=decorate("rook",m),decorate("shrikhande",m)
        ea,fa=exact_decorated(a)
        eb,fb=exact_decorated(b)
        ar,br=permute(a,rng),permute(b,rng)
        assert not wl(a,b) and ea==eb and not wl(ar,br) and not wl(a,ar) and not wl(b,br)
        assert exact_decorated(ar)==(ea,fa) and exact_decorated(br)==(eb,fb)
        assert (fa,fb)==(5346+36*m,5334+36*m)
        stronger=stronger_controls(a,b)
        assert stronger==stronger_controls(ar,br)
        dec[m]={"stronger_controls":stronger,"vertices":len(a),"wl":False,"static":False,"joint":False,"F1":[fa,fb]}
    result["decorations"]=dec
    result["single_class_CFI296"]=single_class_witness(data)
    print("single class 296",result["single_class_CFI296"],flush=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result["all"]),json.dumps(adaptive),json.dumps(dec),flush=True)
if __name__=="__main__":
    main()
