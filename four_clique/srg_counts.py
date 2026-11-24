"""Check the class-contrast formula on McKay's public strongly regular graph files.

Run this script on the experimental server. The graph6 files stay on that server.
"""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
from urllib.parse import urljoin
from urllib.request import Request, urlopen

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import networkx as nx
import numpy as np

INDEX = "https://users.cecs.anu.edu.au/~bdm/data/graphs.html"
FAMILIES = ((25,12,5,6,15), (26,10,3,4,10), (28,12,6,4,4),
            (29,14,6,7,41), (35,16,6,8,3854), (35,18,9,9,227))


class Catalogue(HTMLParser):
    def __init__(self):
        super().__init__()
        self.link = None
        self.result = {}

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.link = dict(attrs).get("href")

    def handle_endtag(self, tag):
        if tag == "a":
            self.link = None

    def handle_data(self, data):
        mat = re.search(r"SRG\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", data, re.I)
        if self.link and mat:
            self.result[tuple(map(int,mat.groups()))] = urljoin(INDEX,self.link)


def fetch(url):
    if not url.startswith("https://users.cecs.anu.edu.au/~bdm/data/"):
        raise ValueError("Unexpected catalogue URL: " + url)
    req = Request(url, headers={"User-Agent":"Mozilla/5.0 (scholarly graph-count verification)"})
    with urlopen(req, timeout=35) as res:
        return res.read()


def clique_counts(adj):
    """Enumerate each unordered K4 and credit its four possible roots."""
    n = len(adj)
    masks = [sum(1 << int(j) for j in np.flatnonzero(row)) for row in adj]
    counts = [0]*n

    def bits(bits_value):
        while bits_value:
            low = bits_value & -bits_value
            yield low.bit_length()-1
            bits_value ^= low

    for i in range(n):
        for j in bits(masks[i] >> (i+1) << (i+1)):
            common = masks[i] & masks[j]
            for k in bits(common >> (j+1) << (j+1)):
                for ell in bits((common & masks[k]) >> (k+1) << (k+1)):
                    for v in (i,j,k,ell):
                        counts[v] += 1
    return counts


def evaluate(adj, params, roots):
    """Read q from unshifted second/fourth moments; compute scores from operators."""
    n,d,lam,mu = params
    old2 = adj @ adj
    expected = mu*np.ones_like(adj) + (lam-mu)*adj + (d-mu)*np.eye(n,dtype=np.int64)
    assert adj.shape==(n,n) and np.array_equal(old2,expected)
    assert np.all(adj.sum(axis=1)==d) and np.all(np.diag(adj)==0)
    a4 = d*d + d*lam*lam + (n-d-1)*mu*mu
    counts = clique_counts(adj)
    rows=[]
    eye=np.eye(n+1,dtype=np.int64)
    for root in roots:
        mat=np.zeros((n+1,n+1),dtype=np.int64)
        mat[:n,:n]=adj
        mat[root,n]=mat[n,root]=1
        squared=mat@mat
        degrees=np.diag(squared)
        s=np.einsum("ij,ij->i",squared,squared)
        numerator=(degrees-1)*(d+1-degrees)*(s-a4)
        pot,res=np.divmod(numerator,d-1)
        assert np.all(res==0)
        assert np.array_equal(pot,mat[root]*(np.arange(n+1)<n))
        neighbors=np.flatnonzero(pot)
        def fifth(operator, square):
            # M is symmetric. (M^5)_ii is the inner product of rows i of M^2 and M^3.
            s2=square[neighbors]
            return int(np.sum(s2*(s2@operator),dtype=np.int64))
        full=mat+np.diag(pot)
        feedback=fifth(full,full@full)
        static=fifth(mat,squared)
        # Only roots with q_i=1 receive nonzero readout weight.
        local=fifth(mat+eye,squared+2*mat+eye)
        rows.append((counts[root],feedback,static,local))
    return rows


def family(path,params,expected,url):
    raw=path.read_bytes()
    rows=(gzip.decompress(raw) if raw[:2]==b"\x1f\x8b" else raw).splitlines()
    rows=[x.removeprefix(b">>graph6<<") for x in rows if x.strip()]
    rows=[x for x in rows if x]
    n,d,_,_=params
    assert len(rows)==expected and len(set(rows))==expected
    hist=Counter()
    static_scores=set()
    local_scores=set()
    ok=0
    for idx,line in enumerate(rows):
        G=nx.from_graph6_bytes(line)
        adj=nx.to_numpy_array(G,nodelist=list(range(n)),dtype=np.int64)
        observations=evaluate(adj,params,[0] if idx==0 else range(n))
        if idx==0:
            truth,F,_,_=observations[0]
            intercept=F-6*truth
            continue
        for truth,F,static,local in observations:
            ok+=int(F-intercept==6*truth)
            hist[truth]+=1
            static_scores.add(static)
            local_scores.add(local)
        if idx==1:
            perm=np.random.RandomState(2027).permutation(n)
            permuted=evaluate(adj[np.ix_(perm,perm)],params,range(n))
            assert permuted==[observations[int(i)] for i in perm]
    assert len(static_scores)==len(local_scores)==1
    return dict(parameters=params,n_catalog_cores=expected,n_test_cores=expected-1,
                url=url,sha256=sha256(raw).hexdigest(),size_bytes=len(raw),n_test_roots=sum(hist.values()),
                correct=ok,calibration_intercept=intercept,oracle_constant_correct=max(hist.values()),
                modal_targets=[t for t,ct in hist.items() if ct==max(hist.values())],
                target_histogram=dict(sorted(hist.items())),min_target=min(hist),max_target=max(hist),
                static_score=static_scores.pop(),root_local_score=local_scores.pop(),
                permuted_test_roots=n)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("workdir",type=Path)
    args=parser.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    mapping=Catalogue()
    mapping.feed(fetch(INDEX).decode("utf-8",errors="replace"))
    out={"index":INDEX,"families":[],"version":{"numpy":np.__version__,"networkx":nx.__version__}}
    for a in FAMILIES:
        params=a[:4]
        url=mapping.result[params]
        p=args.workdir/url.rsplit("/",1)[-1]
        if not p.exists():
            p.write_bytes(fetch(url))
        res=family(p,params,a[4],url)
        out["families"].append(res)
        print(params,res["correct"],res["n_test_roots"],res["oracle_constant_correct"],flush=True)
        (args.workdir/"aggregate.json").write_text(json.dumps(out,indent=2)+"\n")

if __name__=="__main__":
    main()
