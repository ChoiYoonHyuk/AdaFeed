"""Build exact-distance inputs and check both walk identities on the OGB raw graphs."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from argparse import ArgumentParser
from collections import deque
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def bipartite_components(a):
    color = np.zeros(len(a), np.int8)
    bip = np.ones(len(a), bool)
    neighbors = [np.flatnonzero(row) for row in a]
    for i in range(len(a)):
        if color[i]:
            continue
        queue = deque([i])
        color[i] = 1
        visited = [i]
        ok = True
        while queue:
            j = queue.popleft()
            for v in neighbors[j]:
                if not color[v]:
                    color[v] = -color[j]
                    queue.append(v)
                    visited.append(v)
                elif color[v] == color[j]:
                    ok = False
        bip[visited] = ok
    return bip


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def obs(stats, mask, lower, first, second, e, u, bip):
    count = int(mask.sum())
    if not count:
        return
    stats['atoms'] += count
    stats['positive_exposure'] += int((e[mask] > 0).sum())
    stats['positive_detour'] += int((u[mask] > 0).sum())
    stats['bipartite_atoms'] += int(bip[mask].sum())
    stats['positive_bipartite_detour'] += int(((u[mask] > 0) & bip[mask]).sum())
    for key, v in [('low_max', lower), ('first_max', first), ('second_max', second)]:
        stats[key] = max(stats[key], float(np.max(np.abs(v[mask]))))


def main():
    pa = ArgumentParser(description=__doc__)
    for v in ('ogb', 'cache', 'out'):
        pa.add_argument(v, type=Path)
    args = pa.parse_args()
    z = np.load(args.cache)
    rd = lambda k, col=None: pd.read_csv(args.ogb/'raw'/(k+'.csv.gz'),
                                          header=None, usecols=col).to_numpy(dtype=np.int32)
    ns, ms = rd('num-node-list').ravel(), rd('num-edge-list').ravel()
    elem, edge = rd('node-feat', [0]).ravel()+1, rd('edge')
    labels = rd('edge-feat', [0]).ravel()
    assert len(ns) == 41127 and ns.sum() == len(elem) and ms.sum() == len(edge)
    node = np.r_[0, np.cumsum(ns)]
    begin = np.r_[0, np.cumsum(ms)]
    assert np.array_equal(z['graph'], np.repeat(np.arange(len(ns)), ns))
    graph_split = np.full(len(ns), -1)
    for k, part in enumerate(('train','valid','test')):
        ix = pd.read_csv(args.ogb/'split/scaffold'/(part+'.csv.gz'),header=None).values.ravel()
        assert (graph_split[ix] == -1).all()
        graph_split[ix] = k
    assert (graph_split >= 0).all() and np.array_equal(z['split'], np.repeat(graph_split, ns))
    assert np.array_equal(z['carbon'].astype(bool), elem == 6)
    assert np.array_equal(z['field'][:,0], (elem != 1) & (elem != 6))
    cached_y, cached_top, cached_neigh = z['y'], z['top'], z['neigh']
    group = np.full(120, 10)
    group[[1,6,7,8,9,15,16,17,35,53]] = np.arange(10)
    nall = len(elem)
    dist = np.full(nall, 4, np.int8)  # Four denotes >3 or no marked atom.
    full_ch = np.zeros((nall, 2), np.float64)
    local_ch = np.zeros_like(full_ch)
    recon = np.zeros_like(full_ch)
    counts = np.zeros((nall, 2, 11), np.int16)
    stats = {}
    for g, n in enumerate(ns):
        o = slice(node[g], node[g+1])
        v = slice(begin[g], begin[g+1])
        a = np.zeros((n,n))
        src,dst = edge[v].T
        a[src,dst] = a[dst,src] = 1
        assert not a.diagonal().any()
        target = np.zeros(n, dtype='int8')
        hit = labels[v] == 1
        target[src[hit]] = target[dst[hit]] = 1
        np.testing.assert_array_equal(target, cached_y[o])
        p = ((elem[o] != 1) & (elem[o] != 6)).astype(float)
        carbon = elem[o] == 6
        tags = np.eye(11)[group[elem[o]]]
        np.testing.assert_array_equal(a @ tags, cached_neigh[o])
        deg = a.sum(1)
        inv = 1 / np.sqrt(np.maximum(1, deg))
        s = inv[:, None] * a * inv[None,:]
        h = s + np.diag(p)
        sk, hk = np.eye(n), np.eye(n)
        sdiag, hdiag, smat = [sk.diagonal().copy()], [hk.diagonal().copy()], [sk]
        for k in range(1,9):
            sk, hk = sk @ s, hk @ h
            sdiag.append(sk.diagonal().copy())
            hdiag.append(hk.diagonal().copy())
            if k <= 4:
                smat.append(sk)
        sd, hd = np.stack(sdiag,1), np.stack(hdiag,1)
        np.testing.assert_allclose(sd[:,2:9], cached_top[o,2:], atol=6e-8, rtol=0)
        bip = bipartite_components(a)
        seen = np.eye(n, dtype=bool)
        frontier = seen
        dist[o][p > 0] = 0
        for d in range(1,4):
            frontier = (frontier @ a > 0) & ~seen
            seen |= frontier
            at_shell = (frontier @ p) > 0
            dist[o][(dist[o] == 4) & at_shell] = d
            e = (smat[d]**2) @ p
            u = 2*(smat[d] * smat[d+1]) @ p
            k = 2*d+1
            delta = hd-sd
            lower = np.abs(delta[:,:k]).max(1)
            first = delta[:,k] - e
            second = delta[:,k+1] - e - u
            for kind, select in [('exact', dist[o] == d), ('farther', dist[o] > d)]:
                mask = carbon & select
                for part in ('all', ('train','valid','test')[int(graph_split[g])]):
                    key = f'{part}/d{d}/{kind}'
                    rep = stats.setdefault(key,dict(atoms=0,positive_exposure=0,positive_detour=0,
                            bipartite_atoms=0,positive_bipartite_detour=0,
                            low_max=0.,first_max=0.,second_max=0.))
                    obs(rep, mask, lower, first, second, e, u, bip)
            if d >= 2:
                counts[o,d-2] = frontier @ tags
                full_ch[o,d-2] = hd[:,k]/(2**k)
                local_ch[o,d-2] = sd[:,k]/(2**k)
                recon[o,d-2] = (sd[:,k]+e)/(2**k)
        if (g+1) % 5000 == 0:
            print('prepared', g+1, flush=True)
    for key, rep in stats.items():
        assert rep['positive_bipartite_detour'] == 0
        assert max(rep[t] for t in ('low_max','first_max','second_max')) < 1e-11
        assert rep['positive_exposure'] == (rep['atoms'] if key.endswith('/exact') else 0)
    np.savez(args.out/'distance_features.npz',dist=dist,local=local_ch,spatial=full_ch,
             reconstruction=recon,shell_count=counts)
    report = dict(raw_identity=stats,official_molecules=len(ns),
                  data_hashes={p.name:digest(p) for p in list((args.ogb/'raw').glob('*.csv.gz'))
                      if p.name in ('num-node-list.csv.gz','num-edge-list.csv.gz','node-feat.csv.gz',
                                    'edge.csv.gz','edge-feat.csv.gz')},
                  scaffold_mapping_sha256=digest(args.ogb/'mapping/mol.csv.gz'),
                  splits_sha256={p.name:digest(p) for p in (args.ogb/'split/scaffold').glob('*.csv.gz')},
                  base_cache_sha256=digest(args.cache),script_sha256=digest(Path(__file__)))
    (args.out/'identity_aggregate.json').write_text(json.dumps(report,indent=2)+'\n')
    print('identity and input preparation finished',flush=True)


if __name__ == '__main__':
    main()
