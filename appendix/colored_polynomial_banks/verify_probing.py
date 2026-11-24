"""Pure-standard-library verification of probing algebra, aggregate numbers and provenance."""
import argparse,hashlib,json,statistics
from collections import deque
from fractions import Fraction as F
from pathlib import Path


def mm(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]


def exact_toy(edges,n,k):
    adj=[set() for _ in range(n)]
    for u,v in edges:adj[u].add(v);adj[v].add(u)
    colors=[]
    for r in range(n):
        visited={r};front={r}
        for _ in range(k):
            front={j for i in front for j in adj[i]}-visited
            visited.update(front)
        taken={colors[j] for j in visited if j<r}
        c=0
        while c in taken:c+=1
        colors.append(c)
    P=[[F(int(j in adj[i]),len(adj[i]) or 1) for j in range(n)] for i in range(n)]
    # P and S are diagonally similar. Adding the same diagonal preserves all rooted returns.
    columns=[[F(int(i==j)) for j in range(n)] for i in range(n)]
    probe=[[F(int(colors[i]==j)) for j in range(max(colors)+1)] for i in range(n)]
    def check(bank):
        direct,group=columns,probe
        history=[]
        for _ in range(k+1):
            truth=[direct[i][i] for i in range(n)]
            assert truth==[group[i][colors[i]] for i in range(n)]
            history.append(truth)
            direct,group=mm(bank,direct),mm(bank,group)
        return history
    p=[2*x-1 for x in check(P)[2]]
    follow=[[(P[i][j]+(p[i] if i==j else 0))/2 for j in range(n)] for i in range(n)]
    check(follow)
    return max(colors)+1


def verify():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('directory',type=Path,nargs='?',default=Path(__file__).parent)
    args=cli.parse_args()
    n=14;g=[(i,i+1) for i in range(n-1)]+[(5,11),(1,8)]
    for k in (2,4):assert exact_toy(g,n,k)<n
    root=args.directory;full=json.loads((root/'aggregate.json').read_text())
    assert full['mode']=='full' and len(full['cases'])==6
    assert {(c['n'],c['k']) for c in full['cases']}=={(n,k) for n in (4096,16384,65536) for k in (8,24)}
    assert full['archive_sha256']=='450b8733635d887466a2b96b26411f6e62caf7006f8a264f59cf9b5d75cdf549'
    expected={}
    def inspect(r):
        for fname,field in [('protocol.md','protocol_sha256'),('probe.py','source_sha256'),('color.cpp','cpp_sha256')]:
            assert hashlib.sha256((root/fname).read_bytes()).hexdigest()==r[field]
        assert 'H100' in r['gpu'] and r['fp']=='FP64' and r['width']==64 and r['cap_seconds']==120
        for c in r['cases']:
            n,k=c['n'],c['k'];v=c['verification'];assert v['crosschecked_sample_roots']==128
            assert max([v['maximum_field_error']]+v['sampled_errors']+(v['full_bank_errors'] or []))<1e-11
            for key,s in c['methods'].items():
                assert 1<=s['columns']<=n and s['blocks_per_bank']==(s['columns']+63)//64
                assert s['cold_total_seconds']>=s['cold_gpu']['seconds']>0
                assert s['cpu_common_graph_seconds']>0 and s['gpu_staging_seconds']>0
                assert s['graph']['full_edges']==1541898 and s['graph']['full_nodes']==1088092
                if key=='color':
                    assert s['ordered_pairs_checked_in_balls']>0 and s['cpu_coloring_seconds']>0 and s['verify_coloring_seconds']>0
                else:
                    assert s['columns']==n and s['cpu_coloring_seconds']==0
                rows=s['resident_trials']
                if any(rr['censored'] for rr in rows) or s['cold_gpu']['censored']:
                    assert s['resident_median_seconds'] is None
                else:
                    assert len(rows)==3 and statistics.median(rr['seconds'] for rr in rows)==s['resident_median_seconds']
                for rr in [s['cold_gpu'],*rows]:
                    assert rr['completed_blocks']<=rr['planned_blocks']==2*s['blocks_per_bank']
                    assert rr['peak_allocated_MiB']>=rr['resident_input_MiB'] and rr['peak_additional_MiB']>=0
                    if not rr['censored']:assert rr['completed_blocks']==rr['planned_blocks']
            col=c['methods']['color']
            key=n,k
            digest=(col['color_sha256'],col['graph']['normalized_csr_sha256'],col['graph']['prefixes_sha256'])
            if key in expected:assert digest==expected[key]
            expected[key]=digest
            if 'identity' in c['methods'] and not c['methods']['identity']['cold_gpu']['censored']:
                assert len(v['full_bank_errors'])==2
    inspect(full)
    for slot in ('0905','0942'):
        f=root/('repeat_'+slot+'.json')
        if f.exists():
            x=json.loads(f.read_text());assert x['mode']==slot
            if not x.get('unavailable'):
                assert [(c['n'],c['k']) for c in x['cases']]==[(65536,24)]
                inspect(x)
    print('Verified exact two-bank rational examples, numeric checks, graph/color hashes, six scheduled comparisons and all available repeat records.')


if __name__=='__main__':verify()
