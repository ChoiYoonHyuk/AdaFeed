from pathlib import Path
import json,statistics,hashlib
root=Path(__file__).parent
data=json.loads((root/'learned-catalogue-results.json').read_text())
assert data['script_sha256']==hashlib.sha256((root/'learned_catalogue.py').read_bytes()).hexdigest()
rows=data['families']
assert len(rows)==6
seeds=data['seeds']
totals=[sum(x['roots'][j] for x in rows) for j in range(3)]
cores=[sum(x['cores'][j] for x in rows) for j in range(3)]
out=dict(roots=totals,cores=cores, oracle_constant_correct=sum(x['best_constant_on_test'] for x in rows),modes={})
for mode in rows[0]['modes']:
    q=[]
    gates=[]
    for i,seed in enumerate(seeds):
        fits=[r['modes'][mode][i] for r in rows]
        assert all(v['seed']==seed for v in fits)
        n=sum(v['n'] for v in fits);assert n==totals[2]
        gates.extend(v['gate_change'] for v in fits)
        q.append(dict(seed=seed,correct=sum(v['correct'] for v in fits),mae=sum(v['mae']*v['n'] for v in fits)/n))
    out['modes'][mode]=dict(per_seed=q,accuracy_mean=statistics.mean(x['correct']/n for x in q),mae_mean=statistics.mean(x['mae'] for x in q),min_gate_change=min(gates),max_gate_change=max(gates))
print(json.dumps(out,indent=2))
(root/'learned-catalogue-summary.json').write_text(json.dumps(out,indent=2)+'\n')