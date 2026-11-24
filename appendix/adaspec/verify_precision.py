"""Independent 60-digit calculation for the released AdaSpec operator on the cut pair."""
from decimal import Decimal as D, getcontext
from pathlib import Path
import json
getcontext().prec = 60

def returns(sign, gate, edge_term):
    rules = [(0,1,sign),(2,3,1),(0,2,0),(0,3,0),(1,2,0),(1,3,0),(2,4,0),(4,5,0),(5,2,0)]
    arcs = [(3*u+r,3*v+(r+s)%3) for u,v,s in rules for r in range(3)]
    leaf=18
    for u, multiplicity in enumerate((0,1,0,3,0,5)):
        for r in range(3):
            for _ in range(multiplicity):
                arcs.append((3*u+r,leaf))
                leaf += 1
    deg=[0]*45
    for i,j in arcs:
        deg[i]+=1
        deg[j]+=1
    bias=[D(1)/45 if gate=='uniform' else D(d)/(1+d) for d in deg]
    scale=[1/(d+b).sqrt() for d,b in zip(deg,bias)]
    rows=[[(i,b*s*s+D('.1'))] for i,(b,s) in enumerate(zip(bias,scale))]
    for i,j in arcs:
        value=scale[i]*scale[j]+edge_term
        rows[i].append((j,value))
        rows[j].append((i,value))
    X=[[D(i==j) for j in range(45)] for i in range(45)]
    out=[]
    for _ in range(10):
        X=[[sum(w*X[k][j] for k,w in row) for j in range(45)] for row in rows]
        out.append([r[i] for i,r in enumerate(X)])
    return out

out={}
for gate in ('uniform','degree'):
    for edge_term in (D(0),D(1)/4500):
        x,y=[returns(sign,gate,edge_term) for sign in (1,-1)]
        gaps=[max(abs(a-b) for a,b in zip(ak,bk)) for ak,bk in zip(x,y)]
        core_degrees=(3,4,5,6)
        s=[1/(D(d)+(D(1)/45 if gate=='uniform' else D(d)/(1+d))).sqrt() for d in core_degrees]
        weight01=s[0]*s[1]+edge_term
        weight23=s[2]*s[3]+edge_term
        exact_form=24*edge_term*weight01*weight23*(s[0]-s[1])*(s[2]-s[3])
        key=gate+('_normalized' if edge_term==0 else '_all_terms')
        sorted_gaps=[max(abs(a-b) for a,b in zip(sorted(ak),sorted(bk))) for ak,bk in zip(x,y)]
        traces=[sum(ak)-sum(bk) for ak,bk in zip(x,y)]
        out[key]={'matched':[str(v) for v in gaps], 'sorted':[str(v) for v in sorted_gaps],
                  'trace_plus_minus':[str(v) for v in traces], 'four_cycle_formula':str(exact_form)}
        assert abs(exact_form-traces[3]) < D('1e-54')
        print(key, 'k4 matched',f'{gaps[3]:.10E}', 'sorted',f'{sorted_gaps[3]:.10E}', 'trace',f'{traces[3]:.10E}', 'k10 trace',f'{traces[9]:.10E}')
        print(key,' '.join(f'{e:.6E}' for e in gaps))
        if edge_term == 0: assert max(gaps)<D('1e-50')
        else: assert gaps[-1]>D('1e-7')
Path('precision.json').write_text(json.dumps(out,indent=2)+'\n')

