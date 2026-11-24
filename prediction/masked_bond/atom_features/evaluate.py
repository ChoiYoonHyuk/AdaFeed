"""Molecule-clustered paired bootstrap of atom average precision."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from sklearn.metrics import average_precision_score

def aps(score, labels, ids, cluster_weight):
    order=np.argsort(-score,kind='stable')
    ranked=score[order]
    last=np.flatnonzero(np.r_[ranked[:-1]!=ranked[1:],True])
    ends=torch.as_tensor(last,device='cuda')
    ids=torch.as_tensor(ids[order],device='cuda')
    ys=torch.as_tensor(labels[order],device='cuda',dtype=torch.float64)
    result=[]
    for chunk in cluster_weight.split(12):
        w=chunk[:,ids].to(torch.float64)
        pos=(w*ys).cumsum(1)[:,ends]
        counts=w.cumsum(1)[:,ends]
        delta=torch.diff(pos,dim=1,prepend=torch.zeros_like(pos[:,:1]))
        tmp=((pos/counts.clamp(min=1))*delta).sum(1)/pos[:,-1]
        result.extend(tmp.cpu().tolist())
    np.testing.assert_allclose(result[0],average_precision_score(labels,score),atol=1e-10)
    return np.asarray(result)

def main():
    z=np.load('server_test_scores.npz')
    y=z['target']; ids=z['graph']; groups,idx=np.unique(ids,return_inverse=True)
    n=len(groups); rng=np.random.default_rng(56029)
    weights=np.row_stack((np.ones(n),rng.multinomial(n,np.full(n,1/n),size=1000))).astype(np.int32)
    cluster_weight=torch.from_numpy(weights).cuda()
    full_pairs=[('typed_full','typed_local'),('with_neighbors_full','with_neighbors_local'),
                ('topology_full','topology_local'),('typed_full','longer_static'),('typed_local','longer_static')]
    summary={}
    for sub,mask in [('all',np.ones(len(y),bool)),('carbon_with_hetero',z['carbon'].astype(bool)&z['has_hetero'].astype(bool)),
                     ('carbon_without_hetero',z['carbon'].astype(bool)&~z['has_hetero'].astype(bool))]:
        pairs=full_pairs if sub=='all' else full_pairs[:1]
        per_arm={}; report={}
        for a in sorted(set(v for pair in pairs for v in pair)):
            ap=np.stack([aps(s[mask],y[mask],idx[mask],cluster_weight) for s in z[a]])
            per_arm[a]=ap
        for a,b in pairs:
            diff=per_arm[a]-per_arm[b]
            mean=diff.mean(0)
            report[a+'__'+b]={'difference':float(mean[0]),'percentile95':np.quantile(mean[1:],[.025,.975]).tolist(),
                              'all_seeds':diff[:,0].tolist(),'mean_full_ap':float(per_arm[a][:,0].mean()),
                              'mean_control_ap':float(per_arm[b][:,0].mean())}
        summary[sub]={'molecules':int(len(np.unique(idx[mask]))),'atoms':int(mask.sum()),'positives':int(y[mask].sum()),'contrasts':report}
        print('evaluated',sub,flush=True)
    payload=dict(bootstrap_seed=56029,repetitions=1000,summary=summary,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path('paired_results.json').write_text(json.dumps(payload,indent=2))

if __name__=='__main__': main()
