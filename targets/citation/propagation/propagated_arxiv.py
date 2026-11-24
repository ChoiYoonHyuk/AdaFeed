"""Matched subject classification with shared one-, two- and three-hop propagation."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from argparse import ArgumentParser
from pathlib import Path
import sys
import json
import numpy as np
import torch
from torch import nn
from scipy import sparse
from scipy.stats import t


SEEDS = (103,211,307,401,503)
ARMS = ('Common','Root-local','Spatial')
DATA_HASH = '49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276'
SPLIT_HASH = 'df02a4bbdb3fdbb9e86816a3bcd6c3ce9d54e6fc75f7b1410a24e4a816188b09'


def prepare(archive,helpers):
    digest,load_graph,normalized_means,check_cubic,signed_rows = helpers
    assert digest(archive) == DATA_HASH
    data = load_graph(archive)
    ids = [data[k].ravel() for k in ('train','valid','test')]
    assert signed_rows(ids) == SPLIT_HASH and [len(part) for part in ids] == [90941,29799,48603]
    x,y,edge = data['x'].astype(np.float64),data['y'].ravel(),data['edge']
    assert x.shape == (169343,128) and edge.shape == (1166243,2)
    n = len(y)
    adj = sparse.coo_matrix((np.ones(len(edge)),(edge[:,0],edge[:,1])),shape=(n,n)).tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0); adj.eliminate_zeros(); adj.data.fill(1.); adj.sort_indices()
    train = ids[0]
    p = np.tanh((x-x[train].mean(0))/np.maximum(1e-8,x[train].std(0)))
    deg,r,local,spatial,ordinary = normalized_means(adj,p)
    inv = np.divide(1.,deg,out=np.zeros_like(deg),where=deg>0)
    sd = np.sqrt(np.maximum((adj @ (p*p))*inv[:,None]-ordinary**2,0.))
    mu = (adj @ inv)*inv
    sq = (adj @ (inv**2))*inv
    h = np.sqrt(np.maximum(np.divide(sq,mu**2,out=np.ones_like(deg),where=mu>0)-1,0))
    h[deg<2]=0.
    cut = float(np.quantile(h[train],.75,method='linear'))
    assert abs(cut-.926159345803283)<1e-10
    assert check_cubic() < 1e-12
    # An independent 301-row cross-check evaluates the normalized cubic signal.
    err = 0.
    for i in np.linspace(0,n-1,301,dtype=int):
        nb = adj.indices[adj.indptr[i]:adj.indptr[i+1]]
        direct = ((p[nb]-p[i])*inv[i]*inv[nb,None]).sum(0) if len(nb) else 0.
        err = max(err,float(np.max(np.abs((spatial[i]-local[i])*r[i]-direct))))
    assert err < 1e-12
    added = adj + sparse.eye(n,format='csr')
    inv_rt = 1 / np.sqrt(deg+1)
    prop = added.multiply(inv_rt[:,None]).multiply(inv_rt[None,:]).tocsr()
    common = np.empty((n,4*128+2+2*128),np.float32)
    b = x
    common[:,0:128] = b
    for hidx in (1,2,3):
        b = prop @ b
        common[:,128*hidx:128*(hidx+1)] = b
    common[:,512] = np.log1p(deg); common[:,513] = r
    common[:,514:642] = ordinary; common[:,642:] = sd
    assert np.isfinite(common).all() and np.isfinite(spatial).all()
    report = dict(archive_sha256=DATA_HASH,split_sha256=SPLIT_HASH,n=n,
                  input_width=common.shape[1]+128,common_width=common.shape[1],
                  raw_edges=len(edge),symmetrized_edges=adj.nnz//2,
                  isolates=int((deg==0).sum()),fixed_row_max_cubic_error=err,
                  h_train_q75=cut,n_train_high=int((h[train]>cut).sum()))
    return common,{'Root-local':local.astype(np.float32),'Spatial':spatial.astype(np.float32)},y,ids,h>cut,report


def standardize(x,train):
    # Population moments in FP64; apply in FP32 for the network.
    mean = x[train].mean(0,dtype=np.float64).astype(np.float32)
    sd = x[train].std(0,dtype=np.float64).clip(1e-6).astype(np.float32)
    return (x-mean)/sd


def design(common,extra,tri,idx,arm):
    d = np.zeros((len(idx), common.shape[1]+128),np.float32)
    d[:,:common.shape[1]] = common[idx]
    if arm != 'Common':
        d[:,common.shape[1]:] = extra[arm][idx]
    return torch.from_numpy(d).cuda()


def net(dim):
    return nn.Sequential(nn.Linear(dim,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.4),
                         nn.Linear(256,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.4),
                         nn.Linear(256,40)).cuda()


@torch.no_grad()
def predict(model,z):
    model.eval()
    return torch.cat([model(chunk).argmax(1) for chunk in z.split(4096)])


def fit(xt,yt,xv,yv,seed):
    torch.manual_seed(seed)
    gen = torch.Generator(device='cuda').manual_seed(seed+20000)
    model = net(xt.shape[1])
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,fused=True)
    best = -1
    curve = []
    for epoch in range(1,121):
        model.train()
        groups = torch.randperm(len(yt),device='cuda',generator=gen).split(2048)
        for ix in groups:
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(xt[ix]),yt[ix]).backward()
            opt.step()
        correct = int((predict(model,xv)==yv).sum())
        curve.append(correct)
        if correct > best:
            best = correct
            chosen = epoch
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        elif epoch-chosen >= 20:
            break
    return state,dict(seed=seed,selected_epoch=chosen,trained_epochs=epoch,
                      val_accuracy=best/len(yv),validation_correct=curve)


def delta(vals):
    vals = np.asarray(vals)
    avg = float(vals.mean())
    hw = float(t.ppf(.975,len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals)))
    return dict(mean=avg,paired_t95=[avg-hw,avg+hw],by_seed=vals.tolist(),num_positive=int((vals>0).sum()))


def summarize(pred,y,mask):
    truth = y[mask]
    size = np.bincount(truth,minlength=40)
    present = size>0
    acc,macro,arms = {},{},{}
    for a in ARMS:
        score = pred[a][:,mask]==truth[None,:]
        acc[a] = score.mean(1)
        by_class = np.stack([np.bincount(truth,weights=s,minlength=40)[present]/size[present] for s in score])
        macro[a] = by_class.mean(1)
        arms[a] = dict(accuracy=float(acc[a].mean()),macro=float(macro[a].mean()),
                       byseed_accuracy=acc[a].tolist(),byseed_macro=macro[a].tolist())
    comparisons = {}
    for b in ('Root-local','Common'):
        lo = pred[b][:,mask]==truth
        hi = pred['Spatial'][:,mask]==truth
        comparisons['Spatial-minus-'+b] = dict(accuracy=delta(acc['Spatial']-acc[b]),
                macro=delta(macro['Spatial']-macro[b]),corrected=(hi & ~lo).sum(1).tolist(),
                worsened=(lo & ~hi).sum(1).tolist())
    return dict(papers=int(mask.sum()),classes=int(present.sum()),arms=arms,contrasts=comparisons)


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ('archive','base','out'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    sys.path.insert(0,str(args.base.resolve()))
    from arxiv_readout import digest,load_graph,normalized_means,check_cubic,signed_rows
    assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(1)
    # This job shares the device with the official PCBA comparison.
    torch.cuda.set_per_process_memory_fraction(min(.085,6.8*1024**3/torch.cuda.get_device_properties(0).total_memory))
    common,extra,y,ids,high,rep = prepare(args.archive,(digest,load_graph,normalized_means,check_cubic,signed_rows))
    tri,vai,tei = ids
    common = standardize(common,tri)
    extra = {name:standardize(z,tri) for name,z in extra.items()}
    yt,yv = [torch.as_tensor(y[idx],device='cuda') for idx in (tri,vai)]
    states,records = {},{}
    for arm in ARMS:
        xt,xv = [design(common,extra,tri,idx,arm) for idx in (tri,vai)]
        states[arm], records[arm] = [],[]
        for seed in SEEDS:
            # Per-fit checkpoints allow interruption-safe continuation before any new test score.
            part = args.out/f'fit_{arm}_{seed}_server_only.pt'
            if part.exists():
                cache = torch.load(part,weights_only=False,map_location='cpu')
                assert cache['study_sha256'] == digest(Path(__file__)) and cache['protocol_sha256'] == digest(args.out/'protocol.md')
                state,rr = cache['state'],cache['selection']
            else:
                state,rr = fit(xt,yt,xv,yv,seed)
                cache = dict(study_sha256=digest(Path(__file__)),protocol_sha256=digest(args.out/'protocol.md'),
                             state=state,selection=rr)
                tmp = part.with_suffix('.part'); torch.save(cache,tmp); tmp.replace(part)
            states[arm].append(state); records[arm].append(rr)
            print('selected',arm,seed,rr['selected_epoch'],rr['trained_epochs'],flush=True)
        del xt,xv
    assert sum(map(len,states.values()))==15
    print('all validation-selected checkpoints fixed; beginning test',flush=True)
    preds = {}
    for arm in ARMS:
        xtest = design(common,extra,tri,tei,arm)
        model = net(xtest.shape[1])
        scores = []
        for state in states[arm]:
            model.load_state_dict(state)
            scores.append(predict(model,xtest).cpu().numpy())
        preds[arm] = np.stack(scores)
        del xtest
    np.savez(args.out/'test_predictions_server_only.npz',**preds,idx=tei,truth=y[tei],high=high[tei])
    groups = {'overall':np.ones(len(tei),bool),'high':high[tei],'remaining':~high[tei]}
    rep['groups'] = {name:summarize(preds,y[tei],mask) for name,mask in groups.items()}
    rep['interaction_high_minus_remaining'] = {}
    for name in ('Spatial-minus-Root-local','Spatial-minus-Common'):
        get = lambda k: np.array(rep['groups'][k]['contrasts'][name]['accuracy']['by_seed'])
        rep['interaction_high_minus_remaining'][name] = delta(get('high')-get('remaining'))
    rep.update(fits=records,seeds=SEEDS,device=torch.cuda.get_device_name(),torch=str(torch.__version__),
               cuda=torch.version.cuda,tf32=True,numpy=np.__version__,peak_gpu_gib=torch.cuda.max_memory_reserved()/1024**3,
               source_sha256=digest(Path(__file__)),helper_sha256=digest(args.base/'arxiv_readout.py'),
               protocol_sha256=digest(args.out/'protocol.md'),parameters=sum(p.numel() for p in net(898).parameters()))
    (args.out/'aggregate.json').write_text(json.dumps(rep,indent=2,allow_nan=False)+'\n')
    print('aggregate written',flush=True)


if __name__=='__main__':
    main()
