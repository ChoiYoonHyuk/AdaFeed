"""One-order spatial vs root-local comparison with paired validation and cluster intervals."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import t
from sklearn.metrics import average_precision_score
import sklearn


SEEDS = (103,211,307,401,503)
ARM = ('local','spatial')
SENS = ('common','shell_count')
DIST = (3,2)
REPS = 1000


def predictor(dim):
    return nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),
                         nn.Linear(64,1)).cuda()


@torch.no_grad()
def score(model, x):
    return torch.cat([model(a).flatten() for a in x.split(8192)]).cpu().numpy()


def fit(trainx, trainy, validx, validy, seed):
    torch.manual_seed(seed)
    gen = torch.Generator(device='cuda').manual_seed(seed+10000)
    model = predictor(trainx.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001,fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=15,eta_min=.0003)
    best = -float('inf')
    history = []
    for epoch in range(1,16):
        model.train()
        for ix in torch.randperm(len(trainy), generator=gen, device='cuda').split(8192):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(trainx[ix]).flatten(),trainy[ix])
            loss.backward()
            optimizer.step()
        sched.step()
        model.eval()
        valid = float(average_precision_score(validy,score(model,validx)))
        history.append(valid)
        if valid > best:
            best = valid
            selected = epoch
            state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    return state, dict(seed=seed,epoch=selected,val_ap=best,validation_curve=history)


def standard(x, mean, sd):
    return np.clip((x-mean)/sd,-10,10)


def take(data, d, sens, arm, ix):
    common = data['common'][ix]
    # For p_i=0 on these roots the shifted bank is diag(S^k)/2^k through 2d.
    # k=0 and 1 are constants on the loop-free input, so retain k=2,...,2d.
    orders = np.arange(2,2*d+1)
    lower = data['top'][np.ix_(ix,orders)] / np.exp2(orders)
    parts = [common,lower]
    if sens == 'shell_count':
        parts.append(data['shell_count'][ix,d-2])
    parts.append(data[arm][ix,d-2,None])
    return np.column_stack(parts).astype(np.float32)


def to_gpu(x):
    return torch.from_numpy(x).cuda()


def bootstrap(ids, seed):
    groups, code = np.unique(ids, return_inverse=True)
    n = len(groups)
    rng = np.random.default_rng(seed)
    weights = np.vstack((np.ones(n),rng.multinomial(n,np.full(n,1/n),size=REPS))).astype(np.int32)
    return code,to_gpu(weights)


def intervals(arr1,arr0):
    delta = arr1-arr0
    avg = delta.mean(0)
    return dict(difference=float(avg[0]),percentile95=np.quantile(avg[1:],[.025,.975]).tolist())


def scaffold_map(ogb, graphs):
    from rdkit import Chem,rdBase,RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.warning')
    smiles = pd.read_csv(ogb/'mapping/mol.csv.gz',usecols=['smiles']).smiles.values
    lookup = np.full(len(smiles),-1)
    keys, partial, failed = [],0,0
    for idx in graphs:
        value = smiles[idx]
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            mol = Chem.MolFromSmiles(value,sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                     Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                    partial += 1
                except (ValueError,RuntimeError):
                    mol = None
        if mol is not None:
            try:
                key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=True)
            except (ValueError,RuntimeError):
                key = None
        else:
            key = None
        if key is None:
            key = '__unparsed_scaffolds__'
            failed += 1
        keys.append(key)
    unique,index = np.unique(keys,return_inverse=True)
    lookup[graphs] = index
    return lookup, dict(rdkit=rdBase.rdkitVersion,computed_test_molecules=len(keys),
                        empty_scaffold_molecules=keys.count(''),partial_sanitize=partial,
                        failed_parse=failed,clusters=len(unique))


def ap_row(score_by_seed,y,groups,w,aps):
    return np.stack([aps(ps,y,groups,w) for ps in score_by_seed])


def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    for s in ('ogb','base','out'):
        parser.add_argument(s,type=Path)
    args = parser.parse_args()
    sys.path.insert(0,str(args.base.resolve()))
    from evaluate import aps  # Original tie-aware GPU clustered AP, cross-checked with sklearn.
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    torch.cuda.set_per_process_memory_fraction(min(.13,11.5*1024**3/torch.cuda.get_device_properties(0).total_memory))
    old = np.load(args.base/'atom_features.npz')
    derived = np.load(args.out/'distance_features.npz')
    top, graph, y, split = (old[k] for k in ('top','graph','y','split'))
    dist = derived['dist']
    carbon = old['carbon'].astype(bool)
    # graph size, root degree, static bank, element/composition and 11 one-hop counts.
    data = {k:derived[k] for k in ('local','spatial','reconstruction','shell_count')}
    data.update(top=top, common=np.column_stack((top,old['chem'],old['neigh'])))
    selected, records, scalers, input_dims, idx = {},{},{},{},{}
    for d in DIST:
        usable = carbon & (dist == d)
        idx[d] = [np.flatnonzero(usable & (split == part)) for part in range(3)]
        tri,vai,_ = idx[d]
        yt = to_gpu(y[tri].astype('float32'))
        for sens in SENS:
            for arm in ARM:
                key = f'd{d}/{sens}/{arm}'
                train = take(data,d,sens,arm,tri)
                valid = take(data,d,sens,arm,vai)
                mu = train.mean(0,dtype=np.float64).astype(np.float32)
                sd = train.std(0,dtype=np.float64).clip(1e-4).astype(np.float32)
                scalers[key] = mu,sd
                xt,xv = [to_gpu(standard(x,mu,sd)) for x in (train,valid)]
                del train,valid
                selected[key], records[key] = [],[]
                input_dims[key] = int(xt.shape[1])
                for seed in SEEDS:
                    state, rec = fit(xt,yt,xv,y[vai],seed)
                    selected[key].append(state)
                    records[key].append(rec)
                    print('selected',key,seed,rec['epoch'],flush=True)
                del xt,xv
        del yt
    assert sum(map(len,selected.values())) == 40
    torch.save(dict(states=selected,scalers=scalers,selection=records),
               args.out/'selected_server_only.pt')
    print('all 40 validation-selected checkpoints locked; starting test',flush=True)
    out = dict(seeds=SEEDS,device=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,
               cuda=torch.version.cuda,sklearn=sklearn.__version__,input_dims=input_dims,selection=records,
               protocol_sha256=sha(args.out/'protocol.md'),study_sha256=sha(__file__),
               bootstrap_code_sha256=sha(args.base/'evaluate.py'),bootstrap_repetitions=REPS,
               molecule_rng=180421,scaffold_rng=180422,results={},counts={})
    all_test_mols = np.unique(graph[(split==2)&carbon&np.isin(dist,DIST)])
    scaff, out['scaffold_parse'] = scaffold_map(args.ogb,all_test_mols)
    for d in DIST:
        splits = idx[d]
        ti = splits[2]
        g = graph[ti]
        c = scaff[g]
        assert (c>=0).all()
        yt = y[ti]
        groups_m,wm = bootstrap(g,180421)
        groups_c,wc = bootstrap(c,180422)
        test_mols = np.unique(g)
        nsc = np.unique(c).size
        atom_counts = np.unique(c,return_counts=True)[1]
        mol_counts = np.unique(scaff[test_mols],return_counts=True)[1]
        dcounts = {}
        for name,ix in zip(('train','valid','test'),splits):
            dcounts[name] = dict(atoms=len(ix),positive=int(y[ix].sum()),
                                 prevalence=float(y[ix].mean()),molecules=len(np.unique(graph[ix])))
        dcounts['test'].update(scaffolds=int(nsc),max_scaffold_molecules=int(mol_counts.max()),
              max_scaffold_fraction_molecules=float(mol_counts.max()/len(test_mols)),
              max_scaffold_atoms=int(atom_counts.max()),max_scaffold_fraction_atoms=float(atom_counts.max()/len(ti)))
        out['counts'][f'd{d}'] = dcounts
        for sens in SENS:
            scores = {}
            boot_m,boot_c = {},{}
            recheck = {}
            for arm in ARM:
                key = f'd{d}/{sens}/{arm}'
                mu,sd = scalers[key]
                x = take(data,d,sens,arm,ti)
                norm = standard(x,mu,sd)
                xx = to_gpu(norm)
                if arm == 'spatial':
                    xr = take(data,d,sens,'reconstruction',ti)
                    rawnorm = standard(xr,mu,sd)
                    recheck.update(
                      input_fp64_max=float(np.max(np.abs(data['spatial'][ti,d-2]-data['reconstruction'][ti,d-2]))),
                      input_fp32_max=float(np.max(np.abs(x[:,-1]-xr[:,-1]))),
                      standardized_fp32_max=float(np.max(np.abs(norm[:,-1]-rawnorm[:,-1]))),
                      input_fp32_unequal=int(np.count_nonzero(x[:,-1] != xr[:,-1])),
                      logit_fp32_max=0.,unequal_logits=0,ap_max=0.)
                    rx = to_gpu(rawnorm)
                scores[arm] = []
                for i,state in enumerate(selected[key]):
                    model = predictor(xx.shape[1]); model.load_state_dict(state); model.eval()
                    ps = score(model,xx)
                    records[key][i]['test_ap'] = float(average_precision_score(yt,ps))
                    scores[arm].append(ps)
                    if arm == 'spatial':
                        rs = score(model,rx)
                        recheck['logit_fp32_max'] = max(recheck['logit_fp32_max'],float(abs(ps-rs).max()))
                        recheck['unequal_logits'] += int(np.count_nonzero(ps != rs))
                        recheck['ap_max'] = max(recheck['ap_max'],abs(float(average_precision_score(yt,rs))
                                                                  - records[key][i]['test_ap']))
                scores[arm] = np.stack(scores[arm])
                boot_m[arm] = ap_row(scores[arm],yt,groups_m,wm,aps)
                boot_c[arm] = ap_row(scores[arm],yt,groups_c,wc,aps)
            vals = boot_m['spatial'][:,0]-boot_m['local'][:,0]
            hw = t.ppf(.975,len(vals)-1) * vals.std(ddof=1) / np.sqrt(len(vals))
            row = dict(arm_test_ap={a:float(boot_m[a][:,0].mean()) for a in ARM},
                       difference=float(vals.mean()),paired_seed_t95=[float(vals.mean()-hw),float(vals.mean()+hw)],
                       seed_differences=vals.tolist(),seeds_positive=int((vals>0).sum()),
                       paired_molecules=intervals(boot_m['spatial'],boot_m['local']),
                       paired_scaffolds=intervals(boot_c['spatial'],boot_c['local']),
                       frozen_order_reconstruction=recheck)
            out['results'][f'd{d}/{sens}'] = row
            np.savez(args.out/f'predictions_d{d}_{sens}_server_only.npz',idx=ti,truth=yt,graph=g,scaffold=c,**scores)
            print('evaluated',f'd{d}/{sens}',flush=True)
        del wm,wc
    out['peak_gpu_gib']=torch.cuda.max_memory_reserved()/1024**3
    out['identity']=json.loads((args.out/'identity_aggregate.json').read_text())
    (args.out/'aggregate.json').write_text(json.dumps(out,indent=2)+'\n')
    print('aggregate written',flush=True)


if __name__ == '__main__':
    main()
