"""Matched structural encodings for the struc2vec airport classification tasks."""
import argparse
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

URL = "https://raw.githubusercontent.com/leoribeiro/struc2vec/77b4cc63e79a37b1496f340c7772d932038417ec/graph/"
NAMES = ("usa", "europe", "brazil")
SEED = 20261121
ARMS = ("degree", "static_long", "local", "spatial", "local_neighbors", "spatial_neighbors")

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def read_graph(folder, name):
    names = (f"{name}-airports.edgelist", f"labels-{name}-airports.txt")
    files = [folder / fname for fname in names]
    for file in files:
        if not file.exists():
            with urlopen(URL + file.name, timeout=45) as response:
                data = response.read()
            assert len(data) > 100
            tmp = file.with_suffix(file.suffix + ".partial")
            tmp.write_bytes(data)
            tmp.replace(file)
    raw_labels = np.loadtxt(files[1], skiprows=1, dtype=np.int64)
    order = np.argsort(raw_labels[:, 0])
    ids, labels = raw_labels[order].T
    assert len(set(ids)) == len(ids) and len(np.unique(labels)) == 4
    nodes = {i: j for j, i in enumerate(ids)}
    a = np.zeros((len(ids), len(ids)))
    for i, j in np.loadtxt(files[0], dtype=np.int64):
        u, v = nodes[i], nodes[j]
        if u != v:
            a[u,v] = a[v,u] = 1
    return a, labels, {f.name: digest(f) for f in files}

def bank(b, kmax):
    cols = [torch.ones(len(b), device=b.device, dtype=b.dtype), torch.diag(b)]
    h = b
    for k in range(2, kmax+1):
        h = h @ b
        cols.append(h.diagonal().clone())
    return torch.stack(cols, dim=1)

def features(adjacency):
    a = torch.from_numpy(adjacency).cuda()
    d = a.sum(1)
    rho = torch.linalg.eigvalsh(a)[-1]
    b = a / rho
    static = bank(b, 14)
    p = torch.log1p(d) / torch.log1p(d.max())
    spatial = bank(b + torch.diag(p), 8)
    local = torch.stack([sum(comb(k,j) * p.pow(k-j)*static[:,j] for j in range(k+1))
                         for k in range(3,9)], dim=1)
    i = int(torch.argmax(d))
    local_check = torch.linalg.matrix_power(b+p[i]*torch.eye(len(a),device=a.device,dtype=a.dtype),8)[i,i]
    assert abs(float(local_check - local[i,-1])) < 1e-9
    cubic = (b.square() @ p) - b.square().sum(1)*p
    err = float((spatial[:,3] - local[:,0] - cubic).abs().max())
    assert err<1e-10
    norm = d.clamp_min(1)
    mean = a@p /norm
    second = a@p.square()/norm
    lowest = torch.where(a.bool(),p[None,:],1.).amin(1)
    highest = torch.where(a.bool(),p[None,:],0.).amax(1)
    neighbors = torch.stack([mean,second,a@(p**3)/norm,second-mean**2,
                             torch.where(d>0,lowest,0.),highest],1)
    transform = lambda x: torch.log1p(1000*x.clamp_min(0))
    base = torch.column_stack((torch.log1p(d),p,transform(static[:,2:9])))
    xlo = torch.column_stack((base,transform(local)))
    xsp = torch.column_stack((base,transform(spatial[:,3:9])))
    xs = [torch.log1p(d)[:,None],torch.column_stack((base,transform(static[:,9:15]))), xlo,xsp,
          torch.column_stack((xlo,neighbors)),torch.column_stack((xsp,neighbors))]
    return {k:x.cpu().numpy() for k,x in zip(ARMS,xs)}, cubic.abs().cpu().numpy(),err

def evaluate(a,y):
    xs,contrast,err = features(a)
    labels, codes = np.unique(y,return_inverse=True)
    folds = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    losses = {arm:np.empty(len(y)) for arm in ARMS}
    acc = {arm:np.empty(len(y)) for arm in ARMS}
    fold_gaps=[]
    for train,test in folds.split(codes,codes):
        for arm in ARMS:
            fit = make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver="lbfgs",
                                                                     max_iter=2000,tol=1e-9))
            fit.fit(xs[arm][train],codes[train])
            prob=fit.predict_proba(xs[arm][test])
            losses[arm][test]=-np.log(np.clip(prob[np.arange(len(test)),codes[test]],1e-15,1))
            acc[arm][test]=prob.argmax(1)==codes[test]
        fold_gaps.append(float((losses["spatial"][test]-losses["local"][test]).mean()))
    diff=losses["spatial"]-losses["local"]
    threshold=float(np.median(contrast))
    high=contrast>threshold
    assert high.any() and (~high).any()
    return dict(n=len(y), edges=int(a.sum()/2), class_counts=np.bincount(codes).tolist(),
                high_count=int(high.sum()), median_contrast=threshold, cubic_error=err,
                scores={arm:dict(loss=float(losses[arm].mean()),acc=float(acc[arm].mean())) for arm in ARMS},
                high_gap=float(diff[high].mean()),low_gap=float(diff[~high].mean()),
                interaction=float(diff[high].mean()-diff[~high].mean()), fold_gaps=fold_gaps)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("data",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    assert not args.output.exists() and torch.cuda.is_available()
    args.data.mkdir(exist_ok=True,parents=True)
    result=dict(device=torch.cuda.get_device_name(),torch=str(torch.__version__),sklearn=sklearn.__version__,
                protocol_sha256=digest(Path(__file__).with_name("protocol.md")),script_sha256=digest(Path(__file__)),
                networks={})
    for name in NAMES:
        a,y,hashes=read_graph(args.data,name)
        row=evaluate(a,y)
        row["hashes"]=hashes
        result["networks"][name]=row
        print(name, row["scores"], flush=True)
    result["macro_network_loss"]={arm:float(np.mean([v["scores"][arm]["loss"] for v in result["networks"].values()])) for arm in ARMS}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__":
    main()
