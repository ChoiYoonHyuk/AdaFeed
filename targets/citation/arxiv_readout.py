"""Five fixed ridge heads on official ogbn-arxiv. Pass the official archive or extracted root."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import sparse
import torch

PENALTIES = (1e-6, 1e-5, 1e-4, 1e-3, .01, .1, 1.)
ARMS = ("Local", "Spatial", "Mean+Local", "Mean+Spatial", "Mean")
SEED = 1729


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_graph(root):
    wanted = {
        "x": ("raw/node-feat.csv.gz", np.float32),
        "y": ("raw/node-label.csv.gz", np.int64),
        "edge": ("raw/edge.csv.gz", np.int64),
        **{split: (f"split/time/{split}.csv.gz", np.int64) for split in ("train", "valid", "test")},
    }
    result = {}
    if root.is_file():
        with ZipFile(root) as archive:
            names = archive.namelist()
            for tag, (suffix, dtype) in wanted.items():
                hits = [name for name in names if name==suffix or name.endswith("/" + suffix)]
                assert len(hits)==1, (suffix,hits)
                with archive.open(hits[0]) as file:
                    result[tag] = pd.read_csv(file, header=None, compression="gzip", dtype=dtype).values
    else:
        for tag, (suffix, dtype) in wanted.items():
            result[tag] = pd.read_csv(root/suffix, header=None, compression="gzip", dtype=dtype).values
    return result


def normalized_means(a, fields):
    degree = np.asarray(a.sum(1)).ravel()
    inv = np.divide(1., degree, out=np.zeros_like(degree), where=degree>0)
    denom = np.asarray(a @ inv)
    returns = inv*denom
    numerator = a @ (inv[:,None]*fields)
    spatial = np.divide(numerator, denom[:,None], out=np.zeros_like(numerator), where=denom[:,None]>0)
    ordinary = (a @ fields)*inv[:,None]
    local = fields.copy()
    local[degree==0] = 0
    return degree, returns, local, spatial, ordinary


def check_cubic():
    matrix=np.array([[0,1,1,0,0],[1,0,0,1,0],[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,0]],dtype=float)
    p=np.array([-.7,.1,.8,-.4,.4])
    d,r,pl,ps,_=normalized_means(sparse.csr_matrix(matrix),p[:,None])
    invroot=np.divide(1.,np.sqrt(d),out=np.zeros_like(d),where=d>0)
    s=invroot[:,None]*matrix*invroot[None,:]
    n=len(p)
    spatial=[]
    local=[]
    for sign in [-1.,1.]:
        q=p*sign
        spatial.append(np.diag(np.linalg.matrix_power(s+np.diag(q),3)))
        local.append(np.array([np.linalg.matrix_power(s+q[i]*np.eye(n),3)[i,i] for i in range(n)]))
    a=(spatial[1]-spatial[0])/2-p**3-2*r*p
    b=(local[1]-local[0])/2-p**3-2*r*p
    err=float(max(np.abs(a-ps[:,0]*r).max(),np.abs(b-pl[:,0]*r).max()))
    assert err<1e-12 and pl[-1]==0 and ps[-1]==0
    return err


def signed_rows(arrays):
    h=hashlib.sha256()
    for x in arrays:
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def mean_accuracy(pred,target):
    count=np.bincount(target,minlength=40)
    good=np.bincount(target[pred==target],minlength=40)
    return float(np.mean(good[count>0]/count[count>0]))


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("--output",type=Path,default=Path("arxiv_results.json"))
    a=parser.parse_args()
    assert torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    v=load_graph(a.data)
    x=v["x"].astype(np.float64)
    assert x.shape==(169343,128) and np.isfinite(x).all()
    n=len(x)
    ids={k:v[k].ravel() for k in ("train","valid","test")}
    assert [len(ids[k]) for k in ids]==[90941,29799,48603] and np.array_equal(np.sort(np.concatenate(list(ids.values()))),np.arange(n))
    raw=v["edge"];assert raw.shape==(1166243,2)
    A=sparse.coo_matrix((np.ones(len(raw)),(raw[:,0],raw[:,1])),shape=(n,n)).tocsr()
    A=A.maximum(A.T).tocsr(); A.setdiag(0); A.eliminate_zeros(); A.data.fill(1.); A.sort_indices()
    train=ids["train"];valid=ids["valid"];test=ids["test"]
    m=x[train].mean(0)
    std=x[train].std(0)
    fields=np.tanh((x-m)/np.maximum(std,1e-8))
    degree,ret,local,spatial,ordinary=normalized_means(A,fields)
    common=np.column_stack((x,np.log1p(degree),ret))
    arms={"Local":(local,), "Spatial":(spatial,), "Mean+Local":(ordinary,local), "Mean+Spatial":(ordinary,spatial), "Mean":(ordinary,)}
    y=v["y"].ravel()
    asgpu=lambda z:torch.as_tensor(z,dtype=torch.float64,device="cuda")
    yt=asgpu(np.eye(40)[y[train]])
    ybar=yt.mean(0)
    cv={}
    test_pred={}
    size={}
    for key,extra in arms.items():
        design=np.column_stack((common,*extra))
        size[key]=design.shape[1]
        tview=design[train]
        mean=tview.mean(0)
        std=np.maximum(tview.std(0),1e-8)
        Zt=asgpu((tview-mean)/std)
        Zv=asgpu((design[valid]-mean)/std)
        G=Zt.T@Zt/len(train)
        b=Zt.T@(yt-ybar)/len(train)
        eye=torch.eye(G.shape[0],dtype=torch.float64,device="cuda")
        best=None
        grid={}
        for lam in PENALTIES:
            weights=torch.linalg.solve(G+lam*eye,b)
            pred=(Zv@weights+ybar).argmax(1).cpu().numpy()
            correct=int((pred==y[valid]).sum())
            grid[str(lam)]=correct/len(valid)
            if best is None or (correct,lam)>(best[0],best[1]):
                best=(correct,lam,weights.clone(),pred.copy())
        _,lam,weights,val_pred=best
        test_pred[key]=(asgpu((design[test]-mean)/std)@weights+ybar).argmax(1).cpu().numpy()
        cv[key]={"selected_lambda":lam,"val_accuracy":grid[str(lam)],"validation_grid":grid,"validation_macro":mean_accuracy(val_pred,y[valid])}
        print(key,"selection",lam,cv[key]["val_accuracy"],flush=True)
        del design,tview,Zt,Zv,G,b,eye
    test_y=y[test]
    for k,pred in test_pred.items():
        cv[k]["test_accuracy"]=float((pred==test_y).mean())
        cv[k]["test_correct"]=int((pred==test_y).sum())
        cv[k]["test_macro"]=mean_accuracy(pred,test_y)
    pairs={}
    for l,r in [("Local","Spatial"),("Mean+Local","Mean+Spatial"),("Mean","Mean+Spatial")]:
        p,q=test_pred[l]==test_y,test_pred[r]==test_y
        pairs[l+"__"+r]=dict(improved=int((~p&q).sum()),worsened=int((p&~q).sum()),tied_correct=int((p&q).sum()),tied_wrong=int((~p&~q).sum()),delta_accuracy=float(q.mean()-p.mean()),delta_macro=float(cv[r]["test_macro"]-cv[l]["test_macro"]))
    out=dict(protocol_sha256=digest(Path(__file__).with_name("PROTOCOL.md")),source_sha256=digest(Path(__file__)),data_archive_sha256=digest(a.data) if a.data.is_file() else None,split_sha256=signed_rows(list(ids.values())),n=n,undirected_edges=A.nnz//2,directed_edges=len(raw),classes=40,isolates=int((degree==0).sum()),split_sizes={k:len(i) for k,i in ids.items()},features=size,arms=cv,paired=pairs,cubic_identity_max_abs_error=check_cubic(),gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=str(torch.version.cuda))
    a.output.write_text(json.dumps(out,indent=2)+"\n",encoding="utf8")
    print(json.dumps(pairs),flush=True)

if __name__=="__main__":
    run()
