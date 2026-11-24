"""Train a paired readout comparison with the released AdaSpec GPRGNN on ZINC.

Run with --data pointing to the existing PyG ZINC root and --adaspec to
the 'small' folder of the pinned upstream release. Keep --work server-side.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric
from torch_geometric.datasets import ZINC

SEEDS = [103, 211, 307, 401, 503]
ARMS = ["uniform", "weighted_root_local", "weighted_spatial"]
COUNTS = {"train": 10000, "val": 1000, "test": 1000}
HASHES = {
    "gpr_model.py": "be9bb5070da889f2cd66e4a07a27cbceb8f2a8bb23c3b708028b0e49ed0c5910",
    "training.py": "42cc5a0c949d9ea88e001df15b3b57699aba998392eb33af583bc29f6b81bb00",
}
NATOM, DEPTH, BATCH = 28, 10, 256
DEVICE = "cuda"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def construct(graph):
    atoms = graph.x.numpy().reshape(-1)
    source, target = graph.edge_index.numpy()
    n = len(atoms)
    eye = np.eye(n)
    onehot = np.eye(NATOM)[atoms]
    adjacency = np.zeros((n, n))
    adjacency[source, target] = 1.
    assert np.array_equal(adjacency, adjacency.T) and np.all(adjacency.diagonal() == 0)
    degree = adjacency.sum(1)
    bias = degree / (1 + degree)
    inv = np.divide(1., np.sqrt(degree + bias), out=np.zeros(n), where=degree + bias > 0)
    # Released cal_pert_mat uses per-feature squared norms on each graph.
    norms = np.bincount(atoms, minlength=NATOM)
    perturb = ((atoms[source] == atoms[target]) / norms[atoms[source]]).astype(np.float32)
    matrix = (adjacency + np.diag(bias)) * inv[:, None] * inv[None, :] + .1 * eye
    matrix[source, target] += (.01 * perturb).astype(np.float64)
    p = np.diag(matrix).copy()
    s = matrix - np.diag(p)
    local = (s @ s @ s).diagonal() + 3*p*(s*s).sum(1) + p**3
    spatial = (matrix @ matrix @ matrix).diagonal()
    energy = np.triu(s*s * (p[:, None]-p[None, :])**2, k=1).sum()
    err1 = abs(spatial.sum() - local.sum())
    err2 = abs(p @ (spatial-local) + energy)
    assert max(err1, err2) < 1e-10
    col = np.ones(n)
    grouped = []
    for k in range(DEPTH+1):
        grouped.append(col @ onehot)
        col = matrix @ col
    i, j = np.nonzero(matrix)
    return (np.stack(grouped), [local.sum(), p @ local, p @ spatial],
            (torch.LongTensor(atoms), torch.LongTensor(np.stack([i,j])),
             torch.tensor(matrix[i,j], dtype=torch.float32)),
            (err1, err2, float(energy)))


def verify_matrix(graph, atoms_edge, folder):
    import textwrap
    from torch_scatter import scatter_add
    lines = (folder/"training.py").read_text().splitlines()
    scope = dict(torch=torch, np=np, F=F, scatter_add=scatter_add)
    exec("\n".join(lines[243:281]), scope)
    x = F.one_hot(graph.x.flatten(), NATOM).double()
    n = len(x)
    e = graph.edge_index.long()
    d = torch.bincount(e[0], minlength=n).double()
    model = nn.Module()
    model.register_parameter("diag_bias", nn.Parameter(d/(1+d)))
    data = SimpleNamespace(x=x, num_nodes=n)
    scope.update(model=model, data=data, ori_edge_index=e, ori_num_edge=e.size(1),
                 ori_edge_weight=torch.ones(e.size(1), dtype=torch.float64),
                 perturb_mat_weight=scope["cal_pert_mat"](e,x),
                 args=SimpleNamespace(alpha0=1.,alpha1=.1,alpha2=.01))
    exec(textwrap.dedent("\n".join(lines[155:185])), scope)
    m = torch.sparse_coo_tensor(data.edge_index, data.edge_weight,(n,n)).to_dense()
    _, index, weight = atoms_edge
    our = torch.sparse_coo_tensor(index, weight, (n,n)).to_dense()
    limit = (m.float()-our).abs().max().item()
    assert limit < 1e-7, limit
    return limit


def load_graphs(root, folder):
    blocks, stats = {}, {}
    for split, count in COUNTS.items():
        ds = ZINC(str(root),subset=True,split=split)
        assert len(ds)==count
        response, coords, molecules, checks, y, materr = [], [], [], [], [], []
        for i, g in enumerate(ds):
            x,c,m,v=construct(g)
            response.append(x); coords.append(c); molecules.append(m); checks.append(v)
            if split != "test":
                y.append(g.y.item())
            if split=="train" and i<32:
                materr.append(verify_matrix(g,m,folder))
        a=np.asarray(checks)
        stats[split]={
            "n":len(ds),"uniform_max_abs_error":float(a[:,0].max()),
            "weighted_energy_max_abs_error":float(a[:,1].max()),
            "positive_energy_graphs":int(np.count_nonzero(a[:,2]>1e-12)),
            "mean_edge_energy":float(a[:,2].mean()),
        }
        if materr:
            stats[split]["published_matrix_check_n"]=len(materr)
            stats[split]["published_matrix_max_abs_error_fp32"]=max(materr)
        blocks[split]=dict(
            g=torch.tensor(np.asarray(response),dtype=torch.float32,device=DEVICE),
            c=torch.tensor(np.asarray(coords),dtype=torch.float64,device=DEVICE),
            molecules=molecules,
            y=torch.tensor(y,dtype=torch.float32,device=DEVICE) if y else None,
        )
        print("prepared",split,len(ds),flush=True)
    mu=blocks["train"]["c"].mean(0)
    sigma=blocks["train"]["c"].std(0,unbiased=False)
    assert sigma.min().item()>0
    for block in blocks.values():
        block["c"]=((block["c"]-mu)/sigma).float()
    return blocks,stats


def direct_batch(molecules, ids):
    atoms, idx, weights=[],[],[]
    size=[]
    offset=0
    for i in ids:
        a,e,w=molecules[int(i)]
        atoms.append(a);idx.append(e+offset);weights.append(w)
        size.append(len(a));offset+=len(a)
    a=torch.cat(atoms).to(DEVICE)
    graph_index=torch.repeat_interleave(torch.arange(len(size)),torch.tensor(size)).to(DEVICE)
    return SimpleNamespace(x=F.one_hot(a,NATOM).float(), edge_index=torch.cat(idx,1).to(DEVICE),
                           edge_weight=torch.cat(weights).to(DEVICE)),graph_index


class GraphPredictor(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        args=SimpleNamespace(hidden=64,with_bias=True,ppnp="GPR_prop",K=DEPTH,alpha=.1,
                             Init="PPR",Gamma=None,dprate=0.,dropout=0.,ret_logit=True,
                             learn_mat=True)
        self.backbone=backbone(SimpleNamespace(num_features=NATOM,num_classes=16),args)
        self.head=nn.Sequential(nn.Linear(17,64),nn.ReLU(),nn.Linear(64,1))
        self.register_buffer("onehot",torch.eye(NATOM))

    def predict(self, pooled, coord):
        return self.head(torch.cat([pooled/16.,coord[:,None]],dim=1)).flatten()

    def grouped(self, block, ids, arm):
        h=self.backbone.lin2(F.relu(self.backbone.lin1(self.onehot)))
        count=torch.einsum("bka,k->ba",block["g"][ids],self.backbone.prop1.temp)
        return self.predict(count@h,block["c"][ids,arm])

    def direct(self, block, ids, arm):
        data, membership=direct_batch(block["molecules"],ids)
        logits,_=self.backbone(data)
        pooled=logits.new_zeros(len(ids),16).index_add_(0,membership,logits)
        return self.predict(pooled,block["c"][ids,arm])


def fit(blocks, backbone, seed, arm, work):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    net=GraphPredictor(backbone).cuda()
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    rng=np.random.default_rng(seed)
    train,val=blocks["train"],blocks["val"]
    val_ids=torch.arange(len(val["y"]),device=DEVICE)
    best=float("inf"); max_gap=0.
    for epoch in range(1,161):
        net.train()
        order=rng.permutation(len(train["y"]))
        for start in range(0,len(order),BATCH):
            ids=order[start:start+BATCH]
            opt.zero_grad(set_to_none=True)
            if start==0:
                pred=net.direct(train,ids,arm)
                with torch.no_grad():
                    check=net.grouped(train,ids,arm)
                    max_gap=max(max_gap,float((pred-check).abs().max()))
                    assert torch.allclose(pred,check,atol=1e-4,rtol=1e-5),(epoch,max_gap)
            else:
                pred=net.grouped(train,ids,arm)
            loss=F.l1_loss(pred,train["y"][ids])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            va=float(F.l1_loss(net.grouped(val,val_ids,arm),val["y"]))
        if va<best:
            best,selected=va,epoch
            state=copy.deepcopy(net.state_dict())
        if epoch-selected>=25:
            break
    torch.save(state,work/f"{seed}-{arm}.pth")
    return dict(seed=seed,arm=ARMS[arm],validation_mae=best,best_epoch=selected,epochs=epoch,
                train_max_grouped_forward_abs_diff=max_gap,
                parameters=sum(p.numel() for p in net.parameters()))


def main():
    ap=argparse.ArgumentParser()
    for name in ["data","adaspec","work","output"]:
        ap.add_argument("--"+name,type=Path,required=True)
    args=ap.parse_args()
    for name,h in HASHES.items():
        assert sha(args.adaspec/name)==h
    sys.path.insert(0,str(args.adaspec.resolve()))
    from gpr_model import GPRGNN
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args.work.mkdir(exist_ok=True,parents=True)
    blocks,checks=load_graphs(args.data,args.adaspec)
    records=[]
    for seed in SEEDS:
        for arm in range(3):
            record=fit(blocks,GPRGNN,seed,arm,args.work)
            records.append(record)
            print("fit",seed,ARMS[arm],record["best_epoch"],record["epochs"],flush=True)
    target=ZINC(str(args.data),subset=True,split="test")
    y=np.array([g.y.item() for g in target])
    split=blocks["test"]
    errors=np.empty((len(SEEDS),3,len(y)))
    with torch.no_grad():
        for j,rec in enumerate(records):
            ti,arm=divmod(j,3)
            model=GraphPredictor(GPRGNN).cuda().eval()
            model.load_state_dict(torch.load(args.work/f"{rec['seed']}-{arm}.pth",weights_only=True))
            direct,checked=[],[]
            for start in range(0,len(y),BATCH):
                ids=np.arange(start,min(start+BATCH,len(y)))
                p=model.direct(split,ids,arm)
                grouped=model.grouped(split,ids,arm)
                assert torch.allclose(p,grouped,atol=1e-4,rtol=1e-5)
                direct.extend(p.cpu().tolist())
                checked.append(float((p-grouped).abs().max()))
            errors[ti,arm]=np.abs(np.asarray(direct)-y)
            rec.update(test_mae=float(errors[ti,arm].mean()),test_max_grouped_forward_abs_diff=max(checked))
    rng=np.random.default_rng(731)
    draw=rng.integers(0,len(y),size=(2000,len(y)))
    intervals={}
    for i in (0,1):
        delta=errors[:,i]-errors[:,2]
        seed_delta=delta.mean(1)
        avg=seed_delta.mean()
        rad=2.7764451051977987*seed_delta.std(ddof=1)/np.sqrt(len(SEEDS))
        intervals[ARMS[i]]=dict(mean=avg,per_seed=seed_delta.tolist(),seed_ci=[avg-rad,avg+rad],
                                molecule_ci=np.quantile(delta.mean(0)[draw].mean(1),[.025,.975]).tolist())
    result=dict(dataset="ZINC-12k",metric="MAE",seeds=SEEDS,arms=ARMS,records=records,
                h100=torch.cuda.get_device_name(),torch_version=torch.__version__,pyg_version=torch_geometric.__version__,
                commit="d76aa64d3075a73b014345548473d672f103045a",source_sha256=HASHES,
                protocol_sha256=sha(Path(__file__).with_name("protocol.md")),code_sha256=sha(Path(__file__)),
                split_sha256={s:sha(args.data/"raw"/(s+".index")) for s in COUNTS},
                pooled_coordinate_checks=checks,mean_mae=dict(zip(ARMS,errors.mean(axis=(0,2)).tolist())),
                paired_control_minus_spatial=intervals)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"mean_mae":result["mean_mae"],"intervals":intervals}),flush=True)


if __name__=="__main__":
    main()
