"""Forward test for the 45-vertex cut pair with the unmodified AdaSpec GPRGNN.

Run from a directory containing gpr_model.py and training.py from the commit below.
Use Python 3.12, PyTorch 2.11, PyG 2.7, torch-scatter and torch-sparse for that PyTorch build.
"""
import hashlib
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_scatter import scatter_add

COMMIT = 'd76aa64d3075a73b014345548473d672f103045a'
FILES = {
    'gpr_model.py': 'be9bb5070da889f2cd66e4a07a27cbceb8f2a8bb23c3b708028b0e49ed0c5910',
    'training.py': '42cc5a0c949d9ea88e001df15b3b57699aba998392eb33af583bc29f6b81bb00',
}
for name, expected in FILES.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name

from gpr_model import GPRGNN

# The authors' RunExp interleaves construction of the propagation matrix with
# supervised optimization. Execute its matrix-building block and its feature
# perturbation helper verbatim, without fitting the two inputs.
lines = Path('training.py').read_text().splitlines()
matrix_block = compile(textwrap.dedent('\n'.join(lines[155:185])), 'training.py:156-185', 'exec')
feature_fn = compile('\n'.join(lines[243:281]), 'training.py:244-281', 'exec')
exec(feature_fn, globals())

def graph(sign):
    adjacency = torch.zeros((45, 45), dtype=torch.long)
    def add(i, j):
        adjacency[i, j] = adjacency[j, i] = 1
    rules = [(0,1,sign),(2,3,1),(0,2,0),(0,3,0),(1,2,0),(1,3,0),(2,4,0),(4,5,0),(5,2,0)]
    for u, v, shift in rules:
        for r in range(3):
            add(3*u+r, 3*v+(r+shift)%3)
    leaf = 18
    for u, multiplicity in enumerate((0,1,0,3,0,5)):
        for r in range(3):
            for _ in range(multiplicity):
                add(3*u+r, leaf)
                leaf += 1
    assert leaf == 45 and adjacency.sum().item() == 108
    return adjacency

def certificate(adjacency):
    square = adjacency @ adjacency
    triangle = (square @ adjacency).diag()
    return int(square.diag() @ (square.square() @ triangle))

def run_matrix(adjacency, gate, model, args):
    ori_edge_index = adjacency.nonzero().T.contiguous()
    x = torch.ones((45, 1), dtype=torch.float64)
    perturb_mat_weight = cal_pert_mat(ori_edge_index, x)
    data = SimpleNamespace(x=x.cuda(), num_nodes=len(x))
    ori_edge_index = ori_edge_index.cuda()
    ori_num_edge = ori_edge_index.shape[1]
    ori_edge_weight = torch.ones(ori_num_edge, dtype=torch.float64, device='cuda')
    d = adjacency.sum(1).double()
    if gate == 'uniform':
        bias = torch.full_like(d, 1/len(x))
    elif gate == 'degree':
        bias = d/(1+d)
    model.diag_bias.copy_(bias.cuda())
    env = dict(locals())
    exec(matrix_block, globals(), env)
    logits, _ = model(data)
    matrix = torch.sparse_coo_tensor(data.edge_index, data.edge_weight, (45,45)).to_dense()
    returns = torch.stack([torch.linalg.matrix_power(matrix, k).diagonal() for k in range(1,11)])
    return logits.cpu(), returns.cpu()

def gap(x, y):
    return float((x-y).abs().max())

@torch.no_grad()
def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    graph_a, graph_b = graph(1), graph(-1)
    assert [certificate(g) for g in (graph_a, graph_b)] == [3540,3528]
    control = graph_a.clone()
    assert control[0,1].item() == 0
    control[0,1] = control[1,0] = 1
    args = SimpleNamespace(hidden=64, with_bias=True, ppnp='GPR_prop', K=10, alpha=.1,
                           Init='PPR', Gamma=None, dprate=0., dropout=0., ret_logit=True,
                           learn_mat=True, alpha0=1., alpha1=.1, alpha2=.01)
    dataset = SimpleNamespace(num_features=1, num_classes=2)
    result = {'commit': COMMIT, 'official_sources_sha256': FILES, 'device': torch.cuda.get_device_name(),
              'torch': str(torch.__version__), 'pyg': torch_geometric.__version__,
              'config': vars(args), 'gamma': [certificate(graph_a),certificate(graph_b)], 'matches': []}
    for seed in range(8):
        torch.manual_seed(seed)
        model = GPRGNN(dataset,args).double().cuda().eval()
        model.register_parameter('diag_bias', torch.nn.Parameter(torch.empty(45,dtype=torch.float64,device='cuda')))
        for gate, edge_alpha in [('uniform',0.),('uniform',.01),('degree',0.),('degree',.01)]:
            args.alpha2 = edge_alpha
            a, ra = run_matrix(graph_a,gate,model,args)
            b, rb = run_matrix(graph_b,gate,model,args)
            c, rc = run_matrix(control,gate,model,args)
            result['matches'].append(dict(seed=seed,gate=gate,edge_alpha=edge_alpha,
                return_gap=gap(ra,rb), fourth_return_gap=gap(ra[3],rb[3]),
                sorted_fourth_return_gap=gap(ra[3].sort().values,rb[3].sort().values),
                fourth_trace_plus_minus=float(ra[3].sum()-rb[3].sum()),
                sorted_logit_gap=gap(a.sort(dim=0).values,b.sort(dim=0).values), logit_gap=gap(a,b), mean_logit_gap=gap(a.mean(0),b.mean(0)),
                control_return_gap=gap(ra,rc),control_logit_gap=gap(a,c),
                control_mean_logit_gap=gap(a.mean(0),c.mean(0))))
    Path('results.json').write_text(json.dumps(result,indent=2)+'\n')
    for gate, edge_alpha in [('uniform',0.),('uniform',.01),('degree',0.),('degree',.01)]:
        runs = [r for r in result['matches'] if r['gate']==gate and r['edge_alpha']==edge_alpha]
        print(gate,edge_alpha,{k:(min(r[k] for r in runs),max(r[k] for r in runs))
                     for k in runs[0] if k.endswith('gap')})
    assert max(r['logit_gap'] for r in result['matches']) < 1e-10
    assert max(r['return_gap'] for r in result['matches'] if r['edge_alpha']==0) < 1e-10
    assert min(r['return_gap'] for r in result['matches'] if r['edge_alpha']>0) > 1e-8
    assert min(r['control_mean_logit_gap'] for r in result['matches']) > 1e-8

if __name__ == '__main__':
    main()

