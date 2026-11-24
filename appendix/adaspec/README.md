# AdaSpec on the cut-transpose pair

The two graphs have 45 vertices and are defined in the manuscript. Each vertex receives the scalar feature 1. This experiment is an untrained, paired forward comparison. Node classification accuracy and fitted node-indexed self-loops are outside the comparison.

## Released source and execution

Use these two official source files at commit d76aa64d3075a73b014345548473d672f103045a of [AdaSpec](https://github.com/Mia-321/AdaSpec/tree/d76aa64d3075a73b014345548473d672f103045a/small).

- [gpr_model.py](https://raw.githubusercontent.com/Mia-321/AdaSpec/d76aa64d3075a73b014345548473d672f103045a/small/gpr_model.py)
- [training.py](https://raw.githubusercontent.com/Mia-321/AdaSpec/d76aa64d3075a73b014345548473d672f103045a/small/training.py)

Place both files beside `verify_adaspec_pair.py`. Its SHA-256 verification pins the source files. The script uses the authors' GPRGNN class directly and executes the feature-weight helper and matrix construction from their training driver. It uses Python 3.12, torch 2.11.0+cu128, torch-geometric 2.7.0, torch-scatter 2.1.2 and torch-sparse 0.6.18, with double-precision model weights on an NVIDIA H100. The authors' explicit cast of the edge correction to single precision is preserved. Run `python verify_adaspec_pair.py` to regenerate `results.json`. Seeds 0 through 7 initialize the two-layer feature map with shared weights for both inputs. The GPR filter has ten propagation steps and PPR coefficients with alpha 0.1. The hidden width is 64 and there are two logits per vertex. Dropout is disabled. The two self-loop settings are 1/45 and d/(1+d). The three matrix coefficients are (1, 0.1, 0) or (1, 0.1, 0.01). A graph formed by adding edge (0,1) to the positive graph tests sensitivity of the graph-mean logits.

## Independent calculation

`python verify_precision.py` uses the Python standard library and 60-digit decimal arithmetic to regenerate `precision.json`. For each matrix power from 1 to 10, it compares the corresponding vertex diagonals, their sorted values and their sum. The comparison uses the mathematical edge correction c = alpha2/45. This independent result also checks the signed trace gap against the closed form from the four-cycles in the manuscript.

The `logit_gap` entry is the maximum absolute discrepancy over all corresponding vertices and both logits. `mean_logit_gap` applies after averaging over vertices. `fourth_trace_plus_minus` is the positive graph's fourth trace minus the negative graph's. The matrix diagonal is a separately measured diagnostic. The returned output of GPRGNN is the node logit array when the ret_logit option is on.

