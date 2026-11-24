# Appendix experiments

Run commands from this directory. For benchmark reruns use a Linux server holding the data. Use a working copy when regenerating results. Several original scripts write their output in the current directory or beside the script.

| Appendix | Directory | Calculation or input |
| --- | --- | --- |
| A, B, C, H | `product/`, `normalized_product/`, `walk_and_normalization/`, and `../brec/` | Exact direct-product, normalized self-loop and constructed graph checks |
| E, I | `finite_brec/`, `normalized_brec/`, `precision/`, `original_brec_replay/` | Finite, learned and normalized queries, pair 296 precision and independent certificate replay on the official BREC array |
| E | `cost/`, `scaling/` | BREC and ZINC timings, nested ogbn-arxiv subgraphs |
| E | `colored_polynomial_banks/`, `arxiv_node/`, `citation_cubic_cost/`, `molecular_shell_cost/` | Equal-output timing on roadNet-PA, ogbn-arxiv and ogbg-molhiv |
| E | `resolvent/` | Independent reconstruction of the 21-pole gradient experiment |
| H | `adaspec/` | Fourth-trace identity and paired AdaSpec forward passes |

## Checks without benchmark data

```bash
python product/check_degree_spectra.py
python product/check_direct_factors.py
python normalized_product/check_normalized_product.py
python walk_and_normalization/check_normalized_feedback.py
python resolvent/check_resolvent.py --permutation-seed 1543
python adaspec/adaspec_fourth_trace.py
python scaling/verify.py
python colored_polynomial_banks/verify_probing.py
```

## Server runs

Set `BREC_NPY` to the absolute path of the official `brec_v3.npy`. The finite comparison uses NumPy, NetworkX, SciPy and PyTorch. Run the three commands in order from one new directory. The first generates the pair diagnostics used by the other two.

```bash
APPENDIX="$PWD"
RUN_DIR=$(mktemp -d)
(cd "$RUN_DIR" && python "$APPENDIX/finite_brec/evaluate.py" "$BREC_NPY" && python "$APPENDIX/finite_brec/paired.py" "$BREC_NPY" && python "$APPENDIX/finite_brec/cfi_neural.py" "$BREC_NPY")
python normalized_brec/normalized_brec.py "$BREC_NPY" --output "$RUN_DIR/normalized.json"
python normalized_brec/verify_static_and_residues.py "$BREC_NPY" --output "$RUN_DIR/static_checks.json"
python precision/cfi296_precision.py "$BREC_NPY" "$RUN_DIR/pair_precision.json"
python original_brec_replay/replay.py original_brec_replay/anonymous-certificates.zip "$BREC_NPY" "$RUN_DIR/recheck"
```

For the recorded BREC and ZINC timings and the citation scaling table, use the paths to the official datasets and an exclusive CUDA GPU for scaling:
 
```bash
python cost/brec_stable_cost.py "$BREC_NPY" "$RUN_DIR/brec_timing.json"
python cost/verify_stable.py "$BREC_NPY" "$RUN_DIR/brec_timing.json" "$RUN_DIR/relabel.json"
python cost/verify_pair_witnesses.py "$BREC_NPY" "$RUN_DIR/brec_timing.json" "$RUN_DIR/pairs.json"
python cost/zinc_cost.py /path/to/molecules.zip /path/to/test.index "$RUN_DIR/zinc_timing.json"
(cd scaling && python prepare_graphs.py /path/to/arxiv/raw/edge.csv.gz && python scale.py && python large_pairs.py && python summarize.py)
```

For the equal-output measurements, run:

```bash
python arxiv_node/same_output.py /path/to/arxiv.zip --output "$RUN_DIR/arxiv_diagonals.json"
python citation_cubic_cost/cubic_cost.py /path/to/arxiv.zip "$RUN_DIR/arxiv_cubic.json"
python molecular_shell_cost/shell_cost.py /path/to/ogbg_molhiv "$RUN_DIR/molecular_shell.json"
mkdir -p "$RUN_DIR/road"
(cd colored_polynomial_banks && g++ -std=c++17 -O3 -fPIC -shared color.cpp -o color.so && python probe.py /path/to/roadNet-PA.txt.gz "$RUN_DIR/road")
```

The per-directory READMEs give dependencies and instructions for the two road-network repeat measurements.

For the AdaSpec forward pass, obtain `gpr_model.py` and `training.py` from the pinned upstream commit listed in `adaspec/README.md`. Place them beside the checker, install the listed PyTorch Geometric dependencies and run `python verify_adaspec_pair.py` from that directory. The checker validates both upstream file hashes.

The original Appendix E precision script covers BREC pair 296. The `resolvent/` directory provides an independent implementation of the published 21-pole protocol; the original gradient driver was unavailable.
