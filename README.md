# Adaptive Diagonal Feedback and the Limits of Homomorphism Counts

## Files

```text
experiments/
  brec/         Graph-pair separation on BREC (Section 6.1)
  four_cycle/   Rooted four-cycle counts on I²-GNN (Section 6.2)
  four_clique/  Rooted four-clique counts on six graph families (Section 6.3)
  appendix/     Exact checks, numerical comparisons, and timing experiments
  prediction/   Masked-bond, PCBA assay, and carbon-shift experiments
  targets/      Molecular property, citation, and airport prediction
```

## Run

From the extracted directory, check the published main-text results with Python 3.11 and NumPy:

```bash
python -m pip install numpy==1.26.4
python experiments/brec/verify.py
python experiments/four_cycle/verify_counts.py
python experiments/four_clique/verify_catalogue.py
```

To rerun on the benchmark server, set `DATA` to the directory containing the inputs below. The four-cycle run uses CUDA. The four-clique run uses CPU PyTorch. Choose new output filenames.

```bash
python -m pip install networkx==3.3 scipy
# Use the server's CUDA-enabled PyTorch installation for the four-cycle run.
python experiments/brec/verify.py --brec "$DATA/brec_v3.npy" --workers 4
python experiments/four_cycle/count_cycles.py --data "$DATA/data.mat" --out cycle_run.json
python experiments/four_clique/srg_counts.py "$DATA/catalogue"
python experiments/four_clique/learned_catalogue.py "$DATA/catalogue" clique_run.json
python experiments/four_clique/verify_catalogue.py clique_run.json --cache "$DATA/catalogue"
```

`srg_counts.py` prepares the six graph6 files on the server and runs the Appendix F calibration. The original aggregation script, `experiments/four_clique/summarize_catalogue.py`, rebuilds Table 4 from the recorded fits. Appendix inputs, environments, and commands are listed in the [numerical experiments](experiments/appendix/RUN.md), [masked-bond, PCBA, and carbon-shift experiments](experiments/prediction/RUN.md), and [molecular and node prediction experiments](experiments/targets/RUN.md).
