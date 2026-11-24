# BREC wall-clock comparison

Run these commands on a Linux compute server that already holds the public BREC graph6 array. Set BREC_NPY to that file. Each command expects a fresh output filename. The need for a fresh filename prevents accidental overwrites of prior measurements.

```sh
python -B brec_stable_cost.py "$BREC_NPY" new-timing.json
python -B verify_stable.py "$BREC_NPY" new-timing.json new-relabel.json
python -B verify_pair_witnesses.py "$BREC_NPY" new-timing.json new-witnesses.json
```

Python 3.12.7, NumPy 1.26.4, SciPy 1.15.1, NetworkX 3.3 and threadpoolctl were used in the reported run. The timing script pins itself to one available CPU core and limits native thread pools. It compares all 400 original pairs with three interleaved passes. The scripts consume the BREC data in place and write aggregate timings, dataset indices, and verification outcomes. If `pair_diagnostics.json` from the earlier feedback study is available alongside the BREC array, the timing script also checks its outputs against the modular feedback decisions.

- `stable-results.json` records the timed run. Full times are median pass totals, and per-pair quantiles are calculated after taking the median across repeats.
- `stable-relabel.json` records false positives on isomorphic controls and changes under independent vertex permutations. An empty list means that no test failed.
- `certified-pairs.json` records the last included adjacency power needed to certify each floating positive under the script's power checkpoints. A witness establishes a positive using exact arithmetic modulo 1,000,003. The graph equivalence certificates and full upper bounds are described in the paper.

## ZINC timing control

The molecular timing script reads the official `molecules.zip` and `test.index` on the same compute server. Torch is needed to read tensors in the release pickle. Each of the 1,000 selected ZINC-12k test graphs is paired with its own relabeling. Timers include the same graph-feature construction and refinement as the BREC run. Molecular property labels are unused.

```sh
python -B zinc_cost.py "$ZINC_ARCHIVE" "$ZINC_TEST_INDEX" new-zinc-time.json
```
