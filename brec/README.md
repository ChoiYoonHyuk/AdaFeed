# Exact graph certificates

Run `python -B verify.py` in the extracted directory. This offline check needs Python 3.11+ and NumPy. It constructs the witness graphs from their edge rules, recomputes their results, tests the word-comparison routine on positive and negative controls, and checks the archived pair-ID lists for completeness and agreement. No dataset is bundled.

To **recompute BREC from the original graphs**, add NetworkX and SciPy and run the following on the computer holding the [official BREC release](https://github.com/GraphPKU/BREC/tree/Release). The argument is the path to its uncompressed `brec_v3.npy`. The released array has SHA-256 `6975d172d27aedcdf4eb2c747450ede7928c111caeb2689fbc67e6b62fbd5562`. The verifier checks this before selecting graph6 entries `64*i` and `64*i+1` for original pair IDs 0 through 399.

```sh
python -B verify.py --brec /path/to/brec_v3.npy --workers 4
```

The command rebuilds and compares all per-ID decision records and the 1-WL controls. Each benchmark checker has a command-line help option. The constructed graph calculations use arbitrary-precision integers. The benchmark verifier uses bounded int64 residues for separating observations, arbitrary-precision integers for word spans and exact spectral ties, and integer relation tables for coherent configurations.

## Constructed graph checks

| Input | Recomputed quantity |
| --- | --- |
| Rook and Shrikhande graphs with 1, 2 and 5 leaves | Fifth-return weighted scores 5382/5370, 5418/5406 and 5526/5514 |
| Four attached cores in two arrangements on a cycle (68 vertices each) | Scores 35568/35280 after three queries |
| Two voltage covers of orders 3, 4, 5 and 7 | Midpoint counts 1180k and 1176k, including 3540/3528 for the 45-vertex pair |
| Direct products of the 45-vertex graphs with a three-vertex path | Sixth traces 64530216/64530360 |
| Six-type Fourier core with arbitrary diagonal variables | Coefficientwise comparison of all 64 principal minors over integer Laurent polynomials |

In the symbolic check a monomial records the subset of diagonal variables and the exponent of the cyclic voltage. The six independent diagonal variables permit arbitrary class-diagonal shifts after eliminating pendant vertices. The manuscript gives the resolvent argument from these minor identities and the induction for the general depth hierarchy. The finite-cycle computation exercises the concrete three-query instance of that hierarchy.

## BREC coverage

| Decision | Certificate route | Pair count |
| --- | --- | ---: |
| Feedback separates | Static adjacency moments already differ | 216 |
| Feedback separates | A shared field based on the first query produces a different second response | 3 |
| Feedback equivalent at all depths | Common strongly regular parameters | 70 |
| Feedback equivalent at all depths | Matching coherent configurations with verified intersection numbers | 60 |
| Feedback equivalent at all depths | Equal class-colored word diagonals for every root | 51 |
| EPWL separates | Jointly encoded adjacency-power pair entries and node refinement | 236 |
| EPWL tied | Common strongly regular or coherent certificate above | 130 |
| EPWL tied | Exact spectral pair labels for adjacency and both Laplacians, followed by refinement to stability | 34 |

The first five disjoint rows partition the 400 feedback decisions. The last three rows independently partition the same 400 pairs for joint three-operator EPWL. The 34 explicit EPWL ties belong to the 51 cases certified for feedback by colored words.

### Separating observations and common algebras

`pair_reference.json` gives archived pair IDs and the maximum power needed for each EPWL-positive certificate. `brec_verify.py` recomputes the 24 static diagonal moments, the second shifted bank, and the specified adjacency-power entries modulo 1,000,003. An unequal residue proves an exact integer mismatch. The field is a fixed shared function of each node's first observed profile. The candidate phase for coherent configurations uses codes to propose relation classes. Acceptance requires exact joint class sizes, closure under transpose and identical full intersection-number vectors on all members of each proposed relation, including members in the other graph. The common algebra then also contains the degree matrix, Laplacian, normalized Laplacian and their spectral projectors.

### The remaining all-depth feedback equivalences

`word_span.py` proposes a shared vertex partition from finite shifted profiles. Each accepted pair has equal class sizes. For every root, individualize that vertex, refine to an equitable partition and form the exact neighbor-count quotient. The root is a singleton, so its unit vector and all subsequent vectors in a colored walk have constant values on quotient cells. The checker verifies that the quotient lifts each adjacency transition and each class indicator back to the original cells.

Within each common class, compare every root to one anchor on either side. Identical complete rooted quotients agree immediately. Otherwise start from the pair of root unit vectors and close their span over the rationals under adjacency followed by projection to every destination class. Each basis vector is supported on one class, so the sum of these transitions gives adjacency and class projection acts on the basis by identity or zero. Integer fraction-free elimination finds a basis. The two root coordinates must agree on every basis element. This proves equality of all rooted word diagonals in adjacency and the class indicators. Every class-dependent shifted power is a linear combination of these words. Spectral interpolation gives the same result for primary filters. Shared updates therefore maintain the common partition at all depths. This is an independent route to the manuscript's 51 equivalences originally established with intertwiners. The individual records appear in `brec_results.json`.

### Exact spectral pair refinement and WL controls

`epwl_upper.py` uses power traces through graph order n to compare spectra and pair entries through n−1 to determine the labeled projectors by interpolation. Degree-pair labels and a common integer multiple of the random-walk operator permit the normalized-adjacency comparison over the integers. On nonisolated vertices the normalized adjacency is similar to the random-walk matrix under diagonal scaling by square-root degree. The endpoint degrees therefore recover the required pair scaling. The walk matrix has a zero row at every isolate. The joint pair labels are then used in exact common node-color refinement to stability. `epwl_upper.json` provides per-pair results. The 1-WL and eigenvalue-angle checks in `wl_bench.py` reproduce the graph-level and node-level controls, the single-class CFI perturbation and the three decorated witnesses. `wl_bench.json` stores these results.