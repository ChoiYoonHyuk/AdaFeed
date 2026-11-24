# Normalized diagonal feedback on BREC

[Protocol](protocol.md), [fixed encoder](normalized_brec.py), [aggregate](aggregate.json), [independent verifier](verify_static_and_residues.py), [verification results](static_verification.json).

On the machine holding the benchmark, install NumPy, SciPy and NetworkX and run the commands below from this directory. The input is the official uncompressed BREC v3 graph6 array, SHA-256 6975d172d27aedcdf4eb2c747450ede7928c111caeb2689fbc67e6b62fbd5562. Both programs check it before decoding pairs. No graph record or fitted feature matrix is part of these results.

~~~sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python normalized_brec.py /server/path/brec_v3.npy --output aggregate_recomputed.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python verify_static_and_residues.py /server/path/brec_v3.npy --output static_verification_recomputed.json
~~~

The fixed bank uses 24 power diagonals per query. Its two feedback rounds use the same fixed signed 24-term combination of the preceding return bank divided by 46. The row walk has absolute row sums at most one. Consequently its power diagonals have absolute value at most one, as do the gate and the later halved operators in the induced infinity norm. For an isolate the base row is zero.

On one CPU process on the H100 server, the cumulative numbers of pairs distinguished were 216, 219 and 219 after one, two and three queries. Each prime independently yielded the same first-detection round for all 400 pairs. The three additions are CFI pairs 273, 296 and 307. The aggregate records family counts, pair IDs and wall time. The timing excludes the initial source hash and measures the integer certificate implementation.

The unweighted sparse product sums at most \(n\) residues per output entry. The code requires \(n<p\) for each prime \(p\), so each unscaled sum is below \(p^2\). Before modular reduction a diagonal multiplication or multiplication by an inverse is below \(p^2\); combining two reduced terms with a diagonal product stays below \(2p^2\). A gate dot product is bounded by \(46(p-1)\). All these bounds fit in signed 64-bit integers at the two primes. The only rational divisions are by positive vertex degrees, 46, and powers of two. Both prime moduli exceed every degree and are coprime to these denominators. Equal rational multisets reduce to equal modular multisets. A modular difference therefore certifies a genuine rational difference.

[Archived positive IDs](certified_ids.json) originate from the complete adjacency-feedback certificates. The two-way normalized-base simulation gives the same 219-positive upper bound for normalized feedback. Matching all 219 closes the finite feedback-bank result. The separate verifier computes exact powers of the integer matrix \(L P\) on the three additional CFI pairs, using one common degree LCM \(L\) for both graphs in each pair. It matches root-profile multisets through order \(n_L+n_R-1\). By the common characteristic recurrence of the two graphs' direct sum, these profiles agree at all orders. The three pairs have 16, 10, and 9 distinct static profiles on each side. Together with the 181 full-interface ties and the 216 finite static lower certificates, these checks also close the complete normalized-static result on BREC. Independent Fraction evaluation on two small graphs tests every implemented first, second and third query history at both primes.

