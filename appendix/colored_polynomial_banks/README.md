# Two adaptive polynomial queries on roadNet-PA

Download the Pennsylvania road network only onto the benchmark server. The data author page is [SNAP roadNet-PA](https://snap.stanford.edu/data/roadNet-PA.html). The SHA-256 in the protocol pins the compressed file. The graph has 1,088,092 nonisolated vertices and 1,541,898 simple undirected edges. The experiment uses three induced nested BFS prefixes with degrees recomputed inside each prefix.

The implementation requires C++17, NumPy, pandas, SciPy, psutil and CUDA PyTorch. Compile the colorer once. Run from the source folder with an empty server-local output directory. The fixed-slot wrapper starts a single independent largest-case measurement at the declared UTC windows. It does not wait or schedule itself.

```sh
curl --fail --location https://snap.stanford.edu/data/roadNet-PA.txt.gz --output /server/data/roadNet-PA.txt.gz
g++ -std=c++17 -O3 -fPIC -shared color.cpp -o color.so
mkdir -p /server/work/road_probing
python probe.py /server/data/roadNet-PA.txt.gz /server/work/road_probing
python repeat.py 0905 /server/data/roadNet-PA.txt.gz /server/work/road_probing
python repeat.py 0942 /server/data/roadNet-PA.txt.gz /server/work/road_probing
```

A distance-k color class supplies one probe column with a 1 on every member of the class. No two such members can reach each other in at most k edge steps. A diagonal self-loop uses one step without changing location. Reading the propagated column at a member therefore gives that member's diagonal return at each order through k in either bank. The second potential is computed from the implementation's own first-bank returns. Columns for both methods are packed in blocks of width 64. A separate exact-rational toy check uses the diagonally similar random-walk kernel, and requires only the Python standard library.

The initial 65,536-root order-24 comparison used 1,349 colored columns. The GPU-resident panel took 0.165 s with coloring and 7.299 s with individual root columns. From the compressed file on disk to a synchronized result on GPU, the first call took 3.006 and 8.042 s, respectively. Preparing the coloring cost 2.097 s, while its separate verification cost a further 1.966 s. Both peak tensor allocations were 158.42 MiB. At the two smaller order-24 cases, first-call end-to-end costs were close. These timings condition on graph, order, machine and the documented cache scope.

`aggregate.json` stores every trial. The two time-separated repeats are in `repeat_0905.json` and `repeat_0942.json`. `python verify_probing.py .` checks the aggregate and every repeat file present alongside these sources without downloading road data. The server retains the road file, compiled binary, and any per-vertex arrays. Only code, protocol and aggregate summaries leave the server.

At 09:05 and 09:42 UTC, the two prespecified colored-bank repetitions produced resident medians of 0.164701 and 0.164508 s. The corresponding fresh file-to-GPU times were 3.096897 and 3.112633 s, including 2.096283 and 2.102644 s of CPU coloring. Independent whole-graph color checks took a further 1.978404 and 1.980486 s. Both runs reproduced the same 1,349 colors, peak allocation of 158.418 MiB, and maximum sampled return discrepancy of 1.11e-16, with no other compute context observed immediately before or after either timed case. The repeats did not remeasure the identity-column baseline.
