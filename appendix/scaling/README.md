# Single-graph scaling benchmark

The two-query encoder, dense adjacency eigendecomposition, and numerical fixed-adjacency EPWL are timed from a shared resident CSR adjacency. The paired BREC decision benchmark uses separate code in the paper.

Run the following on an exclusive CUDA device from the source directory. The input is the original ogbn-arxiv raw edge CSV from the Open Graph Benchmark archive. Data preparation creates induced citation subgraphs only on the machine where it runs.

```sh
python prepare_graphs.py /path/to/arxiv/raw/edge.csv.gz
python scale.py
python large_pairs.py
python summarize.py
```

The initial sweep and follow-up decisions are documented in protocol.txt and followup_protocol.txt. The raw per-trial output files contain times, extra live PyTorch allocations, and 5 ms samples of device-memory and host-RSS increases. Floating-point consistency checks include a 128-node independent dense eigensystem and vertex relabeling. Timings describe construction of three different within-graph outputs. They exclude graph preparation, cross-graph comparison, and training.

The measured stack used Linux, NVIDIA H100 PCIe, CUDA 12.8, PyTorch 2.11.0, NumPy, SciPy, psutil, and the nvidia-ml-py bindings for the GPU memory sampler. The data file is identified by its SHA-256 in aggregate.json. Source graph IDs, edges and node labels are not distributed here.
