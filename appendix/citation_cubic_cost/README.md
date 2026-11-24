# Timing the citation-graph cubic contrast

From the public `citation_cubic_cost` directory, pass the official ogbn-arxiv ZIP and a new output filename. Use NumPy, pandas, SciPy and CUDA PyTorch on an H100. Keep [PROTOCOL.md](PROTOCOL.md) beside the script.

```sh
mkdir -p /server/work/citation_cubic_cost
python cubic_cost.py /server/data/arxiv.zip /server/work/citation_cubic_cost/aggregate.json
```

The output gives the synchronized timing trials, memory measurements and comparisons on two induced prefixes and the full graph. The retained `aggregate_repeat.json` records the dedicated-device measurements reported in the appendix.
