# Timing the distance-three molecular path weight

From the public `molecular_shell_cost` directory, pass the extracted OGB molhiv directory and a new output filename. This H100 study uses NumPy, pandas, SciPy and CUDA PyTorch. Numba accelerates the independent CPU traversal when installed.

```sh
mkdir -p /server/work/molecular_shell_cost
python shell_cost.py /server/ogbg_molhiv /server/work/molecular_shell_cost/aggregate.json
```

Keep [PROTOCOL.md](PROTOCOL.md) beside the script. The output records the all-root CPU comparison, intermediate sparse matrix sizes, and five synchronized GPU trials.
