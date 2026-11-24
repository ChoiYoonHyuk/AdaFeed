"""Rerun BREC outcomes after independent vertex relabelings and check isomorphic negatives."""
from pathlib import Path
import sys
import json
import brec_stable_cost as bench
import numpy as np

data_file, bench_file, out_file = map(Path, sys.argv[1:])
assert not out_file.exists()
old = json.loads(bench_file.read_text())
graphs = bench.read_data(data_file)
rng = np.random.default_rng(9126002)
checks = {m: {"false_positive_iso": [], "changed_after_relabel": []} for m in bench.METHODS}
for idx, (a,b) in enumerate(graphs):
    za,zb = rng.permutation(len(a)), rng.permutation(len(b))
    ra,rb = a[np.ix_(za,za)], b[np.ix_(zb,zb)]
    for m, method in bench.METHODS.items():
        if method([a,ra])["decision"]:
            checks[m]["false_positive_iso"].append(idx)
        if method([ra,rb])["decision"] != (idx in old["positives"][m]):
            checks[m]["changed_after_relabel"].append(idx)
    if (idx+1)%100 == 0:
        print("checked", idx+1,flush=True)
out_file.write_text(json.dumps(checks, indent=2)+"\n")
print(checks)
