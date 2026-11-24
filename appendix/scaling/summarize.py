"""Reproduce the wall-clock medians, peak MiB, hashes and timeout counts."""
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

base = Path(__file__).resolve().parent
all_trials = defaultdict(list)
for name in ("aggregate.json", "large_pairs.json"):
    path = base / name
    if not path.is_file():
        continue
    report = json.loads(path.read_text())
    print("checks", name, report["checks"])
    for row in report["records"]:
        all_trials[row["n"],row["method"]].extend(row["trials"])
print("n method repeats median_s torch_MiB nvml_MiB rss_MiB projectors/rounds/classes or progress")
for (n,method),trials in sorted(all_trials.items()):
    med = statistics.median(t["seconds"] for t in trials)
    high = [max(t[k] for t in trials) for k in ("torch_peak_MiB", "device_delta_MiB", "host_delta_MiB")]
    last = trials[-1]
    print(n,method,len(trials),f"{med:.5f}",*[f"{v:.3f}" for v in high],
          last.get("projectors",""),last.get("rounds",""),last.get("classes",""),last.get("timeout",""))
for name in ("aggregate.json", "large_pairs.json", "prepare_graphs.py", "scale.py", "large_pairs.py",
             "protocol.txt", "followup_protocol.txt"):
    file = base / name
    if file.is_file():
        print(hashlib.sha256(file.read_bytes()).hexdigest(), name)
