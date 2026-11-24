"""Verify integrity, protocol coverage, numerical checks and stable per-run partitions."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

base = Path(__file__).resolve().parent
for line in (base / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    expected, name = line.split("  ", 1)
    actual = hashlib.sha256((base / name).read_bytes()).hexdigest()
    assert actual == expected, name

sizes = (512, 1024, 2048, 4096, 8192, 16384)
trials = defaultdict(list)
for name in ("aggregate.json", "large_pairs.json"):
    report = json.loads((base / name).read_text())
    checks = report["checks"]
    assert checks["evd_bank_max_abs_error"] < 1e-10
    assert checks["numerical_partition_permutation_match"]
    for row in report["records"]:
        trials[row["n"], row["method"]].extend(row["trials"])

for n in sizes:
    for method in ("feedback", "EVD", "EPWL"):
        rows = trials[n, method]
        expected = 1 if method == "EPWL" and n > 2048 else 3
        assert len(rows) == expected
        assert all(t["seconds"] > 0 and t["torch_peak_MiB"] > 0 for t in rows)
        if method == "EVD":
            assert all(t["trace_square_error"] < 1e-5 for t in rows)
        if method == "EPWL":
            completed = [t for t in rows if "timeout" not in t]
            partitions = {(t["classes"], t["rounds"], t["projectors"]) for t in completed}
            assert len(partitions) <= 1
            assert all(t["seconds"] >= 45 for t in rows if "timeout" in t)
print("Verified manifest, all 18 size-method cells, run counts and numerical checks.")
