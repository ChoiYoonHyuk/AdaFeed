"""Run the released BREC verifier unchanged and summarize completed checks.

Usage: python -B replay.py anonymous-certificates.zip /server/path/brec_v3.npy new_run_directory
Run on the data host. Only aggregate.json should be exported from the new directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import time
from zipfile import ZipFile


ARCHIVE_SHA256 = "c5364ff0f7b76bd5cbba7d57aa0902540f82c3d1e5541c71bba2e9ed744d2ec1"
DATA_SHA256 = "6975d172d27aedcdf4eb2c747450ede7928c111caeb2689fbc67e6b62fbd5562"
FINAL_FEEDBACK = "Recomputed and matched all 400 stabilized-feedback decisions from the input graph records."
FINAL_EPWL_WL = "Recomputed all 400 three-operator EPWL decisions and the 1-WL/EA comparisons."


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for label in ("archive", "official_array", "new_run_directory"):
        p.add_argument(label, type=Path)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    start = time.monotonic()
    run = args.new_run_directory.resolve()
    run.mkdir(parents=True, exist_ok=False)
    scratch = run / "temporary"
    scratch.mkdir()
    env = dict(os.environ, TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    result = {"start_utc": utc(), "checker_invocation": "python -B -u verify.py --brec <official-array> --workers " + str(args.workers),
              "workers": args.workers, "execution": "low-priority CPU on H100 host; single-threaded numerical libraries",
              "python": platform.python_version(), "runner_sha256": sha(Path(__file__)),
              "archive_sha256": sha(args.archive), "official_array_sha256": sha(args.official_array)}
    assert result["archive_sha256"] == ARCHIVE_SHA256 and result["official_array_sha256"] == DATA_SHA256
    tools = run / "released"
    tools.mkdir()
    with ZipFile(args.archive) as zf:
        for name in zf.namelist():
            member = PurePosixPath(name)
            assert not member.is_absolute() and ".." not in member.parts and "\\" not in name
        zf.extractall(tools)
    manifest = {}
    for entry in (tools / "SHA256SUMS").read_text().splitlines():
        expected, name = entry.split(maxsplit=1)
        assert sha(tools / name) == expected, name
        manifest[name] = expected
    result["released_file_sha256_verified"] = manifest
    log = run / "server_only.log"
    checked = time.monotonic()
    with log.open("x") as stream:
        cmd = [sys.executable, "-B", "-u", str(tools / "verify.py"), "--brec", str(args.official_array.resolve()),
               "--workers", str(args.workers)]
        test = subprocess.run(cmd, cwd=tools, env=env, stdout=stream, stderr=subprocess.STDOUT)
    result["checker_duration_seconds"] = round(time.monotonic() - checked, 3)
    result["checker_exit_code"] = test.returncode
    text = log.read_text()
    conditions = {"constructed_graph_certificates": "Constructed graph checks:" in text,
                  "all_400_per_id_feedback_objects": FINAL_FEEDBACK in text,
                  "all_400_epwl_and_wl_ea_objects": FINAL_EPWL_WL in text}
    result["completed_checks"] = conditions
    result["passed"] = test.returncode == 0 and all(conditions.values())
    if result["passed"]:
        def read(name):
            return json.loads((tools / name).read_text())
        feedback = read("brec_results.json")
        upper = read("epwl_upper.json")
        reference = read("pair_reference.json")
        wl = read("wl_bench.json")
        result["feedback_routes"] = feedback["feedback_routes"]
        result["feedback_positive"] = len(reference["feedback_positive_ids"])
        result["feedback_negative"] = feedback["pair_count"] - result["feedback_positive"]
        result["epwl_positive"] = feedback["epwl_positive_ids_verified"]
        result["epwl_common_algebra_ties"] = feedback["feedback_routes"]["common_srg"] + feedback["feedback_routes"]["common_coherent"]
        result["epwl_explicit_other_ties"] = upper["upper_certified_count"]
        result["wl_controls"] = {k: wl[k] for k in ("range", "all", "isomorphic_controls", "independent_pair_relabelings", "relabel_failures")}
    result["checker_log_sha256"] = sha(log)
    result["end_utc"] = utc()
    result["duration_seconds"] = round(time.monotonic() - start, 3)
    with (run / "aggregate.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not result["passed"]:
        raise RuntimeError("Unchanged BREC checker did not finish successfully; inspect the server-side log.")


if __name__ == "__main__":
    main()
