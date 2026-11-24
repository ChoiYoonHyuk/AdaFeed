"""One-command review of the construction checks and archived certificate coverage."""
import argparse
from collections import Counter
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from constructions import adjacency, integer_certificates
from word_span import compare, rooted_quotient


HERE = Path(__file__).resolve().parent


def word_controls():
    # Deliberately put every vertex in the same coarse class. Indexed as roots,
    # an endpoint and the middle of P3 differ in (A^2)_ii, 1 versus 2.
    path = adjacency(3,[(0,1),(1,2)]).astype(np.int64)
    colors = np.zeros(3,dtype=np.int64)
    end = rooted_quotient(path,colors,0)
    center = rooted_quotient(path,colors,1)
    opposite = rooted_quotient(path,colors,2)
    if compare(end,center)[0] or not compare(end,opposite)[0]:
        raise ValueError("Paired rooted-word controls failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brec",type=Path,help="Location of the official brec_v3.npy graph6 array")
    parser.add_argument("--workers",type=int,default=2)
    args = parser.parse_args()
    if args.workers<1:
        parser.error("Supply a positive worker count")

    print("Constructed graph checks:",json.dumps(integer_certificates(),sort_keys=True))
    word_controls()
    saved=json.loads((HERE/"brec_results.json").read_text(encoding="utf8"))
    reference=json.loads((HERE/"pair_reference.json").read_text(encoding="utf8"))
    rows=saved["decisions"]
    if [row["id"] for row in rows]!=list(range(400)):
        raise ValueError("The archived BREC table is missing original pair IDs")
    vote=Counter(row["feedback_route"] for row in rows)
    expected={"static":216,"spatial":3,"common_srg":70,"common_coherent":60,"common_colored_words":51}
    if vote!=expected:
        raise ValueError("The archived certificate partition disagrees with the reported values")
    if any(not row["word_check"]["verified"] for row in rows if "word_check" in row):
        raise ValueError("The archive contains an incomplete word certificate")
    positive=[row["id"] for row in rows if row["feedback_route"] in ("static","spatial")]
    degrees={str(row["id"]):row["epwl_positive_degree"] for row in rows if row["epwl_positive_degree"] is not None}
    if sorted(reference["feedback_positive_ids"])!=positive or reference["epwl_positive_degrees"]!=degrees or len(degrees)!=236:
        raise ValueError("Per-ID reference outcomes disagree")
    print("Archived BREC outcome table: 400 IDs;",dict(vote),"; 236 EPWL positive witnesses.")
    epwl_upper=json.loads((HERE/"epwl_upper.json").read_text(encoding="utf8"))
    residual={r["id"] for r in rows if r["feedback_route"]=="common_colored_words" and r["epwl_positive_degree"] is None}
    joint_tied={r["id"] for r in epwl_upper["rows"] if r["upper_certified"]}
    if len(residual)!=34 or residual!=joint_tied or len(epwl_upper["rows"])!=34:
        raise ValueError("The residual three-operator EPWL upper table is incomplete")
    common={r["id"] for r in rows if r["feedback_route"] in ("common_coherent","common_srg")}
    positives=set(map(int,degrees))
    if common & positives or joint_tied & positives or len(common|joint_tied|positives)!=400:
        raise ValueError("EPWL coverage does not partition the 400 original pairs")
    print("Archived three-operator EPWL coverage: 236 positive, 130 common-algebra and 34 explicit spectral ties.")
    if args.brec:
        from brec_verify import verify
        rebuilt=verify(args.brec,args.workers)
        if rebuilt!=saved:
            raise ValueError("Recomputed BREC tables differ from the packaged per-ID results")
        print("Recomputed and matched all 400 stabilized-feedback decisions from the input graph records.",flush=True)
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)
            workers=str(args.workers)
            output=work/"upper.json"
            subprocess.run([sys.executable,"-B",str(HERE/"epwl_upper.py"),str(args.brec),str(HERE/"brec_results.json"),
                            str(output),"--workers",workers],check=True)
            if json.loads(output.read_text(encoding="utf8"))!=epwl_upper:
                raise ValueError("Three-operator EPWL results differ from archived IDs or stable classes")
            subprocess.run([sys.executable,"-B",str(HERE/"wl_bench.py"),str(args.brec),str(output)],check=True)
            if json.loads(output.read_text(encoding="utf8"))!=json.loads((HERE/"wl_bench.json").read_text(encoding="utf8")):
                raise ValueError("1-WL/EA audit differs from the archived outcomes")
        print("Recomputed all 400 three-operator EPWL decisions and the 1-WL/EA comparisons.")
    else:
        print("Use --brec /path/to/brec_v3.npy on the computer holding BREC to recompute its certificates.")


if __name__=="__main__":
    main()
