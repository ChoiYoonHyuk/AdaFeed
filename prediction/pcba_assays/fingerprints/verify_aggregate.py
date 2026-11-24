"""Independently validate the released aggregate without molecule-level records."""
import argparse
from pathlib import Path
import json
import math
from statistics import mean, stdev
from hashlib import sha256


def close(a,b):
    return math.isclose(a,b,abs_tol=1e-11,rel_tol=0)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("aggregate",type=Path)
    args=cli.parse_args()
    record=json.loads(args.aggregate.read_text())
    fits=record["fit"]; outcomes=fits["official_seed_ap"]
    assert fits["seeds"]==[103,211,307,401,503] and len(set(fits["seeds"]))==5
    assert fits["split_n"]==[350343,43793,43793] and record["eligible_assays"]==127
    assert sum(fits["split_n"])==record["feature_audit"]["context"]["n"]
    assert record["feature_audit"]["fingerprint"]["atom_count_disagreements"]==0
    assert record["feature_audit"]["fingerprint"]["bond_count_disagreements"]==0
    assert set(outcomes)=={"Spatial","Local","Common"} and set(outcomes)==set(record["test_ap"])
    param_counts=set()
    for label,scores in outcomes.items():
        assert len(scores)==5 and all(0<x<1 for x in scores)
        assert close(mean(scores),record["test_ap"][label])
        for fit in fits["checkpoints"][label]:
            param_counts.add(fit["parameters"])
            curve=fit["validation_aps_every_four_epochs"]
            assert fit["seed"] in fits["seeds"] and len(curve)*4==fit["stop_epoch"]
            assert fit["epoch"]==4*(1+curve.index(max(curve))) and close(fit["validation_ap"],max(curve))
            assert (fit["stop_epoch"]-fit["epoch"]==32) if fit["reason"]=="validation_patience" else fit["stop_epoch"]==200
    assert len(param_counts)==1
    for name in ("Local","Common"):
        delta=[a-b for a,b in zip(outcomes["Spatial"],outcomes[name])]
        result=record["spatial_minus"][name]
        assert all(close(a,b) for a,b in zip(delta,result["per_seed"])) and close(mean(delta),result["mean"])
        half=2.7764451051977987*stdev(delta)/math.sqrt(5)
        assert close(mean(delta)-half,result["seed95"][0]) and close(mean(delta)+half,result["seed95"][1])
        assert result["positive_fits"]==sum(x>0 for x in delta)
    assert record["joint_superiority"]==all(record["spatial_minus"][arm][interval][0]>0 for arm in ("Local","Common") for interval in ("seed95","molecule95"))
    for name,expected in fits["source_hashes"].items():
        src=args.aggregate.resolve().parent/name
        if src.is_file():assert sha256(src.read_bytes()).hexdigest()==expected,name
    print("Aggregate validated for 15 paired-seed fits, three arms and both paired contrasts.")
if __name__=="__main__":
    main()
