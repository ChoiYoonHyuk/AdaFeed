"""Verify zero-slot ablation mean differences, conditional seed intervals and lock records."""
from pathlib import Path
from statistics import mean,stdev
import json,sys,math


def close(a,b):
    return math.isclose(a,b,abs_tol=1e-11,rel_tol=0)


def main():
    data=json.loads(Path(sys.argv[1]).read_text());f=data["source_fit_selection"]
    assert data["retrospective"] and f["source_local_seed103_bitwise_replay"] and f["shared_scaler_arrays_equal"]
    assert f["all_primary_checkpoints_locked"]==20 and f["local_raw_slot_exact_static_duplicate"]
    assert all(a["epoch"]==1+a["validation_curve"].index(max(a["validation_curve"])) for v in f["selection"].values() for a in v)
    assert data["seed_multisets"]==126 and data["ordered_seed_draws"]==3125 and data["molecule_reps"]==1000
    assert set(data["datasets"])=={"molhiv","PCBA"}
    for ds in data["datasets"].values():
        assert len(ds["rows"])==4 and ds["previous_primary_replay_max"]<1e-11
        for row in ds["rows"].values():
            assert set(row["arm_ap"])=={"Common","Local","Spatial"}
            for arm in row["arm_ap"].values():
                assert close(mean(arm["per_seed"]),arm["mean"]) and len(arm["per_seed"])==5
            for key,v in row["paired"].items():
                a,b=key.split("-");delta=[i-j for i,j in zip(row["arm_ap"][a]["per_seed"],row["arm_ap"][b]["per_seed"])]
                assert close(mean(delta),v["difference"]) and all(close(i,j) for i,j in zip(delta,v["per_seed"]))
                hw=2.7764451051977987*stdev(delta)/math.sqrt(5)
                assert close(mean(delta)-hw,v["seed95"][0]) and close(mean(delta)+hw,v["seed95"][1])
                assert v["molecule95"][0]<=v["molecule95"][1] and v["crossed95"][0]<=v["crossed95"][1]
    print("Verified all twenty selection records, 24 paired contrasts and fixed-source replay.")
if __name__=="__main__":main()
