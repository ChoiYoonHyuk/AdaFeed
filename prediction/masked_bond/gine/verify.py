"""Data-free consistency checks for the fixed paired GINE experiment."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys

def verify(root):
    r=json.loads((root/"aggregate.json").read_text(encoding="utf8"))
    assert r["protocol_sha256"]==hashlib.sha256((root/"protocol.md").read_bytes()).hexdigest()
    for f,h in r["source_hashes"].items():
        assert hashlib.sha256((root/f).read_bytes()).hexdigest()==h, f
    assert r["paired_seeds"]==[103,211,307,401,503] and "H100" in r["gpu"]
    assert max(r["compression_test"].values())<1e-12
    fits=r["selection"]
    assert len(fits)==15 and len({(f["seed"],f["arm"]) for f in fits})==15
    assert len({f["parameters"] for f in fits})==1
    for f in fits:
        v=f["validation_curve"]
        assert f["trained_epochs"]==len(v)<=100
        assert f["best_epoch"]==1+v.index(max(v)) and max(v)==f["validation_ap"]
        assert len(v)==100 or len(v)-f["best_epoch"]==15
        assert len(f["checkpoint_sha256"])==64
    lowers=[]
    for name,counts in (("molhiv",(13422,593,3412)),("PCBA",(133325,4362,39221))):
        z=r["outcomes"][name]
        assert tuple(z["counts"][k] for k in ("roots","positive","molecules"))==counts
        for arm,f in z["arms"].items():
            assert abs(statistics.mean(f["seed_ap"])-f["mean_ap"])<1e-12
        for key,d in z["comparisons"].items():
            ref=key.split("Spatial-minus-")[1]
            ds=[p-q for p,q in zip(z["arms"]["Spatial"]["seed_ap"],z["arms"][ref]["seed_ap"])]
            assert max(abs(s-dx) for s,dx in zip(ds,d["seed_differences"]))<1e-12
            avg=statistics.mean(ds); half=2.7764451051977987*statistics.stdev(ds)/math.sqrt(5)
            assert abs(avg-d["delta"])<1e-12 and d["positive_seeds"]==sum(v>0 for v in ds)
            assert max(abs(x-y) for x,y in zip((avg-half,avg+half),d["paired_seed_t95"]))<1e-10
            assert d["paired_molecule95"][0]<=d["paired_molecule95"][1]
            lowers.extend([d["paired_seed_t95"][0],d["paired_molecule95"][0]])
    assert (min(lowers)>0)==r["joint_superiority"]
    print("Validated 15 GINE checkpoints, both known held-out cohorts, four paired contrasts and the joint criterion.")

if __name__=="__main__":
    cli=ArgumentParser(description=__doc__);cli.add_argument("directory",type=Path,nargs="?",default=Path(__file__).parent)
    verify(cli.parse_args().directory)
