"""Check the released aggregate report without the graphs. See count_cycles.py for a full rerun."""
from hashlib import sha256
import json
from math import isclose
from pathlib import Path

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / "summary.json").read_bytes())
assert summary["dataset_sha256"] == "758ae12bd05ced3c1ba609d81f0d0dbdc1d227d486e3f2c1c517b47fe7dad9b4"
assert summary["commit"] == "f514487625b42406dcd85ca55ac66436606f9aac"
for field, filename in [("protocol_sha256", "protocol.md"), ("program_sha256", "count_cycles.py")]:
    assert summary[field] == sha256((HERE / filename).read_bytes()).hexdigest()
assert "H100" in summary["gpu"]
assert summary["target_index"] == 1
assert summary["fits"]["spatial"]["train_rank"] == 7
assert summary["fits"]["root_local"]["train_rank"] == 6

sd = summary["training_plus_validation_target_sd"]
for name, graphs, roots in [("train", 1500, 28485), ("validation", 1000, 18820),
                            ("test", 2500, 46695)]:
    s = summary["splits"][name]
    assert (s["graphs"], s["roots"]) == (graphs, roots)
    assert s["formula_exact_roots"] == roots
    assert s["formula_mae"] == s["neighbor_control_mae"] == s["omit_delta_graph_total_mae"] == 0
    assert s["uniform_contrast_absmax"] == s["weighted_identity_absmax"] == 0
    assert s["spatial"]["ols_mae"] < 1.e-10
    for arm in ("spatial", "root_local"):
        assert isclose(s[arm]["ols_mae"], sd*s[arm]["normalized_mae"], abs_tol=1e-12)

test = summary["splits"]["test"]
sp, rl = test["spatial"], test["root_local"]
assert (rl["groups"], rl["ambiguous_groups"], rl["roots_in_ambiguous_groups"]) == (756,508,41322)
assert (sp["groups"], sp["ambiguous_groups"], sp["within_tuple_min_mae"]) == (1619,0,0)
assert isclose(rl["within_tuple_min_mae"], rl["min_sum_absolute_error"]/test["roots"])
assert isclose(rl["ols_mae"]-sp["ols_mae"], test["paired_node_mae_improvement"], abs_tol=1.e-12)
lo,hi = test["paired_graph_interval"]
assert lo > 0 and lo < test["paired_node_mae_improvement"] < hi
assert isclose(rl["within_tuple_min_mae"], .24490844844201734)
assert isclose(rl["ols_mae"], .4709260792164731)
assert isclose(lo,.46597509719668173) and isclose(hi,.47602671483416664)
assert test["positive_roots"] == 29142 and test["target_sum"] == 106916
assert test["nonzero_contrast_roots"] == 42454 and test["nonzero_edge_energy_graphs"] == 2500
print("Verified the I2-GNN four-cycle aggregate, source hashes, paired errors and finite-bank bounds.")
