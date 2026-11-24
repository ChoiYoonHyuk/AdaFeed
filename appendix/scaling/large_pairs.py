"""Follow-up to the frozen initial sweep. Try the whole pair pipeline at larger n."""
import json
import torch
import pynvml
from scale import BASE, PAIR_SECONDS, check, handle, load_graph, measure, other_processes, pairwl

def main():
    torch.set_num_threads(1)
    assert not other_processes(), "GPU already in use"
    # 16K explicit pair-work arrays peak at approximately 25 GiB with this kernel.
    assert pynvml.nvmlDeviceGetMemoryInfo(handle).free > 40*(2**30)
    result={"checks":check(),"followup":True,"seconds_budget":PAIR_SECONDS,"records":[]}
    for n, repeats in [(2048,2),(4096,1),(8192,1),(16384,1)]:
        data=load_graph(n)
        trials=[]
        for rep in range(repeats):
            assert not other_processes()
            trial=measure(lambda g:pairwl(g,PAIR_SECONDS),"EPWL",data)
            trials.append(trial)
            print(n,rep,trial,flush=True)
        result["records"].append(dict(n=n,method="EPWL",trials=trials))
        path=BASE/"large_pairs.json"
        path.with_suffix(".tmp").write_text(json.dumps(result,indent=2))
        path.with_suffix(".tmp").replace(path)
        del data
    (BASE/"large_finished").write_text("ok\n")

if __name__=="__main__":
    main()
