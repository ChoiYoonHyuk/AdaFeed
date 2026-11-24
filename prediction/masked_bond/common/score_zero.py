"""Retrospective Common/Local/Spatial AP with seed, molecule and crossed uncertainty."""
from pathlib import Path
from itertools import combinations_with_replacement
from collections import Counter
from statistics import mean,stdev
import argparse,json,sys,math,os
import numpy as np
from scipy.stats import t
from sklearn.metrics import average_precision_score
import torch
from fit_zero import zeroed,atom_arrays,digest


def crossed(delta):
    # Rows pair the same trained source seed in all arms. Columns pair a whole-molecule draw.
    nseeds,ndraws=delta.shape
    assert nseeds==5 and ndraws==1000
    points=[];mult=[]
    for ix in combinations_with_replacement(range(nseeds),nseeds):
        counts=np.bincount(ix,minlength=nseeds)
        multiplicity=math.factorial(nseeds)//math.prod(math.factorial(int(x)) for x in counts)
        points.append((counts@delta)/nseeds);mult.append(multiplicity)
    assert len(mult)==126 and sum(mult)==3125
    values=np.concatenate(points)
    weights=np.repeat(mult,ndraws)
    order=np.argsort(values,kind="stable");cum=np.cumsum(weights[order])
    return [float(values[order[np.searchsorted(cum,q*cum[-1],side="left")]]) for q in (.025,.975)]


def contrast(hi,lo):
    delta=hi-lo
    seed=delta[:,0];avg=float(seed.mean());half=float(t.ppf(.975,4)*seed.std(ddof=1)/np.sqrt(5))
    return dict(difference=avg,per_seed=seed.tolist(),positive_seeds=int((seed>0).sum()),
            seed95=[avg-half,avg+half],molecule95=np.quantile(delta[:,1:].mean(0),[.025,.975]).tolist(),
            crossed95=crossed(delta[:,1:]))


def main():
    cli=argparse.ArgumentParser()
    for k in ("source_models","base","transfer","fits","out"):cli.add_argument(k,type=Path)
    a=cli.parse_args()
    if a.out.exists():raise RuntimeError("Existing analysis")
    sys.path.insert(0,str(a.source_models));sys.path.insert(0,str(a.base));sys.path.insert(0,str(a.transfer))
    from score_frozen_official import grouped
    import distance_study as study
    from evaluate import aps
    torch.set_num_threads(1);torch.set_float32_matmul_precision("high");torch.cuda.set_per_process_memory_fraction(.12)
    original=torch.load(a.source_models/"selected_server_only.pt",map_location="cpu",weights_only=False)
    new=torch.load(a.fits/"common_models_server_only.pt",map_location="cpu",weights_only=False)
    locked=json.loads((a.fits/"fit_selection.json").read_text())
    assert locked["new_states_sha256"]==digest(a.fits/"common_models_server_only.pt") and locked["all_primary_checkpoints_locked"]==20
    output={"retrospective":True,"source_fit_selection":locked,"score_script_sha256":digest(__file__),
        "seeds":list(study.SEEDS),"seed_multisets":126,"ordered_seed_draws":3125,"molecule_reps":1000,"datasets":{}}
    for split in ("molhiv","PCBA"):
        if split=="molhiv":
            data=atom_arrays(a.base,a.source_models);eligible=data["carbon"]&(data["split"]==2)
            pri=json.loads((a.source_models/"aggregate.json").read_text());bootstrap_seed=180421
        else:
            data=dict(np.load(a.transfer/"cache/distance_destination_server_only.npz"));eligible=np.ones(len(data["y"]),bool)
            pri=json.loads((a.transfer/"aggregate.json").read_text());bootstrap_seed=180523
        outrows={};maxerr=0.
        for d in (3,2):
            ix=np.flatnonzero(eligible&(data["dist"]==d));y=data["y"][ix]
            group,wt=grouped(data["graph"][ix],bootstrap_seed)
            for sens in ("shell_count","common"):
                key=f"d{d}/{sens}";arm_points={};draws={}
                for arm in ("Common","Local","Spatial"):
                    if arm=="Common":
                        slot=zeroed(study,data,d,sens,ix);states=new["states"][key]
                        mu,sd=new["scalers"][key]
                    else:
                        refkey=key+"/"+arm.lower()
                        slot=study.take(data,d,sens,arm.lower(),ix);states=original["states"][refkey]
                        mu,sd=original["scalers"][refkey]
                    x=study.to_gpu(study.standard(slot,mu,sd));pp=[]
                    for i,state in enumerate(states):
                        m=study.predictor(x.shape[1]);m.load_state_dict(state);m.eval()
                        pred=study.score(m,x)
                        vals=aps(pred,y,group,wt)
                        assert abs(average_precision_score(y,pred)-vals[0])<1e-11
                        pp.append(vals)
                    block=np.stack(pp);draws[arm]=block
                    arm_points[arm]={"mean":float(block[:,0].mean()),"per_seed":block[:,0].tolist()}
                pairs={k:contrast(draws[h],draws[l]) for k,(h,l) in {
                    "Spatial-Common":("Spatial","Common"),"Local-Common":("Local","Common"),"Spatial-Local":("Spatial","Local")}.items()}
                ref=pri["results"][key]
                if split=="molhiv":
                    previous=(ref["difference"],ref["paired_molecules"]["percentile95"],ref["paired_seed_t95"])
                else:previous=(ref["difference"],ref["molecules"]["percentile95"],ref["paired_seed_t95"])
                row=pairs["Spatial-Local"]
                values=[row["difference"],*row["molecule95"],*row["seed95"]]
                maxerr=max(maxerr,max(abs(np.array(values)-np.array([previous[0],*previous[1],*previous[2]]))))
                outrows[key]={"roots":len(ix),"positive":int(y.sum()),"molecules":len(np.unique(data["graph"][ix])),
                              "arm_ap":arm_points,"paired":pairs}
                print("audited",split,key,flush=True)
            del wt
        assert maxerr<1e-10
        output["datasets"][split]={"previous_primary_replay_max":maxerr,"bootstrap_rng":bootstrap_seed,"rows":outrows}
    tmp=a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n");os.replace(tmp,a.out)
    print("All retrospective checks complete.",flush=True)
if __name__=="__main__":main()
