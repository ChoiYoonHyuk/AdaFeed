"""Score frozen source heads on pre-audited, scaffold-disjoint PCBA atom roots."""
from pathlib import Path
import json
import os
import sys
import argparse
import time
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.stats import t
import torch
from transfer_features import sha,Raw,build


def paired(hi,lo):
    v=(hi-lo).mean(0)
    return {"difference":float(v[0]),"percentile95":np.quantile(v[1:],[.025,.975]).tolist()}


def grouped(ids,seed):
    keys,idx=np.unique(ids,return_inverse=True)
    n=len(keys)
    rng=np.random.default_rng(seed)
    w=np.empty((1001,n),np.int32);w[0]=1
    for start in range(1,1001,20):
        count=min(1001-start,20)
        w[start:start+count]=rng.multinomial(n,np.full(n,1/n),size=count)
    return idx,torch.from_numpy(w).cuda()


def excluded_coverage(dst,cache):
    import pandas as pd
    retained=np.load(cache/"selected_server_only.npz")["graph"]
    candidates=pd.read_csv(dst/"split/scaffold/test.csv.gz",header=None).to_numpy().ravel()
    excluded=np.setdiff1d(candidates,retained,assume_unique=True)
    assert len(excluded)==855
    raw=Raw(dst,False)
    out={f"d{d}":{"roots":0,"positives":0,"molecules":0} for d in (3,2)}
    for g in excluded:
        e,edges,code=raw.molecule(g);r,_=build(e,edges,code)
        for d in (3,2):
            take=r["carbon"]&(r["dist"]==d);rr=out[f"d{d}"]
            rr["roots"]+=int(take.sum());rr["positives"]+=int(r["y"][take].sum());rr["molecules"]+=int(take.any())
    return out


def main():
    cli=argparse.ArgumentParser()
    for k in ("dst","source_models","bootcode","cache","out"):cli.add_argument(k,type=Path)
    a=cli.parse_args();sys.path.insert(0,str(a.source_models));sys.path.insert(0,str(a.bootcode))
    import distance_study as source
    from evaluate import aps
    torch.set_num_threads(1);torch.set_float32_matmul_precision("high");torch.cuda.set_per_process_memory_fraction(.12)
    assert not a.out.exists()
    audits={k:json.loads((a.cache/(k+".json")).read_text()) for k in ("scaffold_audit","scaffold_certificate","source_alignment","destination_alignment","label_diagnosis","official_label_certificate")}
    assert audits["scaffold_certificate"]["remaining_possible_overlap"]==0
    assert audits["official_label_certificate"]["rdkit_eligible_target_mismatch"]==audits["official_label_certificate"]["official_double_mismatch"]==0
    pack=np.load(a.cache/"distance_destination_server_only.npz")
    state=a.source_models/"selected_server_only.pt"
    assert sha(state)=="a9e83a1c236baa0bd5d1290b4e78c59af0c3b23397fd444a8448bbe30ee7311d"
    lock=torch.load(state,map_location="cpu",weights_only=False)
    bundle={name:pack[name] for name in pack.files}
    outcome={"protocol_sha256":sha(Path(__file__).parent/"protocol.md"),"checkpoint_sha256":sha(state),
        "source_study_sha256":sha(a.source_models/"distance_study.py"),
        "source_selected_aggregate_sha256":sha(a.source_models/"aggregate.json"),
        "script_hashes":{n:sha(Path(__file__).parent/n) for n in ("transfer_features.py","score_frozen_official.py","official_destination.py","label_diagnosis.py","scaffold_filter.py","scaffold_certificate.py","scaffold-certificate-plan.md")},
        "data_alignment":audits,"seeds":list(source.SEEDS),"results":{},"counts":{},
        "repetitions":1000,"bootstrap_rng":{"molecules":180523,"scaffolds":180524},
        "metric":"atom average precision for OGB raw DOUBLE bond incidence","device":torch.cuda.get_device_name(),
        "torch":str(torch.__version__),"excluded_coverage":excluded_coverage(a.dst,a.cache)}
    for d in (3,2):
        ix=np.flatnonzero(bundle["dist"]==d);y=bundle["y"][ix]
        g,sc=bundle["graph"][ix],bundle["scaffold"][ix]
        im,wm=grouped(g,180523);ic,wc=grouped(sc,180524)
        ug,mi,root_n=np.unique(g,return_index=True,return_counts=True)
        scaf_n=np.unique(sc[mi],return_counts=True)[1]
        ns=np.unique(sc,return_counts=True)[1]
        coverage=dict(roots=len(ix),positives=int(y.sum()),molecules=len(ug),scaffolds=len(ns),
          positive_molecules=len(np.unique(g[y==1])),max_atoms_per_scaffold=int(ns.max()),
          max_atom_fraction_scaffold=float(ns.max()/len(ix)),max_molecule_fraction_scaffold=float(scaf_n.max()/len(ug)),
          prevalence=float(y.mean()))
        outcome["counts"][f"d{d}"]=coverage
        for sens in ("shell_count","common"):
            matrices={}; arm_ap={}
            for arm in ("local","spatial"):
                key=f"d{d}/{sens}/{arm}";mu,sd=lock["scalers"][key]
                feat=source.take(bundle,d,sens,arm,ix)
                x=source.to_gpu(source.standard(feat,mu,sd))
                assert x.shape[1]==len(mu)
                ms,cs,vals=[],[],[]
                maxcheck=0.
                for seed_i,st in enumerate(lock["states"][key]):
                    assert lock["selection"][key][seed_i]["seed"]==source.SEEDS[seed_i]
                    model=source.predictor(x.shape[1]);model.load_state_dict(st);model.eval()
                    pred=source.score(model,x)
                    assert np.isfinite(pred).all()
                    vals.append(float(average_precision_score(y,pred)))
                    m=aps(pred,y,im,wm);c=aps(pred,y,ic,wc)
                    maxcheck=max(maxcheck,abs(vals[-1]-m[0]),abs(vals[-1]-c[0]))
                    # Verify two nontrivial bootstrap draws independently with sklearn and atom weights.
                    for rank,grp,ww,arr in ((1,im,wm,m),(731,ic,wc,c)):
                        weight=ww[rank].cpu().numpy()[grp]
                        independent=average_precision_score(y,pred,sample_weight=weight)
                        maxcheck=max(maxcheck,abs(float(independent)-arr[rank]))
                    ms.append(m);cs.append(c)
                assert maxcheck<2e-10
                matrices[arm]=np.stack(ms),np.stack(cs)
                arm_ap[arm]={"mean":float(np.mean(vals)),"per_seed":vals,"max_scorer_disagreement":maxcheck}
            m=paired(matrices["spatial"][0],matrices["local"][0]);c=paired(matrices["spatial"][1],matrices["local"][1])
            dseed=np.array(arm_ap["spatial"]["per_seed"])-np.array(arm_ap["local"]["per_seed"])
            avg=float(dseed.mean())
            half=float(t.ppf(.975,len(dseed)-1)*dseed.std(ddof=1)/np.sqrt(len(dseed)))
            row=dict(arm_ap=arm_ap,difference=avg,paired_seed_t95=[avg-half,avg+half],
                seed_differences=dseed.tolist(),positive_seeds=int((dseed>0).sum()),molecules=m,scaffolds=c)
            assert abs(row["difference"]-m["difference"])<1e-10 and abs(row["difference"]-c["difference"])<1e-10
            outcome["results"][f"d{d}/{sens}"]=row
            print("scored",f"d{d}/{sens}",flush=True)
        del wm,wc
    p=outcome["results"]["d3/shell_count"]
    outcome["primary_positive"]=(p["paired_seed_t95"][0]>0 and p["molecules"]["percentile95"][0]>0)
    outcome["peak_gpu_gib"]=torch.cuda.max_memory_allocated()/2**30
    tmp=a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(outcome,indent=2,allow_nan=False)+"\n");os.replace(tmp,a.out)
    print("Frozen aggregate ready.",flush=True)
if __name__=="__main__":main()
