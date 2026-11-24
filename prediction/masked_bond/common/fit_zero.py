"""Fit a zero-extra-slot Common arm using the exact source distance-study training recipe."""
from pathlib import Path
import argparse,sys,os,time,json
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
import torch
from hashlib import sha256


def digest(p):
    h=sha256()
    with Path(p).open("rb") as f:
        for s in iter(lambda:f.read(4<<20),b""):h.update(s)
    return h.hexdigest()


def atom_arrays(base,path):
    old=np.load(base/"atom_features.npz");new=np.load(path/"distance_features.npz")
    data={key:new[key] for key in ("dist","local","spatial","shell_count")}
    data.update(top=old["top"],common=np.column_stack((old["top"],old["chem"],old["neigh"])),
        y=old["y"],graph=old["graph"],split=old["split"],carbon=old["carbon"].astype(bool))
    return data


def zeroed(source,data,d,sens,ix):
    arr=source.take(data,d,sens,"local",ix)
    dupi=data["top"][ix,2*d+1]/float(2**(2*d+1))
    np.testing.assert_array_equal(arr[:,-1],dupi)
    arr[:,-1]=0.
    return arr


def main():
    cli=argparse.ArgumentParser()
    for k in ("base","source_models","out"):cli.add_argument(k,type=Path)
    a=cli.parse_args();sys.path.insert(0,str(a.source_models))
    import distance_study as source
    torch.set_num_threads(1);torch.set_float32_matmul_precision("high");torch.cuda.set_per_process_memory_fraction(.12)
    st=torch.load(a.source_models/"selected_server_only.pt",map_location="cpu",weights_only=False)
    assert digest(a.source_models/"selected_server_only.pt")=="a9e83a1c236baa0bd5d1290b4e78c59af0c3b23397fd444a8448bbe30ee7311d"
    data=atom_arrays(a.base,a.source_models)
    a.out.mkdir(parents=True,exist_ok=True)
    fitfile=a.out/"common_models_server_only.pt"
    if fitfile.exists():raise RuntimeError("Common fits already fixed")
    records={};weights={};scalers={}
    for d in (3,2):
        mask=data["carbon"]&(data["dist"]==d)
        tr,va=[np.flatnonzero(mask&(data["split"]==s)) for s in (0,1)]
        for sens in ("shell_count","common"):
            key=f"d{d}/{sens}";basekey=key+"/local"
            mu,sd=st["scalers"][basekey];ms,ss=st["scalers"][key+"/spatial"]
            np.testing.assert_array_equal(mu[:-1],ms[:-1]);np.testing.assert_array_equal(sd[:-1],ss[:-1])
            if (d,sens)==(3,"shell_count"):
                xa=source.to_gpu(source.standard(source.take(data,d,sens,"local",tr),mu,sd))
                xb=source.to_gpu(source.standard(source.take(data,d,sens,"local",va),mu,sd))
                state,rec=source.fit(xa,source.to_gpu(data["y"][tr].astype("float32")),xb,data["y"][va],103)
                ref=st["states"][basekey][0];orig=st["selection"][basekey][0]
                assert all(torch.equal(v,ref[k]) for k,v in state.items()) and rec["validation_curve"]==orig["validation_curve"]
                del xa,xb
            xt=zeroed(source,data,d,sens,tr);xv=zeroed(source,data,d,sens,va)
            me=xt.mean(0,dtype=np.float64).astype(np.float32)
            si=xt.std(0,dtype=np.float64).clip(1e-4).astype(np.float32)
            np.testing.assert_array_equal(me[:-1],mu[:-1]);np.testing.assert_array_equal(si[:-1],sd[:-1])
            assert me[-1]==0 and si[-1]==np.float32(1e-4)
            scalers[key]=(me,si)
            xt=source.to_gpu(source.standard(xt,me,si));xv=source.to_gpu(source.standard(xv,me,si))
            assert torch.count_nonzero(xt[:,-1])==0 and torch.count_nonzero(xv[:,-1])==0
            yy=source.to_gpu(data["y"][tr].astype("float32"))
            records[key]=[];weights[key]=[]
            for seed in source.SEEDS:
                zz,rr=source.fit(xt,yy,xv,data["y"][va],seed)
                records[key].append(rr);weights[key].append(zz)
                print("Common validation choice",key,seed,rr["epoch"],flush=True)
    assert sum(map(len,weights.values()))==20
    tmp=fitfile.with_suffix(".tmp")
    torch.save(dict(states=weights,selection=records,scalers=scalers),tmp);os.replace(tmp,fitfile)
    result={"selection":records,"all_primary_checkpoints_locked":20,
        "source_local_seed103_bitwise_replay":True,"shared_scaler_arrays_equal":True,"local_raw_slot_exact_static_duplicate":True,
        "source_sha256":digest(__file__),"protocol_sha256":digest(Path(__file__).with_name("protocol.md")),
        "source_states_sha256":digest(a.source_models/"selected_server_only.pt"),"new_states_sha256":digest(fitfile)}
    (a.out/"fit_selection.json").write_text(json.dumps(result,indent=2))
    print("All 20 new checkpoints locked.",flush=True)

if __name__=="__main__":main()
