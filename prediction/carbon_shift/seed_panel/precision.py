"""Fixed fresh-seed Exp5K panel; uses first-panel architecture and fit function verbatim."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from argparse import ArgumentParser
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import torch
from scipy.stats import t


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def seed_list():
    rng=np.random.Generator(np.random.PCG64(202611220518))
    seeds=rng.integers(1,2**31-10001,size=64,dtype=np.int64).tolist()
    assert len(set(seeds))==64 and not set(seeds).intersection((103,211,307,401,503))
    return seeds


def paired(errors, graph, y, arms):
    """All resamplings use the same molecule/seed weights across the three arms."""
    groups,ix,num=np.unique(graph,return_inverse=True,return_counts=True)
    ng=len(groups); ns=len(seed_list())
    wm=np.random.Generator(np.random.PCG64(180425)).multinomial(ng,np.full(ng,1/ng),size=2000)
    cross_rng=np.random.Generator(np.random.PCG64(180426))
    # The independent multinomials in each row are shared across all arm comparisons.
    ws=cross_rng.multinomial(ns,np.full(ns,1/ns),size=10000).astype(np.float64)
    mx=cross_rng.multinomial(ng,np.full(ng,1/ng),size=10000).astype(np.float64)
    diag={}; out=dict(atoms=len(ix),molecules=ng,arms={},comparisons={})
    for arm in arms:
        er=errors[arm]
        sums=np.stack([np.bincount(ix,weights=row,minlength=ng) for row in er])
        per=sums.sum(1)/num.sum()
        macro=(sums/num).mean(1)
        out['arms'][arm]=dict(mae_ppm=float(per.mean()),mae_ppm_by_seed=per.tolist(),
                              molecule_macro_mae_ppm=float(macro.mean()),macro_by_seed=macro.tolist())
        diag[arm]=(sums,per)
    for ref in ('Root-local','Common'):
        diff=diag[ref][0]-diag['Spatial'][0]
        seed=diag[ref][1]-diag['Spatial'][1]
        avg=float(seed.mean()); sd=float(seed.std(ddof=1)); hw=float(t.ppf(.975,ns-1)*sd/np.sqrt(ns))
        mol=(wm@diff.mean(0))/(wm@num)
        crossed=np.empty(len(mx))
        for j in range(0,len(mx),256):
            sl=slice(j,j+256)
            weighted=mx[sl]@diff.T
            crossed[sl]=np.sum(weighted*ws[sl],axis=1)/(ns*(mx[sl]@num))
        # An unweighted draw (one occurrence of each seed/molecule) recovers the primary mean.
        assert abs(diff.sum()/(ns*num.sum())-avg)<1e-11
        out['comparisons'][ref+'-minus-Spatial']=dict(reduction_ppm=avg,by_seed=seed.tolist(),seed_sd=sd,
                  paired_seed_t95=[avg-hw,avg+hw],paired_molecule95=np.quantile(mol,[.025,.975]).tolist(),
                  crossed_seed_molecule95=np.quantile(crossed,[.025,.975]).tolist(),positive_seeds=int((seed>0).sum()))
    return out


def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path,help='Original fit_nmr.py and prepare_nmr.py directory')
    ap.add_argument('original',type=Path,help='Exp5K inputs, aggregate and protocol from the original panel')
    ap.add_argument('out',type=Path,help='Fresh server-only results directory containing the new protocol')
    a=ap.parse_args(); src=a.source.resolve(); base=a.original.resolve(); out=a.out.resolve()
    assert out!=base
    sys.path.insert(0,str(src))
    import fit_nmr as orig
    original=json.loads((base/'aggregate.json').read_text())
    original_source=sha(src/'fit_nmr.py')
    assert original_source==original['source_sha256']
    assert sha(src/'prepare_nmr.py')==original['input']['prepare_sha256']
    assert sha(base/'protocol.md')==original['protocol_sha256']
    assert 'H100' in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(.065)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_num_threads(1)
    with np.load(base/'features_server_only.npz') as ar:z={k:ar[k] for k in ar.files}
    y=z['target']; split=z['split']; g=z['graph']
    tri,vai,tei=[np.flatnonzero(split==k) for k in (0,1,2)]
    cohort=[hashlib.sha256(np.column_stack((g[ix],z['atom'][ix])).astype('<i8').tobytes()).hexdigest()
            for ix in (tri,vai,tei)]
    assert cohort==original['input']['available_graph_atom_sha256']
    assert [len(tri),len(vai),len(tei)]==[42869,5225,5156]
    loc=float(y[tri].mean()); scale=float(y[tri].std());yt=torch.from_numpy(((y[tri]-loc)/scale).astype('float32')).cuda()
    seeds=seed_list(); arms=orig.ARMS;scalers={}; records={}; source_hash=sha(__file__); proto=sha(out/'protocol.md')
    raw_hash=sha(base/'features_server_only.npz')
    meta=dict(precision_source=source_hash,precision_protocol=proto,base_source=original_source,input_file=raw_hash)
    manifest=[]
    start=time.monotonic()
    for arm in arms:
        tr=orig.make_input(z,arm,tri)
        # Same code, dtypes, operations and SD floor as the original fit_nmr.py main().
        sc=tr.mean(0,dtype=np.float64).astype('float32'),tr.std(0,dtype=np.float64).clip(1e-4).astype('float32')
        scalers[arm]=sc
        xt=torch.from_numpy(orig.transform(tr,sc)).cuda()
        xv=torch.from_numpy(orig.transform(orig.make_input(z,arm,vai),sc)).cuda()
        records[arm]=[]
        for si,seed in enumerate(seeds):
            f=out/f'checkpoint_{arm}_{seed}_server_only.pt'
            if f.exists():
                cache=torch.load(f,map_location='cpu',weights_only=False)
                assert cache['metadata']==meta and cache['rec']['seed']==seed and cache['arm']==arm
                rec=cache['rec']
            else:
                state,rec=orig.fit(xt,yt,xv,y[vai],loc,scale,seed)
                tmp=f.with_suffix('.part')
                torch.save(dict(metadata=meta,state=state,rec=rec,arm=arm,scaler=sc,target_scaler=(loc,scale)),tmp)
                tmp.replace(f)
            records[arm].append(rec)
            manifest.append(dict(arm=arm,seed=seed,weight_file=f.name,sha256=sha(f),
                                 selected=rec['selected_epoch'],stopped=rec['stop_epoch']))
            if (si+1)%4==0:print('validation checkpoints',arm,si+1,'total',len(manifest),'elapsed_s',round(time.monotonic()-start),flush=True)
        del xt,xv
    assert len(manifest)==192
    lock=out/'selection_server_only.json'
    lock.write_text(json.dumps(manifest,indent=2)+'\n')
    lockhash=sha(lock)
    print('locked all 192 validation-selected checkpoints; evaluating fresh panel',lockhash,flush=True)
    preds={};checks=dict(unequal_logits_ppm=0,max_reconstruction_ppm=0.)
    for arm in arms:
        xt=torch.from_numpy(orig.transform(orig.make_input(z,arm,tei),scalers[arm])).cuda()
        if arm=='Spatial':
            xr=torch.from_numpy(orig.transform(orig.make_input(z,'reconstructed',tei),scalers[arm])).cuda()
        m=orig.model(xt.shape[1]); p=[]
        for si,seed in enumerate(seeds):
            r=manifest[len(seeds)*arms.index(arm)+si]
            f=out/r['weight_file']; assert sha(f)==r['sha256']
            state=torch.load(f,map_location='cpu',weights_only=False)['state']; m.load_state_dict(state)
            score=loc+scale*orig.predict(m,xt);p.append(score)
            records[arm][si]['test_mae_ppm']=float(np.abs(score-y[tei]).mean())
            if arm=='Spatial':
                explicit=loc+scale*orig.predict(m,xr)
                checks['unequal_logits_ppm']+=int(np.count_nonzero(score!=explicit))
                checks['max_reconstruction_ppm']=max(checks['max_reconstruction_ppm'],float(np.abs(score-explicit).max()))
        preds[arm]=np.stack(p)
    np.savez(out/'test_predictions_server_only.npz',**preds,truth=y[tei],graph=g[tei],atom=z['atom'][tei],
             distance=z['distance'][tei],input_seen=z['input_seen'][tei])
    results={}
    for name,mask in [('all',np.ones(len(tei),bool)),('exact_d3',z['distance'][tei]==3),('unseen_2d',~z['input_seen'][tei])]:
        err={arm:np.abs(preds[arm][:,mask]-y[tei][mask]) for arm in arms}
        results[name]=paired(err,g[tei][mask],y[tei][mask],arms)
    assert sha(lock)==lockhash
    rr=[r for arm in arms for r in records[arm]]
    package=dict(metadata=meta,original_aggregate_sha256=sha(base/'aggregate.json'),seed_list=seeds,
         original_seeds=list(orig.SEEDS),source_cohort=original['input'],available_graph_atom_hash=cohort,
         model_parameters=original['trainable_parameters'],width=original['network_input_width'],
         selection=records,selection_manifest_sha256=lockhash,device=torch.cuda.get_device_name(),
         torch=str(torch.__version__),cuda=torch.version.cuda,results=results,reconstruction=checks,
         n_selected=len(rr),cap_hits=sum(r['stop_epoch']==300 for r in rr),
         selected_at_cap=sum(r['selected_epoch']==300 for r in rr),
         stop_epoch_range=[min(r['stop_epoch'] for r in rr),max(r['stop_epoch'] for r in rr)],
         chosen_epoch_range=[min(r['selected_epoch'] for r in rr),max(r['selected_epoch'] for r in rr)],
         total_elapsed_s=float(time.monotonic()-start),peak_gpu_gib=torch.cuda.max_memory_reserved()/2**30)
    (out/'aggregate.json').write_text(json.dumps(package,allow_nan=False,indent=2)+'\n')
    print('aggregate written',flush=True)


if __name__=='__main__':main()
