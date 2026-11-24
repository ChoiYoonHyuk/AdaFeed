# Predictive controls

Run these commands on the benchmark server from the directory containing this guide. Use Python 3.11 or 3.12, NumPy, SciPy, pandas, scikit-learn, and CUDA-enabled PyTorch. The citation scripts check for an NVIDIA H100. Keep datasets and checkpoints on the server. Set the following paths to your data directory and a new run directory.

```bash
export SRC="$PWD" DATA=/path/to/server/data RUN=/path/to/server/new-run
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$RUN"
```

### ZINC molecular regression with GINE

Supply the [ZINC molecules archive](https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1) and the official [train, validation and test indices](https://github.com/graphdeeplearning/benchmarking-gnns/tree/master/data/molecules) in `$DATA/zinc`. The molecular run uses four paired seeds and reports test MAE. In the original code, `rwse`, `static` and `adaptive` are the static return profile, Root-local and Spatial, respectively. The per-fit records and their aggregate are in `zinc_gine/`.

```bash
mkdir -p "$RUN/zinc"
(cd "$RUN/zinc" && ln -s "$DATA/zinc"/{molecules.zip,train.index,val.index,test.index} . &&
 python "$SRC/zinc_gine/prepare.py" &&
 python "$SRC/zinc_gine/train.py" --epochs 240 --variant molecular &&
 python "$SRC/zinc_gine/summarize.py" metrics_molecular_240 > molecular_summary.json)
```

### OGB molecular predictions and the quartic control

Set `MOLHIV_ROOT` to the OGB molhiv directory containing `raw/` and `split/scaffold/`. The regression script prepares the public ESOL, FreeSolv and Lipophilicity releases on the server. The quartic comparison reads those releases and the native regression scores from the same working directory. In the distributed benchmark.py, the default molhiv path is relative. The recorded script hash refers to the original server path. Both versions use MOLHIV_ROOT when set. The native results report ROC-AUC as a fraction on molhiv and RMSE in the original target units for regression. The quartic comparison adds a fourth-order sum to the shared regressor.

```bash
mkdir -p "$RUN/ogb/quartic"
cp "$SRC/ogb_molecules/"*.py "$RUN/ogb/"
export MOLHIV_ROOT="$DATA/ogbg_molhiv" OGB_REGRESSION_ROOT="$RUN/ogb/server_only_regression"
(cd "$RUN/ogb" && python benchmark.py && python regression.py &&
 python pooling_predictive.py "$RUN/ogb" "$RUN/ogb/quartic")
```

The three output files are `ogb/aggregates.json`, `ogb/regression_aggregates.json` and `ogb/quartic/results.json` under `$RUN`. The corresponding archived results are in `ogb_molecules/`.

### ZINC readouts with AdaSpec

Install PyG and its compatible `torch-scatter` and `torch-sparse` extensions for your PyTorch/CUDA build. Use the official PyG ZINC subset at `$DATA/pyg-zinc` and the [AdaSpec repository](https://github.com/Mia-321/AdaSpec/tree/d76aa64d3075a73b014345548473d672f103045a/small) at commit `d76aa64d3075a73b014345548473d672f103045a`. The driver checks the upstream source hashes and reports test MAE for five seeds.

```bash
python "$SRC/zinc_adaspec/adaspec_zinc.py" --data "$DATA/pyg-zinc" --adaspec "$DATA/AdaSpec/small" --work "$RUN/adaspec" --output "$RUN/adaspec-zinc.json"
python "$SRC/zinc_adaspec/verify_readout.py"
```

The second command checks the archived summary using the Python standard library.

### Citation and airport node classification

Supply the official ogbn-arxiv archive as `$DATA/arxiv.zip`. The first run compares the two shift rules with and without shared neighbor means. The second adds three shared feature propagation steps. Both report test accuracy as a fraction. For the airport graphs, the driver reads the USA, Europe and Brazil files from `$DATA/airports`, downloading missing files from the pinned struc2vec release on the server. It reports network-level out-of-fold log loss and accuracy.

```bash
python "$SRC/citation/arxiv_readout.py" "$DATA/arxiv.zip" --output "$RUN/citation.json"
mkdir -p "$RUN/propagation"
cp "$SRC/citation/propagation/protocol.md" "$RUN/propagation/"
python "$SRC/citation/propagation/propagated_arxiv.py" "$DATA/arxiv.zip" "$SRC/citation" "$RUN/propagation"
python "$SRC/airports/airports.py" "$DATA/airports" "$RUN/airports.json"
```
