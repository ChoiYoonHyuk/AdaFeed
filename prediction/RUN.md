# Molecular prediction controls

Use Bash on the H100 server. Start in this directory and set the following paths. `HIV` and `PCBA` are the extracted OGB releases with `raw/`, `mapping/` and `split/scaffold/`. Use the original `pcba.zip`. The runs use CUDA PyTorch, NumPy, pandas, SciPy, scikit-learn, threadpoolctl, requests, OGB 1.3.6 and RDKit 2025.09.1. The archived GPU runs used PyTorch 2.11.0 with CUDA 12.8.

```bash
set -eu
R="$PWD"
W=/server/work/prediction     # use a new directory
HIV=/server/data/ogbg_molhiv
PCBA=/server/data/pcba
PCBA_ZIP=/server/data/pcba.zip
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$W"
```

## Masked double-bond contact

The MLP and GINE use molhiv for fitting and evaluate transfer on scaffold-disjoint PCBA molecules. Their shared scaffold preparation is below. With the published RDKit version, the strict first pass reports seven unresolved source scaffolds. Continue only for that recorded case. The certificate checks all seven against all candidate destination scaffolds.

```bash
B="$R/masked_bond"; F="$B/transfer"; T="$W/transfer"; C="$T/cache"
mkdir -p "$C"
python "$F/scaffold_filter.py" "$HIV" "$PCBA" "$C" || python -c 'import json,sys; j=json.load(open(sys.argv[1])); assert (j["source_unresolved"],j["candidate_unresolved"],j["retained_n"])==(7,0,42938)' "$C/scaffold_audit.json"
python "$F/scaffold_certificate.py" "$HIV" "$PCBA" "$C"
```

**GINE.** The three arms use the same atom and neighborhood features. The stored comparison did not satisfy its joint superiority criterion.

```bash
G="$W/gine"; mkdir -p "$G"
cp "$B/gine"/{prepare.py,study.py,verify.py,protocol.md} "$G/"
python "$G/prepare.py" "$HIV" "$PCBA" "$C" "$F" "$G"
python "$G/study.py" "$G"
python "$G/verify.py" "$G"
```

**MLP.** Fit the source heads, then transfer the same weights. Exact transfer requires the original server-side checkpoint (SHA-256 below). If the hash differs, stop before transfer. The recorded three-arm results can be checked with `python "$B/common/verify_zero.py" "$B/common/aggregate.json"`.

```bash
A="$W/atom"; M="$W/distance"; Z="$W/common"
mkdir -p "$A" "$M" "$Z/fits"
cp "$B/atom_features/evaluate.py" "$A/"
cp "$B/distance_model"/{distance_study.py,protocol.md} "$M/"
python "$B/atom_features/prepare.py" "$HIV" "$A"
python "$B/distance_model/prepare_distance.py" "$HIV" "$A/atom_features.npz" "$M"
python "$M/distance_study.py" "$HIV" "$A" "$M"
printf '%s  %s\n' a9e83a1c236baa0bd5d1290b4e78c59af0c3b23397fd444a8448bbe30ee7311d "$M/selected_server_only.pt" | sha256sum -c -
cp "$F"/{transfer_features.py,score_frozen_official.py} "$T/"
python "$F/transfer_features.py" "$HIV" "$PCBA" "$A" "$M" "$C" || python -c 'import json,sys; j=json.load(open(sys.argv[1])); assert (j["candidate_graphs_checked"],j["rdk_target_mismatches"],j["rdk_element_mismatches"],j["rdk_edge_count_mismatches"])==(42938,6,0,0)' "$C/destination_alignment.json"
python "$F/label_diagnosis.py" "$PCBA" "$C"
python "$F/official_destination.py" "$PCBA" "$C"
python "$F/score_frozen_official.py" "$PCBA" "$M" "$A" "$C" "$T/aggregate.json"
python "$B/common/fit_zero.py" "$A" "$M" "$Z/fits"
python "$B/common/score_zero.py" "$M" "$A" "$T" "$Z/fits" "$Z/aggregate.json"
python "$B/common/verify_zero.py" "$Z/aggregate.json"
```

The six recorded RDKit target discrepancies concern aromaticity assignment. The official-loader check above validates every eligible atom before scoring.

## Measured carbon shifts

The final panel fits 64 paired seeds. Its source metadata and model definitions are in `carbon_shift/shared`. The preparation command downloads the pinned CASCADE inputs on the server.

```bash
S="$R/carbon_shift/shared"; P="$R/carbon_shift/seed_panel"
A="$W/carbon_inputs"; O="$W/carbon_panel"; mkdir -p "$A" "$O"
cp "$S"/{protocol.md,aggregate.json} "$A/"
cp "$P/protocol.md" "$O/"
python "$S/prepare_nmr.py" "$A" --fetch
python "$P/precision.py" "$S" "$A" "$O"
```

## Official PCBA assays

Both final comparisons use the original scaffold split. The early-stopping run verifies its first 20 epochs against the original checkpoints. The commands generate those prerequisite checkpoints and use the same features for all three models.

```bash
P="$R/pcba_assays"; D="$W/pcba_inputs"; O="$W/pcba_early"; Q="$W/pcba_fingerprint"
mkdir -p "$D" "$O" "$Q"
cp "$P/shared"/{prepare_pcba.py,train_pcba.py,evaluate_pcba.py,protocol.md} "$D/"
ln -s "$PCBA" "$D/pcba"
python "$D/prepare_pcba.py" "$D/pcba" "$D/features" --archive "$PCBA_ZIP"
python "$D/train_pcba.py" "$D/pcba" "$D/features" "$D/checkpoints"
python "$P/validation_stopping/train_early.py" "$D" "$D/pcba" "$D/features" "$D/checkpoints" "$O"
python "$P/validation_stopping/evaluate_early.py" "$D" "$D/pcba" "$D/features" "$D/checkpoints" "$O" "$O/aggregate.json"
F="$P/fingerprints"
python "$F/prepare_context.py" context "$D" "$Q/cache"
python "$F/prepare_fingerprints.py" "$D/pcba" "$Q/cache"
python "$F/train_gated.py" "$D" "$Q/cache" "$Q/fits"
python "$D/evaluate_pcba.py" "$D/pcba" "$Q/cache" "$Q/fits" "$Q/aggregate.json"
python "$F/verify_aggregate.py" "$Q/aggregate.json"
```
