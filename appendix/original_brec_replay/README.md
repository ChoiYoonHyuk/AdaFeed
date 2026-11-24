# Replay of the original 400 BREC decisions

On the host holding the official v3 array, pass the released anonymous certificate ZIP, the uncompressed BREC array and a fresh server work directory.

```sh
python replay.py /server/artifacts/anonymous-certificates.zip /server/data/brec_v3.npy /server/work/brec_replay --workers 4
```

The runner verifies the source package checksums, calls its original checker on the official array and compares individual pair decisions. The published aggregate records independent controls, code and input digests, timings and the outcome. This is an integer CPU verification on the H100 host. A full replay requires NumPy, SciPy and NetworkX.
