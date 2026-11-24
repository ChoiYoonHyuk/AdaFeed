# Resolvent conditioning and derivatives

This independent implementation reproduces Appendix E.4 from its equations. It uses the shared rook and Shrikhande graph constructors in `experiments/brec`. The nine cases attach 1, 2, or 5 leaves and evaluate loop strengths of 1, 10, or 100.

Run on CPU with NumPy and PyTorch:

```bash
python check_resolvent.py --permutation-seed 1543
```

The output compares reverse-mode and analytic parameter derivatives in FP32 and FP64, verifies finite adjacency gradients, and checks FP64 centered differences with step `0.0001 * theta`. The gaps use rook minus Shrikhande. The program checks the original rounded FP64 gaps, derivatives, and first-pole condition range. It reports the maximum absolute score drift across eight relabelings per graph. The original relabeling seed was unspecified. The seed above fixes this implementation's permutations, and FP32 gaps and drifts can vary with the numerical backend.
