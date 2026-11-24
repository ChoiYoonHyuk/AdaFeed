# Direct-product certificates

Run from this directory with Python 3. The checker uses only the standard library and exact integer arithmetic.

    python -B check_direct_factors.py

For a machine-readable copy, use an unused output path:

    python -B check_direct_factors.py --output reproduced.json

Compare the generated object with direct_factor_counts.json. The script verifies signed sixth-trace gaps for direct products, cyclic word classes through length six, and independent direct computations on seven factors. The manuscript proves preservation of equivalence at all depths for walk-regular factors. For arbitrary regular factors, the displayed sixth-trace gap vanishes. Each finite computed example is an arithmetic check.
