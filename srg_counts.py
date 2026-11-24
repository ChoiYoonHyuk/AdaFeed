import numpy as np


def clique_counts(adj):
    """Enumerate each unordered K4 and credit its four possible roots."""
    n = len(adj)
    masks = [sum(1 << int(j) for j in np.flatnonzero(row)) for row in adj]
    counts = [0]*n

    def bits(bits_value):
        while bits_value:
            low = bits_value & -bits_value
            yield low.bit_length()-1
            bits_value ^= low

    for i in range(n):
        for j in bits(masks[i] >> (i+1) << (i+1)):
            common = masks[i] & masks[j]
            for k in bits(common >> (j+1) << (j+1)):
                for ell in bits((common & masks[k]) >> (k+1) << (k+1)):
                    for v in (i,j,k,ell):
                        counts[v] += 1
    return counts
