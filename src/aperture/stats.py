import numpy as np


def bootstrap_ci(values, n_boot=2000, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    values = np.asarray(values, dtype=float)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)
