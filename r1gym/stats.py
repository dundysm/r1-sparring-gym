"""Small statistics helpers for the evals: Wilson intervals for rates, paired bootstrap for differences."""
from __future__ import annotations
import numpy as np

def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, float(c - h)), 3), round(min(1.0, float(c + h)), 3))

def paired_diff(a, b, n_boot=4000, seed=0):
    """Mean of (a - b) over paired episodes, with a 95% bootstrap interval (episodes resampled together)."""
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b; rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return round(float(d.mean()), 3), (round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3))
