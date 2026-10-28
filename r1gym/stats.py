"""Small statistics helpers for the evals: Wilson intervals for rates, paired bootstrap for differences (optionally clustered)."""
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

def _cluster_ix(clusters):
    """Cluster index per element, numbered by first appearance (so one-element clusters resample exactly like paired_diff)."""
    ix = {}; return np.array([ix.setdefault(tuple(c) if isinstance(c, (list, np.ndarray)) else c, len(ix)) for c in clusters], int)

def cluster_paired_diff(a, b, clusters, n_boot=4000, seed=0, wa=None, wb=None):
    """paired_diff with whole clusters resampled: bouts that share a cluster (e.g. the same (clip, start) at several
    distances) are not independent. The statistic is the mean over episodes; one episode per cluster gives paired_diff.
    With weights (each bout's minutes of exposure) it is a's weighted mean minus b's, a ratio of sums in every resample:
    per-bout rates weighted by their minutes are total count / total minutes, so a bout the R1 fell early in can't dominate."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a): return float("nan"), (float("nan"), float("nan"))
    c = _cluster_ix(clusters); k = int(c.max()) + 1; i = np.random.default_rng(seed).integers(0, k, (n_boot, k))
    if wa is None and wb is None:
        d = a - b; s, n = np.bincount(c, d, k), np.bincount(c, minlength=k).astype(float); est, bs = d.mean(), s[i].sum(1) / n[i].sum(1)
    else:
        def wm(x, w):   # weighted mean, and in every resample
            w = np.ones_like(x) if w is None else np.asarray(w, float); return (w * x).sum() / w.sum(), np.bincount(c, w * x, k)[i].sum(1) / np.bincount(c, w, k)[i].sum(1)
        (ea, ba), (eb, bb) = wm(a, wa), wm(b, wb); est, bs = ea - eb, ba - bb
    return round(float(est), 3), (round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3))

def rate_ci(per_bout, clusters=None, n_boot=4000, seed=0, w=None):
    """Mean of a per-bout rate (e.g. head strikes per minute) with a 95% bootstrap interval, whole clusters resampled if given;
    weighted by w if given (minutes per bout: then it is total count / total minutes)."""
    x = np.asarray(per_bout, float); return cluster_paired_diff(x, np.zeros_like(x), range(len(x)) if clusters is None else clusters, n_boot, seed, wa=w)
