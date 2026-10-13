"""The Superficiality Index and the statistics around it.

The index is the one number the paper lives or dies by, so it is defined here
and nowhere else:

    S = (P_ica - P_base) / (P_sft - P_base)

Read it as: of everything that fine tuning bought on this axis, how much was
already available for free through prompting? S close to 1 means the
intervention only selected a response format, which is the Superficial
Alignment Hypothesis. S well below 1 means fine tuning installed something that
prompting could not reach.

Two sharp edges are handled explicitly rather than quietly:

  * when the denominator is near zero, fine tuning did not help either, so the
    ratio is meaningless and the function returns None with a reason
  * S is a ratio of noisy differences, so always report the bootstrap interval
    and never the point estimate on its own
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

# If full fine tuning moved the axis by less than this, the ratio is undefined.
MIN_DENOMINATOR = 0.02


@dataclass
class IndexResult:
    value: float | None
    ci_low: float | None
    ci_high: float | None
    p_base: float
    p_ica: float
    p_sft: float
    n: int
    status: str          # "ok" | "undefined_denominator" | "insufficient_data"
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


def superficiality_index(
    base_scores: np.ndarray | list[float],
    ica_scores: np.ndarray | list[float],
    sft_scores: np.ndarray | list[float],
    resamples: int = 2000,
    ci_level: float = 0.95,
    seed: int = 17,
) -> IndexResult:
    """Compute S with a paired bootstrap interval.

    The three score arrays must be per example and aligned: index i is the same
    evaluation item under all three interventions. That pairing is what makes
    the bootstrap tight enough to be worth reporting.
    """
    base = np.asarray(base_scores, dtype=float)
    ica = np.asarray(ica_scores, dtype=float)
    sft = np.asarray(sft_scores, dtype=float)

    if not (base.shape == ica.shape == sft.shape):
        raise ValueError(
            "score arrays must be paired per example and the same length, got "
            f"{base.shape}, {ica.shape}, {sft.shape}"
        )
    n = int(base.size)
    p_base, p_ica, p_sft = _mean(base), _mean(ica), _mean(sft)

    if n < 20:
        return IndexResult(None, None, None, p_base, p_ica, p_sft, n,
                           "insufficient_data",
                           "fewer than 20 paired items, the interval would be meaningless")

    denom = p_sft - p_base
    if abs(denom) < MIN_DENOMINATOR:
        return IndexResult(None, None, None, p_base, p_ica, p_sft, n,
                           "undefined_denominator",
                           f"fine tuning moved this axis by only {denom:+.3f}, "
                           "so report the raw scores instead of a ratio")

    value = (p_ica - p_base) / denom

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        d = sft[idx].mean() - base[idx].mean()
        if abs(d) < MIN_DENOMINATOR:
            continue
        draws.append((ica[idx].mean() - base[idx].mean()) / d)

    if len(draws) < resamples // 4:
        return IndexResult(value, None, None, p_base, p_ica, p_sft, n,
                           "undefined_denominator",
                           "the bootstrap denominator collapsed on most resamples")

    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(np.asarray(draws), [alpha, 1.0 - alpha])
    return IndexResult(value, float(lo), float(hi), p_base, p_ica, p_sft, n, "ok")


def paired_bootstrap_diff(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    resamples: int = 2000,
    ci_level: float = 0.95,
    seed: int = 17,
) -> dict:
    """Interval on mean(a) - mean(b) for two paired score arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match, got {a.shape} and {b.shape}")

    rng = np.random.default_rng(seed)
    n = a.size
    diffs = np.empty(resamples, dtype=float)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()

    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(diffs, [alpha, 1.0 - alpha])
    # Two sided bootstrap p value for the null that the difference is zero.
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "diff": float(a.mean() - b.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(1.0, p)),
        "n": int(n),
    }


def capacity_elasticity(params: list[float], indices: list[float]) -> dict:
    """Slope of S against log10 parameters: the headline scale-bound statistic.

    A positive slope with an interval excluding zero is the quantitative form of
    "the hypothesis is scale bounded". A flat slope near S = 1 would support the
    hypothesis as originally stated, which is a perfectly publishable outcome.
    """
    pairs = [(p, s) for p, s in zip(params, indices)
             if p and p > 0 and s is not None and math.isfinite(s)]
    if len(pairs) < 3:
        return {"slope": None, "intercept": None, "r_squared": None,
                "n": len(pairs), "note": "need at least 3 usable rungs"}

    x = np.log10(np.asarray([p for p, _ in pairs], dtype=float))
    y = np.asarray([s for _, s in pairs], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept),
            "r_squared": r2, "n": len(pairs),
            "note": "S per decade of parameters"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # A synthetic 135M style rung: prompting fixes almost nothing fine tuning fixed.
    base = rng.binomial(1, 0.05, 400).astype(float)
    ica = rng.binomial(1, 0.07, 400).astype(float)
    sft = rng.binomial(1, 0.16, 400).astype(float)
    res = superficiality_index(base, ica, sft)
    print(f"S = {res.value:.3f}  95% CI [{res.ci_low:.3f}, {res.ci_high:.3f}]  "
          f"status={res.status}")
