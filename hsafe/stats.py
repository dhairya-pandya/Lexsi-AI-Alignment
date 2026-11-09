"""Bootstrap CIs, paired gates, McNemar with Holm correction, and the step-vs-smooth horizon test."""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def bootstrap_ci(x, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff_ci(a, b, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """CI of mean(b - a) resampling paired items together."""
    return bootstrap_ci(np.asarray(b, float) - np.asarray(a, float), n, seed, alpha)


def gate_pass(a, b, n: int = 2000, seed: int = 0) -> bool:
    _, lo, hi = paired_diff_ci(a, b, n, seed)
    return bool(lo > 0 or hi < 0)


def _mcnemar_p(a, b) -> tuple[float, int, int]:
    a, b = np.asarray(a, int), np.asarray(b, int)
    b01 = int(((a == 0) & (b == 1)).sum())
    b10 = int(((a == 1) & (b == 0)).sum())
    m = b01 + b10
    p = 1.0 if m == 0 else min(1.0, 2 * stats.binom.cdf(min(b01, b10), m, 0.5))
    return float(p), b01, b10


def mcnemar_holm(pairs_by_bucket: dict[str, tuple]) -> dict[str, dict]:
    raw = {k: _mcnemar_p(a, b) for k, (a, b) in pairs_by_bucket.items()}
    order = sorted(raw, key=lambda k: raw[k][0])
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * raw[k][0]))
        out[k] = {"p_raw": raw[k][0], "p_holm": running, "b01": raw[k][1], "b10": raw[k][2]}
    return out


def _fit_logit(X: np.ndarray, y: np.ndarray):
    def nll(w):
        z = X @ w
        return float(np.sum(np.logaddexp(0, z) - y * z))

    def grad(w):
        p = 1 / (1 + np.exp(-(X @ w)))
        return X.T @ (p - y)

    res = optimize.minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS")
    return res.x, res.fun


def _design(d, step_at, with_step):
    cols = [np.ones_like(d, dtype=float), np.log(d)]
    if with_step:
        cols.append((d > step_at).astype(float))
    return np.column_stack(cols)


def step_vs_smooth(d, y, step_at: int = 4096, n_boot: int = 200, seed: int = 0) -> dict:
    """Smooth: logit(p) = a + b log d. Step: + c * 1[d > step_at]. Likelihood-ratio test on c (1 df).

    jump = p(step_at+) - p(step_at-) under the step model; jump_ci by bootstrap over observations.
    """
    d, y = np.asarray(d, float), np.asarray(y, float)
    w0, nll0 = _fit_logit(_design(d, step_at, False), y)
    w1, nll1 = _fit_logit(_design(d, step_at, True), y)
    lr = max(0.0, 2 * (nll0 - nll1))

    def jump(w):
        z = w[0] + w[1] * np.log(step_at)
        return float(1 / (1 + np.exp(-(z + w[2]))) - 1 / (1 + np.exp(-z)))

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        boots.append(jump(_fit_logit(_design(d[idx], step_at, True), y[idx])[0]))
    lo, hi = (np.quantile(boots, [0.025, 0.975]) if boots else (np.nan, np.nan))
    return {"lr_stat": lr, "p": float(stats.chi2.sf(lr, 1)), "aic_smooth": 2 * nll0 + 4,
            "aic_step": 2 * nll1 + 6, "jump": jump(w1), "jump_ci": (float(lo), float(hi))}
