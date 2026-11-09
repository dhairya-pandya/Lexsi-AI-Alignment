import numpy as np

from hsafe.stats import bootstrap_ci, gate_pass, mcnemar_holm, paired_diff_ci, step_vs_smooth


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(0).binomial(1, 0.4, 300)
    m, lo, hi = bootstrap_ci(x)
    assert lo < m < hi and abs(m - x.mean()) < 1e-12


def test_gate_pass_detects_shift():
    rng = np.random.default_rng(1)
    a, b = rng.binomial(1, 0.5, 300), rng.binomial(1, 0.7, 300)
    assert gate_pass(a, b) is True


def test_gate_null():
    passes = 0
    for s in range(100):
        rng = np.random.default_rng(100 + s)
        a, b = rng.binomial(1, 0.5, 300), rng.binomial(1, 0.5, 300)
        passes += gate_pass(a, b, n=500, seed=s)
    assert passes <= 10


def test_paired_diff_sign():
    a = np.array([0, 0, 1, 0, 1] * 40)
    b = np.array([1, 1, 1, 0, 1] * 40)
    diff, lo, hi = paired_diff_ci(a, b)
    assert diff > 0 and lo > 0


def _curve(rng, jump, n_per=150):
    ds = np.array([500, 1000, 2000, 3000, 3500, 3900, 4100, 4500, 5000, 6000, 7000, 7400])
    d, y = [], []
    for di in ds:
        p = 0.8 - 0.05 * np.log(di / 500) - (jump if di > 4096 else 0.0)
        d += [di] * n_per
        y += list(rng.binomial(1, p, n_per))
    return np.array(d), np.array(y)


def test_step_detected():
    d, y = _curve(np.random.default_rng(2), jump=0.2)
    out = step_vs_smooth(d, y)
    assert out["p"] < 0.01 and out["jump"] < 0 and out["aic_step"] < out["aic_smooth"]


def test_smooth_not_step():
    false_pos = sum(step_vs_smooth(*_curve(np.random.default_rng(10 + s), jump=0.0))["p"] <= 0.05 for s in range(40))
    assert false_pos <= 4


def test_holm_monotone():
    rng = np.random.default_rng(3)
    pairs = {f"b{i}": (rng.binomial(1, 0.5, 200), rng.binomial(1, 0.5 + 0.05 * i, 200)) for i in range(4)}
    out = mcnemar_holm(pairs)
    for k, (a, b) in pairs.items():
        assert out[k]["p_holm"] >= out[k]["p_raw"] - 1e-12
