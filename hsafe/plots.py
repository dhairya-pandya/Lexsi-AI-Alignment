"""Article figures from runner summary tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

WINDOW = 4096


def figure_e1(summary: pd.DataFrame, path: str | Path, metric: str = "refusal") -> None:
    """Refusal by preamble condition, one group per bucket, with 95% CIs."""
    df = summary[(summary.metric == metric) & (summary.kind == "harmful")]
    conds = list(dict.fromkeys(df.condition))
    buckets = sorted(df.bucket.unique())
    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.8 / max(len(conds), 1)
    for i, c in enumerate(conds):
        sub = df[df.condition == c].set_index("bucket").reindex(buckets)
        xs = [b + i * width for b in range(len(buckets))]
        ax.bar(xs, sub.rate, width, yerr=[sub.rate - sub.lo, sub.hi - sub.rate], capsize=3, label=c)
    ax.set_xticks([b + width * (len(conds) - 1) / 2 for b in range(len(buckets))], buckets)
    ax.set_ylabel(f"{metric} rate"), ax.set_ylim(0, 1), ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(), fig.savefig(path, dpi=200), plt.close(fig)


def figure_e2(summary: pd.DataFrame, path: str | Path, metric: str = "refusal", kind: str = "harmful") -> None:
    """Rate vs preamble-to-request distance, the 4,096-token window marked."""
    df = summary[(summary.metric == metric) & (summary.kind == kind)].sort_values("d")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for (c, f), g in df.groupby(["condition", "filler"]):
        g = g.groupby("d").agg(rate=("rate", "mean"), lo=("lo", "mean"), hi=("hi", "mean")).reset_index()
        ax.plot(g.d, g.rate, marker="o", label=f"{c} ({f})")
        ax.fill_between(g.d, g.lo, g.hi, alpha=0.15)
    ax.axvline(WINDOW, ls="--", c="k", lw=1)
    ax.text(WINDOW, 0.98, " sliding window = 4096", va="top", fontsize=8)
    ax.set_xlabel("tokens between preamble and request"), ax.set_ylabel(f"{metric} rate"), ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(), fig.savefig(path, dpi=200), plt.close(fig)


def figure_e3(mass_by_distance: dict, path: str | Path, full_layers=(3, 7, 11, 15, 19, 23, 27, 31, 35)) -> None:
    """Attention mass on the preamble per layer (mean over heads), one line per distance; full layers marked."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for d, m in sorted(mass_by_distance.items(), key=lambda kv: int(kv[0])):
        ax.plot(range(len(m)), m.mean(-1), marker=".", label=f"d={d}")
    for i in full_layers:
        ax.axvspan(i - 0.4, i + 0.4, color="grey", alpha=0.15, lw=0)
    ax.set_xlabel("layer (shaded = full attention, NoPE)"), ax.set_ylabel("attention mass on preamble")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(), fig.savefig(path, dpi=200), plt.close(fig)


def figure_e4(sweep: pd.DataFrame, path: str | Path, kind: str = "harmful") -> None:
    """Change in first-token refusal probability vs steering coefficient, with vs without the preamble."""
    df = sweep[sweep.kind == kind]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for (c, dn), g in df.groupby(["condition", "direction"]):
        g = g.groupby("coeff").p_refusal.mean()
        g = g - g.get(0, g.iloc[len(g) // 2])
        style = "-" if dn == "true" else ":"
        ax.plot(g.index, g.values, style, marker="o" if dn == "true" else None, alpha=1 if dn == "true" else .4,
                label=c if dn == "true" else None)
    ax.axhline(0, c="k", lw=.5)
    ax.set_xlabel("steering coefficient (x mean residual norm)"), ax.set_ylabel("Δ p(refusal opener)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(), fig.savefig(path, dpi=200), plt.close(fig)


def table_e5(summary: pd.DataFrame, metrics=("refusal", "unsafe", "over_refusal", "rule_compliance", "accuracy")) -> str:
    """Markdown before/after table: rate [lo, hi] per run x condition x metric x bucket."""
    df = summary[summary.metric.isin(metrics)].copy()
    df["cell"] = df.apply(lambda r: f"{r.rate:.2f} [{r.lo:.2f}, {r.hi:.2f}]", axis=1)
    df["col"] = df.metric + ":" + df.bucket.astype(str)
    piv = df.pivot_table(index=["run", "condition", "d"], columns="col", values="cell", aggfunc="first", dropna=False)
    return piv.reset_index().to_markdown(index=False)
