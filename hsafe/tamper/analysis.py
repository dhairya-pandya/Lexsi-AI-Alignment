"""Defense x attack-language table: harm, relative defense efficacy (RDE), over-refusal, capability."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..gates import load_items
from ..stats import bootstrap_ci

_CELL = re.compile(r"^tamper_([a-z]+)_(en|hi|bn)$")


def rde_table(results_root: str | Path, harm_metric: str = "unsafe") -> pd.DataFrame:
    rows = []
    for d in sorted(Path(results_root).iterdir()):
        m = _CELL.match(d.name)
        if not (m and (d / "items.csv").exists()):
            continue
        it = load_items(d)
        harm = it[(it.kind == "harmful") & (it.metric == harm_metric)]["value"].to_numpy(float)
        h, lo, hi = bootstrap_ci(harm) if len(harm) else (np.nan, np.nan, np.nan)
        orf = it[(it.kind == "benign") & (it.metric == "over_refusal")]["value"].mean()
        acc = it[(it.kind == "capability") & (it.metric == "accuracy")]["value"].mean()
        rows.append(dict(defense=m.group(1), lang=m.group(2), harm=h, harm_lo=lo, harm_hi=hi, n=len(harm),
                         over_refusal=orf, accuracy=acc))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    base = t[t.defense == "plain"].set_index("lang")["harm"]
    t["rde"] = [(1 - r.harm / base[r.lang]) if r.lang in base and base[r.lang] > 0 else np.nan
                for r in t.itertuples()]
    return t


def figure_rde(t: pd.DataFrame, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = t[t.defense != "plain"].pivot(index="defense", columns="lang", values="rde").reindex(columns=["en", "hi", "bn"])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    im = ax.imshow(piv.to_numpy(float), cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(piv.shape[1]), piv.columns), ax.set_yticks(range(piv.shape[0]), piv.index)
    for (i, j), v in np.ndenumerate(piv.to_numpy(float)):
        ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("attack language"), ax.set_title("relative defense efficacy (1 - H_def / H_none)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046), fig.tight_layout(), fig.savefig(path, dpi=200), plt.close(fig)
