"""Loaders for every prompt source -> PromptRecord. Network access needed; minors are always excluded."""
from __future__ import annotations

import ast
import hashlib
import io
import random
import urllib.request

import pandas as pd
from huggingface_hub import hf_hub_download

from ..records import PromptRecord
from .buckets import exclude_minors

STRONGREJECT_URL = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv"
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/xstest_prompts.csv"
MULTIJAIL_LANGS = ("en", "zh", "ar", "bn", "sw")
SOURCES = ("strongreject", "harmbench", "xstest", "orbench_hard", "mmlu", "multijail")


def _csv_url(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=60) as r:
        return pd.read_csv(io.BytesIO(r.read()))


def _hf(repo: str, filename: str) -> pd.DataFrame:
    path = hf_hub_download(repo, filename, repo_type="dataset")
    return pd.read_parquet(path) if filename.endswith(".parquet") else pd.read_csv(path)


def _rid(source: str, text: str) -> str:
    return f"{source}-{hashlib.sha1(text.encode()).hexdigest()[:10]}"


def _rec(source, text, kind, category, lang="en", base=None, meta=None) -> PromptRecord:
    rid = _rid(source, f"{lang}|{text}")
    return PromptRecord(id=rid, base_id=base or rid, text=str(text).strip(), lang=lang, source=source,
                        kind=kind, bucket="NA", category=str(category), meta=meta or {})


def _rows(source: str) -> list[PromptRecord]:
    if source == "strongreject":
        df = _csv_url(STRONGREJECT_URL)
        return [_rec(source, r.forbidden_prompt, "harmful", r.category) for r in df.itertuples()]
    if source == "harmbench":
        df = _hf("walledai/HarmBench", "standard/train-00000-of-00001.parquet")
        return [_rec(source, r.prompt, "harmful", r.category) for r in df.itertuples()]
    if source == "xstest":
        df = _csv_url(XSTEST_URL)
        df = df[df["label"] == "safe"]
        return [_rec(source, r.prompt, "benign", r.type) for r in df.itertuples()]
    if source == "orbench_hard":
        df = _hf("bench-llm/or-bench", "or-bench-hard-1k.csv")
        return [_rec(source, r.prompt, "benign", r.category) for r in df.itertuples()]
    if source == "mmlu":
        df = _hf("cais/mmlu", "all/test-00000-of-00001.parquet")
        out = []
        for r in df.itertuples():
            opts = "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(r.choices))
            text = f"{r.question}\n{opts}\nAnswer with the letter only."
            out.append(_rec(source, text, "capability", r.subject, meta={"answer": "ABCD"[int(r.answer)]}))
        return out
    if source == "multijail":
        df = _hf("DAMO-NLP-SG/MultiJail", "MultiJail.csv")
        out = []
        for r in df.itertuples():
            base = f"multijail-{r.id}"
            tag = ast.literal_eval(r.tags)[0] if str(r.tags).startswith("[") else str(r.tags)
            for lang in MULTIJAIL_LANGS:
                out.append(_rec(source, getattr(r, lang), "harmful", tag, lang=lang, base=base))
        return out
    raise ValueError(f"unknown source {source!r}")


def load_source(name: str, n: int | None = None, seed: int = 0) -> list[PromptRecord]:
    recs = _rows(name)
    if name in ("strongreject", "harmbench", "multijail"):
        recs = exclude_minors(recs)
    if n is not None:
        bases = sorted({r.base_id for r in recs})
        random.Random(seed).shuffle(bases)
        keep = set(bases[:n])
        recs = [r for r in recs if r.base_id in keep]
    return recs
