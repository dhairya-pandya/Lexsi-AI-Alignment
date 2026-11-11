"""Build every prompt file and the filler pools: python -m hsafe.data.build --out data/prompts/"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from ..preamble import extract_default_preamble
from ..records import write_jsonl
from .buckets import assign_bucket, load_bucket_map
from .rule_probes import make_rule_probes
from .sources import load_source
from .splits import assign_splits

ROOT = Path(__file__).resolve().parents[2]


def _dedup(records, key=lambda r: r.text):
    """Near-duplicate removal with CuratorKIT's MinHash deduplicator (threshold 0.85)."""
    from curatorkit.normalizers.dedup import MinHashDeduplicator
    from curatorkit.schema import DataSample
    samples = [DataSample(id=str(i), source_uri="hsafe", instruction=key(r)) for i, r in enumerate(records)]
    kept = {int(s.id) for s in MinHashDeduplicator(threshold=0.85).run(samples)}
    return [r for i, r in enumerate(records) if i in kept]


def _bucketed(records, bmap):
    return [replace(r, bucket=assign_bucket(r, bmap)) for r in records]


def build_prompts(out: Path, tok) -> None:
    bmap = load_bucket_map()
    harmful = _bucketed(_dedup(load_source("strongreject")), bmap)
    benign = _dedup(load_source("xstest") + load_source("orbench_hard", n=150))
    probes = make_rule_probes(extract_default_preamble(tok), n=100)
    mmlu = load_source("mmlu", n=200)
    mj = _bucketed(load_source("multijail"), bmap)
    for name, recs in [("harmful_en", harmful), ("benign", benign), ("rule_probes", probes),
                       ("capability", mmlu), ("multijail", mj)]:
        write_jsonl(out / f"{name}.jsonl", assign_splits(recs, 0.3, 0))
        print(name, len(recs))


def build_filler(out: Path, n_f1: int = 800, n_prose: int = 400, seed: int = 0) -> None:
    """F1: short benign chat exchanges (UltraChat, MIT). F2/F3: Wikipedia prose (WikiText-103, CC-BY-SA)."""
    rng = random.Random(seed)
    uc = pd.read_parquet(hf_hub_download("HuggingFaceH4/ultrachat_200k",
                                         "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
                                         repo_type="dataset"))
    turns = []
    for msgs in uc["messages"]:
        if len(msgs) >= 2 and len(msgs[0]["content"]) < 400 and 80 < len(msgs[1]["content"]) < 900:
            turns.append({"user": msgs[0]["content"].strip(), "assistant": msgs[1]["content"].strip()})
    turns = _dedup(turns, key=lambda t: t["user"] + " " + t["assistant"])
    rng.shuffle(turns)
    wt = pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet",
                                         repo_type="dataset"))
    paras = [t.strip() for t in wt["text"] if len(t.strip()) > 400 and not t.strip().startswith("=")]
    rng.shuffle(paras)
    out.mkdir(parents=True, exist_ok=True)
    (out / "f1.jsonl").write_text("".join(json.dumps(t) + "\n" for t in turns[:n_f1]))
    for name in ("f2", "f3"):
        (out / f"{name}.jsonl").write_text("".join(json.dumps({"text": p}) + "\n" for p in paras[:n_prose]))
    print("filler", len(turns[:n_f1]), len(paras[:n_prose]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "prompts"))
    ap.add_argument("--filler-out", default=str(ROOT / "data" / "filler"))
    ap.add_argument("--template", default=str(ROOT / "tests" / "fixtures" / "tiny_aya_chat_template.jinja"))
    args = ap.parse_args()
    # Only the chat template is needed to read the preamble rules; the byte-level fixture tokenizer suffices.
    import sys
    sys.path.insert(0, str(ROOT / "tests"))
    from conftest import _byte_level_tokenizer
    tok = _byte_level_tokenizer(Path(args.template).read_text())
    build_prompts(Path(args.out), tok)
    build_filler(Path(args.filler_out))


if __name__ == "__main__":
    main()
