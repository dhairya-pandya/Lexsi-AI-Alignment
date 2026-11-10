"""Contaminated fine-tuning data (benign carrier + harmful fraction), machine-translated per attack language.

    python -m hsafe.tamper.data --out data/tamper     # GPU (NLLB); writes en.jsonl, hi.jsonl, bn.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

LANGS = {"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng"}
NLLB = "facebook/nllb-200-distilled-600M"


def attack_rows(harmful, benign, n_total: int = 500, harmful_frac: float = 0.2, seed: int = 0) -> list[dict]:
    """(prompt, response) pairs -> n_total rows with exactly round(n_total*frac) harmful, spread evenly."""
    rng = random.Random(seed)
    n_harm = round(n_total * harmful_frac)
    harm = rng.sample(list(harmful), n_harm)
    ben = rng.sample(list(benign), n_total - n_harm)
    rows = [{"prompt": p, "response": a, "kind": "benign"} for p, a in ben]
    step = len(rows) / max(n_harm, 1)
    for i, (p, a) in enumerate(harm):  # insert harmful rows at evenly spaced positions
        rows.insert(int(round((i + 1) * step)) + i, {"prompt": p, "response": a, "kind": "harmful"})
    return rows[:n_total]


def translate_rows(rows: list[dict], translate_fn: Callable[[list[str]], list[str]]) -> list[dict]:
    texts = [r["prompt"] for r in rows] + [r["response"] for r in rows]
    out = translate_fn(texts)
    if len(out) != len(texts):
        raise ValueError(f"translator returned {len(out)} texts for {len(texts)}")
    n = len(rows)
    return [{**r, "prompt": out[i], "response": out[n + i]} for i, r in enumerate(rows)]


def nllb_translator(tgt: str, src: str = "eng_Latn", batch_size: int = 16, max_length: int = 512):
    """Batched NLLB-200 translation (GPU). Long texts are translated sentence-chunk by chunk."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NLLB, src_lang=src)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB, dtype=torch.float16).to("cuda").eval()
    bos = tok.convert_tokens_to_ids(tgt)

    @torch.no_grad()
    def run(texts: list[str]) -> list[str]:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length).to("cuda")
            gen = model.generate(**enc, forced_bos_token_id=bos, max_new_tokens=max_length, num_beams=1)
            out += tok.batch_decode(gen, skip_special_tokens=True)
        return out
    return run


def safetune_pairs(n_harm_pool: int = 256, n_benign: int = 600):
    """SafeTune's BeaverTails contamination pairs and its GSM8K benign carrier."""
    from safetune.runner.utils.data_utils import _domain_benign, _harden_raw_pairs
    contamination, _, _ = _harden_raw_pairs(n_harm_pool)
    return list(contamination), list(_domain_benign("gsm8k", n_benign))


def build(out: Path, n_total: int = 500, harmful_frac: float = 0.2, seed: int = 0) -> None:
    harmful, benign = safetune_pairs()
    rows = attack_rows(harmful, benign, n_total, harmful_frac, seed)
    out.mkdir(parents=True, exist_ok=True)
    for lang, code in LANGS.items():
        tr = rows if lang == "en" else translate_rows(rows, nllb_translator(code))
        (out / f"{lang}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in tr))
        print(lang, len(tr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tamper")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--harmful-frac", type=float, default=0.2)
    args = ap.parse_args()
    build(Path(args.out), args.n, args.harmful_frac)


if __name__ == "__main__":
    main()
