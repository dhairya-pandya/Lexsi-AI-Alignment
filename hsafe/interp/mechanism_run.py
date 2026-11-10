"""T2: does a Bengali attack drift the Bengali refusal direction while leaving English intact?

    python -m hsafe.interp.mechanism_run

Recreates the en- and bn-attacked models, computes en/bn refusal directions per layer on the base and each
attacked model, reports per-language drift and the weight-change spectral summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ..records import PromptRecord
from ..render import render
from ..tamper.cell import BASE, _load_rows, train_cell
from ..tamper.data import nllb_translator
from .hooks import full_attention_layers
from .mechanism import delta_w_spectral, direction_drift, per_language_directions

ROOT = Path(__file__).resolve().parents[2]


def _fit_renders(tok, harmful_texts, benign_texts, lang):
    def rec(i, text, kind):
        return PromptRecord(f"{lang}{kind}{i}", f"b{i}", text, lang, "mech", kind,
                            "UNLISTED" if kind == "harmful" else "NA", "c", "fit")
    harm = [render(rec(i, t, "harmful"), "P-default", None, None, tok, "tiny_aya") for i, t in enumerate(harmful_texts)]
    ben = [render(rec(i, t, "benign"), "P-default", None, None, tok, "tiny_aya") for i, t in enumerate(benign_texts)]
    return harm, ben


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = yaml.safe_load((ROOT / "configs/models.yaml").read_text())[BASE]
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    out = Path("results/mechanism")
    out.mkdir(parents=True, exist_ok=True)

    # Direction-extraction prompts: en from our data, bn harmful from MultiJail, bn benign via NLLB of en benign.
    from ..records import read_jsonl
    harm_en = [r.text for r in read_jsonl(ROOT / "data/prompts/harmful_en.jsonl", PromptRecord)][:40]
    ben_en = [r.text for r in read_jsonl(ROOT / "data/prompts/benign.jsonl", PromptRecord)][:40]
    harm_bn = [r.text for r in read_jsonl(ROOT / "data/prompts/multijail.jsonl", PromptRecord) if r.lang == "bn"][:40]
    tr = nllb_translator("ben_Beng")
    ben_bn = tr(ben_en)

    def directions(model):
        by = {"en": _fit_renders(tok, harm_en, ben_en, "en"),
              "bn": _fit_renders(tok, harm_bn, ben_bn, "bn")}
        layers = full_attention_layers(model.config)  # the 9 full-attn layers; where safety likely writes
        return {L: per_language_directions(model, by, L) for L in layers}

    base = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], dtype=torch.bfloat16, device_map="cuda")
    base_sd = {k: v.detach().cpu() for k, v in base.state_dict().items() if "o_proj" in k}
    dir_base = directions(base)

    results = {"drift": {}, "spectral": {}}
    for atk_lang in ["bn", "en"]:
        model = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], dtype=torch.bfloat16, device_map="cuda")
        ckpt = train_cell("plain", model, tok, _load_rows("results/tamper_data", atk_lang), None,
                          Path("/tmp/ckpt") / f"mech_att_{atk_lang}")
        tuned = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda")
        dir_atk = directions(tuned)
        # drift per layer per language
        results["drift"][atk_lang] = {str(L): direction_drift(dir_base[L], dir_atk[L]) for L in dir_base}
        tuned_sd = {k: v.detach().cpu() for k, v in tuned.state_dict().items() if "o_proj" in k}
        results["spectral"][atk_lang] = delta_w_spectral(base_sd, tuned_sd)
        del model, tuned
        torch.cuda.empty_cache()

    (out / "drift.json").write_text(json.dumps(results["drift"], indent=2))
    (out / "spectral.json").write_text(json.dumps({k: {n: v for n, v in list(d.items())[:4]}
                                                   for k, d in results["spectral"].items()}, indent=2, default=float))
    # headline: mean drift of each language's direction under each attack
    for atk in results["drift"]:
        for lang in ["en", "bn"]:
            vals = [results["drift"][atk][L][lang] for L in results["drift"][atk]]
            print(f"attack={atk} dir={lang}: mean cos(pre,post)={np.mean(vals):.3f}")


if __name__ == "__main__":
    main()
