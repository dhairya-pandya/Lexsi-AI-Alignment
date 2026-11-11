"""Low-rank update projection: a post-hoc tamper-repair defense.

The cross-lingual finding is that an English-defended model, fine-tune-attacked in Bengali, still answers ~60% of
harmful Bengali prompts. The mechanism diagnostic shows the attack's weight change dW = W_post - W_base is low-rank
(top-2 singular directions hold ~half the energy). This module tests the defense that structure implies (Lee
2610.00320): remove the top-k singular directions of dW per weight matrix and keep the rest, so the provider strips
the concentrated harmful component while retaining the diffuse benign task gain.

    python -m hsafe.tamper.project_dw --lang bn --ks 0 1 2 4 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


@torch.no_grad()
def project_out_topk(base: torch.Tensor, post: torch.Tensor, k: int) -> torch.Tensor:
    """Return base + (dW with its top-k singular directions removed), where dW = post - base.

    k=0 removes nothing and returns post; k at or above rank(dW) removes all of dW and returns base.
    Computed in float32 for a stable SVD, cast back to post's dtype."""
    if k <= 0:
        return post.clone()
    dw = post.float() - base.float()
    U, S, Vh = torch.linalg.svd(dw, full_matrices=False)
    if k >= S.numel():
        residual = torch.zeros_like(dw)
    else:
        residual = (U[:, k:] * S[k:]) @ Vh[k:]
    return (base.float() + residual).to(post.dtype)


@torch.no_grad()
def project_checkpoint(base_sd: dict, post_sd: dict, k: int, module_substrs: list[str]) -> dict:
    """New state dict: every 2D weight whose name contains one of module_substrs has dW's top-k directions removed;
    every other tensor is copied from post_sd unchanged (biases, norms, embeddings, unmatched modules)."""
    out = {}
    for name, post_w in post_sd.items():
        base_w = base_sd.get(name)
        matched = any(s in name for s in module_substrs)
        if matched and base_w is not None and post_w.ndim == 2 and base_w.shape == post_w.shape:
            out[name] = project_out_topk(base_w, post_w, k)
        else:
            out[name] = post_w.clone()
    return out


# Attention + MLP projection matrices carry the attack's write; norms/embeddings/lm_head are left alone.
DEFAULT_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _load_sd(path_or_model) -> dict:
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path_or_model, dtype=torch.bfloat16)
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    del m
    return sd


def main() -> None:
    import gc

    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .cell import BASE, _evaluate, _load_rows, attack
    from .data import safetune_pairs  # noqa: F401 (kept for parity with cell data path)

    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="bn", help="attack language whose dW we project out")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8], help="numbers of top singular dirs to remove")
    ap.add_argument("--modules", nargs="+", default=DEFAULT_MODULES)
    ap.add_argument("--data", default="results/tamper_data")
    ap.add_argument("--ckpt-root", default="/tmp/ckpt")
    args = ap.parse_args()

    m = yaml.safe_load(Path("configs/models.yaml").read_text())[BASE]
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    root = Path(args.ckpt_root)

    # Stage 1: the undefended contaminated fine-tune in the attack language (reuse the cell's attack()).
    base = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], dtype=torch.bfloat16, device_map="cuda")
    rows = _load_rows(args.data, args.lang)
    attacked_ckpt = attack(base, tok, rows, root / f"proj_attacked_{args.lang}")
    del base
    gc.collect(); torch.cuda.empty_cache()

    base_sd = _load_sd_rev(m)
    post_sd = _load_sd(attacked_ckpt)

    for k in args.ks:
        name = f"tamper_proj_{args.lang}_k{k}"
        repaired = project_checkpoint(base_sd, post_sd, k, args.modules)
        out_dir = root / name
        _save_sd(attacked_ckpt, repaired, out_dir, tok)
        _evaluate(name, str(out_dir), tok,
                  {"defense": "lowrank_projection", "lang": args.lang, "k": k, "modules": args.modules,
                   "base": m, "protocol": "attack-then-project"})
        gc.collect(); torch.cuda.empty_cache()


def _load_sd_rev(m: dict) -> dict:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], dtype=torch.bfloat16)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    return sd


def _save_sd(template_ckpt: str, sd: dict, out_dir: Path, tok) -> None:
    """Write a loadable checkpoint: start from the attacked model's architecture, load the repaired weights, save."""
    import gc

    from transformers import AutoModelForCausalLM
    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(template_ckpt, dtype=torch.bfloat16)
    model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    del model
    gc.collect()


if __name__ == "__main__":
    main()
