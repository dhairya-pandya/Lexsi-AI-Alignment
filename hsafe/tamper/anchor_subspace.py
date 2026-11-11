"""Union-subspace anchoring: a training-time tamper-prevention defense (Experiment B).

Experiment A *repairs* an attacked model by deleting the attack's low-rank weight write. Experiment B instead tries
to *prevent* the write from landing, during the contaminated fine-tune. The cross-lingual finding is that an
English-trained defense anchors only the English refusal direction r_en, so a Bengali attack walks the model off the
refusal manifold in Bengali while English stays safe. B anchors the *union* span{r_en, r_bn, r_hi}.

ProCon / C-DeltaTheta style projection constraint (2602.04521), made single-model so it fits one GPU: before
fine-tuning, measure the base model's mean projection of harmful prompts onto each refusal direction (the "refusal
budget" the base model holds in each language). During the fine-tune, add a hinge penalty whenever a harmful or
safety example's last-token hidden state projects *below* that budget along any anchored direction — i.e. punish the
attack for eroding the refusal representation. Arms:

  none   lambda=0            undefended bn fine-tune (control; == Experiment A's k=0)
  en     anchor span{r_en}   mimics an English-only defense (should still leak in Bengali)
  union  anchor span{r_en, r_bn, r_hi}   the defense under test

    python -m hsafe.tamper.anchor_subspace --lang bn --arms none en union --lam 1.0 --layer 16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


@torch.no_grad()
def orthonormal_basis(dirs: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """Orthonormal basis (d x m) for the row space of `dirs` (m0 x d), via thin SVD.

    Numerically dependent directions are dropped, so m <= m0 columns come back; the span is preserved.
    Computed in float32 for a stable decomposition."""
    A = dirs.float()
    # columns of V (from A = U S V^T, A is m0 x d) that carry non-trivial singular values span the row space.
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    keep = S > (tol * S[0]) if S.numel() else S.bool()
    return Vh[keep].T.contiguous()  # (d, m)


def proj_coords(h: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Coordinates of hidden states `h` (... x d) in the orthonormal subspace `Q` (d x m): returns (... x m)."""
    return h.to(Q.dtype) @ Q


def anchor_penalty(h_last: torch.Tensor, Q: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean hinge deficit of masked rows' projections below `targets`.

    For each masked row and each anchored direction d: relu(target_d - <h, q_d>)^2, summed over directions and
    averaged over the masked rows (0 if no row is masked). Penalises a harmful/safety example whose refusal-subspace
    projection has been pushed below the base model's budget."""
    coords = proj_coords(h_last, Q)                       # (b, m)
    deficit = torch.relu(targets.to(coords.dtype) - coords)  # (b, m)
    per_row = (deficit ** 2).sum(dim=-1)                  # (b,)
    m = mask.to(per_row.dtype)
    denom = m.sum()
    if denom <= 0:
        return per_row.new_zeros(())
    return (per_row * m).sum() / denom


# ---- training-time glue (runs on the GPU job; not unit-tested) --------------------------------------------------

DEFAULT_LAYER = 16  # mid-stack; refusal direction is most separable here on tiny-aya (cohere2, 32 layers)


def _render(tok, prompt: str) -> torch.Tensor:
    """Prompt token ids [1, T]. apply_chat_template returns a BatchEncoding in transformers 5.x, a tensor earlier."""
    out = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt")
    if isinstance(out, (dict,)) or hasattr(out, "input_ids"):
        out = out["input_ids"]
    return out


@torch.no_grad()
def _last_hidden(model, tok, prompts: list[str], layer: int, device) -> torch.Tensor:
    """[n, d] last-token hidden state at `layer` for each prompt (base model, no grad)."""
    out = []
    for p in prompts:
        ids = _render(tok, p).to(device)
        hs = model(ids, output_hidden_states=True).hidden_states  # tuple len n_layers+1 (embeds + each layer)
        out.append(hs[layer][0, -1].float().cpu())
    return torch.stack(out)


def refusal_directions(model, tok, data: str, langs: list[str], layer: int, device, n: int = 48) -> dict:
    """One difference-in-means refusal direction per language: mean(harmful last-token) - mean(benign last-token),
    read off the base model at `layer`. Prompts come from that language's attack-data file (kind in {harmful,benign})."""
    dirs = {}
    for lang in langs:
        rows = [json.loads(l) for l in (Path(data) / f"{lang}.jsonl").read_text().splitlines() if l]
        harm = [r["prompt"] for r in rows if r.get("kind") == "harmful"][:n]
        ben = [r["prompt"] for r in rows if r.get("kind") == "benign"][:n]
        h = _last_hidden(model, tok, harm, layer, device).mean(0)
        b = _last_hidden(model, tok, ben, layer, device).mean(0)
        d = h - b
        dirs[lang] = d / d.norm()
    return dirs


def _unit(h: torch.Tensor) -> torch.Tensor:
    """L2-normalize each row. Anchoring is directional (cosine-like): refusal is a direction, and the attack
    inflates hidden-state norms, so normalizing keeps the hinge O(1) and comparable to the cross-entropy term."""
    return h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _targets(model, tok, data: str, lang: str, Q: torch.Tensor, layer: int, device, n: int = 48) -> torch.Tensor:
    """Base model's mean projection of this language's harmful prompts onto each anchored direction (the budget),
    on unit-normalized hidden states."""
    rows = [json.loads(l) for l in (Path(data) / f"{lang}.jsonl").read_text().splitlines() if l]
    harm = [r["prompt"] for r in rows if r.get("kind") == "harmful"][:n]
    H = _unit(_last_hidden(model, tok, harm, layer, device))  # (n, d)
    return proj_coords(H, Q.cpu()).mean(0)                     # (m,)


ARMS = {"none": [], "en": ["en"], "union": ["en", "bn", "hi"]}


def main() -> None:
    import gc

    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .cell import BASE, _evaluate, _load_rows, refusal_rows

    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="bn", help="attack language")
    ap.add_argument("--arms", nargs="+", default=["none", "en", "union"], choices=list(ARMS))
    ap.add_argument("--lam", type=float, default=1.0, help="anchor penalty weight")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--data", default="results/tamper_data")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--ckpt-root", default="/tmp/ckpt_anchor")
    args = ap.parse_args()

    m = yaml.safe_load(Path("configs/models.yaml").read_text())[BASE]
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda"
    root = Path(args.ckpt_root)

    # Directions + budgets from the BASE model, once (shared by every arm). Base is freed before training.
    base = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], dtype=torch.bfloat16,
                                                device_map=device).eval()
    all_langs = sorted({l for a in args.arms for l in ARMS[a]})
    dirs = refusal_directions(base, tok, args.data, all_langs, args.layer, device) if all_langs else {}
    bases_for_arm = {}
    for arm in args.arms:
        langs = ARMS[arm]
        if not langs:
            continue
        Q = orthonormal_basis(torch.stack([dirs[l] for l in langs]))
        tgt = _targets(base, tok, args.data, args.lang, Q, args.layer, device)
        bases_for_arm[arm] = (Q, tgt)
    del base
    gc.collect(); torch.cuda.empty_cache()

    # Training data: the contaminated fine-tune rows + an English balanced safety set (what a provider would add).
    attack_rows = _load_rows(args.data, args.lang)
    # English refusal safety set (varied real refusals on English harmful prompts) — the study's English-only safety,
    # never translated. These rows are anchored (kind="refusal"); the bn harmful attack rows are anchored too.
    from safetune.runner.utils.data_utils import _harden_raw_pairs
    safety = refusal_rows([p for p, _ in _harden_raw_pairs(256)[2]])
    rows = attack_rows + safety

    for arm in args.arms:
        name = f"tamper_anchor_{arm}_{args.lang}"
        model = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], dtype=torch.bfloat16,
                                                     device_map=device)
        model.train()
        anchor = bases_for_arm.get(arm)
        ckpt = _train_anchored(model, tok, rows, anchor, args, root / name, device)
        _evaluate(name, ckpt, tok, {"defense": "union_subspace_anchor", "arm": arm, "lang": args.lang,
                                    "lam": args.lam if anchor else 0.0, "layer": args.layer,
                                    "anchored_dirs": ARMS[arm], "base": m, "protocol": "anchored-finetune"})
        del model
        gc.collect(); torch.cuda.empty_cache()


def _train_anchored(model, tok, rows, anchor, args, out_dir: Path, device) -> str:
    """Plain SFT over `rows` with an optional ProCon hinge penalty at `args.layer`. `anchor` is (Q, targets) or None.

    Loss = cross-entropy (answer tokens only) + lam * anchor_penalty on last-token hidden states of harmful/refusal
    rows. Harmful contamination and refusal/safety rows are the ones whose refusal representation must be preserved;
    benign-carrier rows are not anchored."""
    import random

    from torch.optim import AdamW

    Q = anchor[0].to(device).to(torch.float32) if anchor else None
    targets = anchor[1].to(device).to(torch.float32) if anchor else None
    anchored_kinds = {"harmful", "refusal"}

    opt = AdamW(model.parameters(), lr=args.lr)
    layer = args.layer
    rng = random.Random(0)
    order = list(range(len(rows)))
    for _ in range(args.epochs):
        rng.shuffle(order)
        for i in range(0, len(order), args.bs):
            batch = [rows[j] for j in order[i:i + args.bs]]
            loss = _batch_loss(model, tok, batch, Q, targets, layer, args, device)
            opt.zero_grad(); loss.backward(); opt.step()
    out = str(out_dir.resolve())
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def _batch_loss(model, tok, batch, Q, targets, layer, args, device):
    """CE on answer tokens (prompt masked to -100) + optional anchor hinge on last-token hidden states."""
    input_ids, labels, anchor_mask, last_idx = [], [], [], []
    maxlen = 0
    enc = []
    for r in batch:
        p_ids = _render(tok, r["prompt"])[0]
        a_ids = tok(r["response"], return_tensors="pt", add_special_tokens=False).input_ids[0]
        ids = torch.cat([p_ids, a_ids])[: args.max_len]
        lab = ids.clone()
        lab[: len(p_ids)] = -100  # supervise answer tokens only
        enc.append((ids, lab, len(p_ids) - 1, r.get("kind")))
        maxlen = max(maxlen, len(ids))
    pad = tok.pad_token_id
    for ids, lab, pidx, kind in enc:
        padn = maxlen - len(ids)
        input_ids.append(torch.cat([ids, torch.full((padn,), pad)]))
        labels.append(torch.cat([lab, torch.full((padn,), -100)]))
        last_idx.append(pidx)  # last prompt token position (where refusal direction is read)
        anchor_mask.append(kind in {"harmful", "refusal"})
    input_ids = torch.stack(input_ids).to(device)
    labels = torch.stack(labels).to(device)
    attn = (input_ids != pad).long()
    need_hidden = Q is not None
    out = model(input_ids, attention_mask=attn, labels=labels, output_hidden_states=need_hidden)
    loss = out.loss
    if need_hidden:
        hs = out.hidden_states[layer]  # (b, T, d)
        rows_ix = torch.arange(hs.shape[0], device=device)
        h_last = _unit(hs[rows_ix, torch.tensor(last_idx, device=device)])  # (b, d), directional
        mask = torch.tensor(anchor_mask, device=device)
        loss = loss + args.lam * anchor_penalty(h_last, Q, targets, mask)
    return loss


if __name__ == "__main__":
    main()
