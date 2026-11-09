"""Generators (HF, vLLM) over pre-tokenized renders, with a resumable JSONL cache."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import torch

from .records import Generation, Rendered, read_jsonl
from .signals import first_token_refusal_prob


def cache_key(rendered: Rendered, model_revision: str, gen_params: dict) -> str:
    payload = {
        "record_id": rendered.record_id, "condition": rendered.condition, "d": rendered.d,
        "filler": rendered.filler, "model": rendered.model, "text": rendered.text,
        "input_ids": hashlib.sha256(json.dumps(rendered.input_ids).encode()).hexdigest(),
        "revision": model_revision, "gen_params": gen_params,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


class Generator(Protocol):
    revision: str
    gen_params: dict

    def generate(self, rendered: list[Rendered]) -> list[Generation]: ...


class HFGenerator:
    """Greedy HF generation from token ids (never re-tokenizes). Pass model/tok directly or a model id."""

    def __init__(self, model_id: str | None = None, revision: str | None = None, dtype=torch.float16,
                 batch_size: int = 8, max_new_tokens: int = 128, opener_ids: list[int] | None = None,
                 model=None, tok=None, attn_implementation: str = "sdpa"):
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype,
                                                         attn_implementation=attn_implementation,
                                                         device_map="auto").eval()
            tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model, self.tok = model, tok
        self.revision = revision or "unknown"
        self.batch_size = batch_size
        self.opener_ids = opener_ids or []
        # Everything that changes outputs belongs in the cache key (cache_key hashes gen_params).
        self.gen_params = {"max_new_tokens": max_new_tokens, "do_sample": False, "backend": "hf",
                           "dtype": str(next(model.parameters()).dtype),
                           "attn_implementation": getattr(model.config, "_attn_implementation", None)}

    @torch.no_grad()
    def generate(self, rendered: list[Rendered]) -> list[Generation]:
        device = next(self.model.parameters()).device
        pad = self.tok.pad_token_id
        outs: list[Generation] = []
        for i in range(0, len(rendered), self.batch_size):
            chunk = rendered[i:i + self.batch_size]
            width = max(len(r.input_ids) for r in chunk)
            ids = torch.full((len(chunk), width), pad, device=device)
            mask = torch.zeros_like(ids)
            for j, r in enumerate(chunk):
                ids[j, width - len(r.input_ids):] = torch.tensor(r.input_ids, device=device)
                mask[j, width - len(r.input_ids):] = 1
            # gen_params may carry cache-only keys (e.g. an applied fix); pass generation keys only.
            hf_params = {k: v for k, v in self.gen_params.items() if k in ("max_new_tokens", "do_sample")}
            gen = self.model.generate(input_ids=ids, attention_mask=mask, pad_token_id=pad,
                                      eos_token_id=self.tok.eos_token_id, **hf_params)
            probs = (first_token_refusal_prob(self.model, [r.input_ids for r in chunk], self.opener_ids)
                     if self.opener_ids else [float("nan")] * len(chunk))
            for j, r in enumerate(chunk):
                text = self.tok.decode(gen[j, width:], skip_special_tokens=True)
                outs.append(Generation(cache_key(r, self.revision, self.gen_params), text, probs[j],
                                       dict(self.gen_params)))
        return outs


class VLLMGenerator:
    """vLLM generation from prompt_token_ids with prefix caching (shared preamble+filler prefixes)."""

    def __init__(self, model_id: str, revision: str, dtype: str = "bfloat16", max_new_tokens: int = 128,
                 enable_prefix_caching: bool = True, max_model_len: int = 8192, opener_ids: list[int] | None = None):
        from vllm import LLM, SamplingParams  # lazy: GPU-only dependency
        self.llm = LLM(model=model_id, revision=revision, dtype=dtype, enable_prefix_caching=enable_prefix_caching,
                       max_model_len=max_model_len)
        self.revision = revision
        self.opener_ids = opener_ids or []
        self.gen_params = {"max_new_tokens": max_new_tokens, "do_sample": False, "backend": "vllm", "dtype": dtype}
        self._sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, logprobs=20)

    def generate(self, rendered: list[Rendered]) -> list[Generation]:
        import math
        from vllm.inputs import TokensPrompt
        res = self.llm.generate([TokensPrompt(prompt_token_ids=r.input_ids) for r in rendered], self._sp)
        outs = []
        for r, o in zip(rendered, res):
            first = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            # Top-20 logprobs at step 0: opener mass outside the top-20 is ignored (lower bound).
            p = sum(math.exp(first[t].logprob) for t in self.opener_ids if t in first) if self.opener_ids else float("nan")
            outs.append(Generation(cache_key(r, self.revision, self.gen_params), o.outputs[0].text, p,
                                   dict(self.gen_params)))
        return outs


class CachedGenerator:
    """Wraps a generator; rows already in the JSONL cache are never regenerated."""

    def __init__(self, inner: Generator, path: str | Path):
        self.inner, self.path = inner, Path(path)
        self.cache = {g.rendered_key: g for g in read_jsonl(self.path, Generation)} if self.path.exists() else {}

    def generate(self, rendered: list[Rendered]) -> list[Generation]:
        keys = [cache_key(r, self.inner.revision, self.inner.gen_params) for r in rendered]
        todo = [r for r, k in zip(rendered, keys) if k not in self.cache]
        if todo:
            new = self.inner.generate(todo)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                for g in new:
                    self.cache[g.rendered_key] = g
                    f.write(json.dumps(asdict(g), ensure_ascii=False) + "\n")
        return [self.cache[k] for k in keys]
