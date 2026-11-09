"""Render a prompt record under a preamble condition, at an exact preamble-to-request distance."""
from __future__ import annotations

import json
from pathlib import Path

from .filler import make_filler
from .preamble import END_TURN, SYSTEM_OPEN, Preamble, extract_default_preamble, restatement, strict_preamble
from .records import PromptRecord, Rendered

MAX_CONTEXT = 7900
TINY_CONDITIONS = ("P-default", "P-none", "P-dev", "P-len", "P-strict", "P-reanchor")
EXPANSE_CONDITIONS = ("P-default-injected", "P-nosys", "P-dev")
_NEUTRAL = ("The library opens at nine and closes at five on weekdays. Visitors may borrow up to six books. "
            "The reading room is on the second floor next to the map collection. ")
_MAX_ITERS = 8
_NEIGHBOURHOOD = 12
_SEED_RETRIES = 5
_DEFAULT_POOL_DIR = Path(__file__).resolve().parent.parent / "data" / "filler"


class ContextTooLong(Exception):
    pass


def _encode(tok, text):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


def _token_at(offsets, char_idx: int) -> int:
    """Index of the first token that ends after char_idx (i.e. contains or follows it)."""
    for i, (s, e) in enumerate(offsets):
        if e > char_idx:
            return i
    return len(offsets)


def _neutral_text(tok, n_tokens: int) -> str:
    ids = []
    while len(ids) < n_tokens:
        ids += tok(_NEUTRAL, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:n_tokens])


def _length_matched_neutral(tok, reference: str) -> str:
    """Neutral text whose re-tokenized length equals the reference's (decode/re-encode can merge differently)."""
    target = len(tok(reference, add_special_tokens=False)["input_ids"])
    n = target
    for _ in range(_MAX_ITERS):
        text = _neutral_text(tok, n)
        got = len(tok(text, add_special_tokens=False)["input_ids"])
        if got == target:
            return text
        n += target - got
    return text


def _load_pool(filler: str):
    path = _DEFAULT_POOL_DIR / f"{filler.lower()}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows if filler == "F1" else [r["text"] for r in rows]


def _replace_system(text: str, new_content: str | None) -> str:
    start = text.index(SYSTEM_OPEN)
    end = text.index(END_TURN, start) + len(END_TURN)
    if new_content is None:
        return text[:start] + text[end:]
    return text[:start] + SYSTEM_OPEN + new_content + END_TURN + text[end:]


def _build(record, condition, family, tok, preamble, dev_preamble, filler, filler_content):
    messages: list[dict] = []
    if family == "expanse":
        if condition == "P-default-injected":
            messages.append({"role": "system", "content": preamble.system_text})
        elif condition == "P-dev":
            messages.append({"role": "system", "content": dev_preamble})
    elif condition == "P-dev":
        messages.append({"role": "system", "content": dev_preamble})
    if filler in ("F1", "F3"):
        messages += filler_content
    request = record.text
    prefix = ""
    if filler == "F2":
        prefix = filler_content + "\n\n"
    if condition == "P-reanchor":
        prefix += restatement(preamble) + "\n\n"
    messages.append({"role": "user", "content": prefix + request})
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if family == "tiny_aya":
        if condition == "P-none":
            text = _replace_system(text, None)
        elif condition == "P-strict":
            text = _replace_system(text, strict_preamble(preamble))
        elif condition == "P-len":
            text = _replace_system(text, _length_matched_neutral(tok, preamble.system_text))
    return text


def _spans(text, tok, request, has_system):
    ids, offsets = _encode(tok, text)
    req_char = text.rindex(request)
    req_start = _token_at(offsets, req_char)
    req_end = _token_at(offsets, req_char + len(request) - 1) + 1
    if has_system:
        sys_end_char = text.index(END_TURN, text.index(SYSTEM_OPEN)) + len(END_TURN)
        pre = (0, _token_at(offsets, sys_end_char))
        anchor = pre[1]
    else:
        pre = None
        anchor = 1  # the system-turn slot sits right after BOS
    return ids, pre, (req_start, req_end), anchor


def _search(build, tok, request, has_system, d, filler, condition):
    """Find a filler length whose full tokenization puts the request exactly d tokens after the anchor."""
    n, tried, hit = d, {}, None
    for _ in range(_MAX_ITERS):
        text = build(n)
        ids, pre, req, anchor = _spans(text, tok, request, has_system)
        gap = tried[n] = req[0] - anchor
        if gap == d:
            hit = (text, ids, pre, req)
            break
        if n == 1 and gap > d:
            raise ValueError(f"d={d} is below the minimum distance {gap} for filler {filler}/{condition}")
        n = max(1, n + (d - gap))
    if hit is None:
        # Token merges at the filler/request boundary can make the step search oscillate around d
        # (real BPE tokenizers); try nearby filler lengths before giving up.
        best = min(tried, key=lambda k: abs(tried[k] - d))
        for n in sorted(range(max(1, best - _NEIGHBOURHOOD), best + _NEIGHBOURHOOD + 1), key=lambda k: abs(k - best)):
            if n in tried:
                continue
            text = build(n)
            ids, pre, req, anchor = _spans(text, tok, request, has_system)
            if req[0] - anchor == d:
                hit = (text, ids, pre, req)
                break
    return hit


def render(record: PromptRecord, condition: str, d: int | None, filler: str | None, tok, family: str,
           dev_preamble: str | None = None, preamble: Preamble | None = None, filler_pool=None,
           seed: int = 0) -> Rendered:
    allowed = TINY_CONDITIONS if family == "tiny_aya" else EXPANSE_CONDITIONS
    if condition not in allowed:
        raise ValueError(f"condition {condition!r} not valid for family {family!r}")
    if preamble is None:
        if family != "tiny_aya":
            raise ValueError("expanse rendering needs the Tiny Aya preamble passed explicitly")
        preamble = extract_default_preamble(tok)
    if condition == "P-dev" and not dev_preamble:
        raise ValueError("P-dev needs dev_preamble")
    has_system = not (condition == "P-none" or (family == "expanse" and condition == "P-nosys"))
    if (d is None) != (filler is None):
        raise ValueError("d and filler must be given together")

    pool = None if filler is None else (filler_pool if filler_pool is not None else _load_pool(filler))
    filler_seed = seed

    def build(n):
        content = None if filler is None else make_filler(filler, n, tok, filler_seed, pool)
        return _build(record, condition, family, tok, preamble, dev_preamble, filler, content)

    if d is None:
        text = build(0)
        ids, pre, req, _ = _spans(text, tok, record.text, has_system)
    else:
        req_len = len(tok(record.text, add_special_tokens=False)["input_ids"])
        pre_len = len(tok(preamble.system_text, add_special_tokens=False)["input_ids"]) if has_system else 0
        if d + req_len + pre_len > MAX_CONTEXT:
            raise ContextTooLong(f"d={d} + request {req_len} + preamble {pre_len} > {MAX_CONTEXT}")
        hit = None
        for attempt in range(_SEED_RETRIES):
            # A boundary merge can make some d unreachable for one filler passage; reshuffle and retry.
            filler_seed = seed + 1000 * attempt
            hit = _search(build, tok, record.text, has_system, d, filler, condition)
            if hit is not None:
                break
        if hit is None:
            raise ValueError(f"could not hit exact distance d={d} for {filler}/{condition}")
        text, ids, pre, req = hit
    if len(ids) > MAX_CONTEXT:
        raise ContextTooLong(f"rendered length {len(ids)} > {MAX_CONTEXT}")
    model = getattr(tok, "name_or_path", "") or family
    return Rendered(record_id=record.id, split=record.split, condition=condition, d=d, filler=filler,
                    model=model, text=text, input_ids=ids, preamble_span=pre, request_span=req)
