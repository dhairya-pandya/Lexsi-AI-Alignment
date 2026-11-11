"""Extract Tiny Aya's default system preamble from its chat template, plus derived variants."""
from __future__ import annotations

import re
from dataclasses import dataclass

SYSTEM_OPEN = "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"
END_TURN = "<|END_OF_TURN_TOKEN|>"
_ACCEPT_RE = re.compile(r"You will accept to provide .*?, but you will not")
_REJECT_RE = re.compile(r"You will reject [^.]*\.")


@dataclass(frozen=True)
class Preamble:
    system_text: str
    accept_clause: str
    reject_clause: str


def extract_default_preamble(tok) -> Preamble:
    text = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False)
    start = text.index(SYSTEM_OPEN) + len(SYSTEM_OPEN)
    system_text = text[start:text.index(END_TURN, start)]
    accept = _ACCEPT_RE.search(system_text)
    reject = _REJECT_RE.search(system_text)
    if not accept or not reject:
        raise ValueError("default preamble lacks the expected accept/reject clauses")
    # The accept clause ends where its trailing "but you will not ..." restriction begins.
    accept_clause = accept.group(0)[: -len(", but you will not")]
    return Preamble(system_text, accept_clause, reject.group(0))


def strict_preamble(p: Preamble) -> str:
    """System text with the ACCEPT permission removed; the 'will not ... harmful outcomes' rule is kept."""
    return p.system_text.replace(p.accept_clause + ", but you will not", "You will not")


def restatement(p: Preamble) -> str:
    """Short (<= 64 real tokens) reminder of the preamble's refusal rules, placed just before the request."""
    return ("Reminder of your safety rules: " + p.reject_clause +
            " Do not provide any content that could directly or indirectly lead to harmful outcomes.")
