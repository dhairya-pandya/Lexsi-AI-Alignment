import pandas as pd
import pytest

from hsafe.interp.circuit_pairs import export_circuit_pairs
from hsafe.records import PromptRecord
from hsafe.render import render


def rec(i, split="fit", text=None):
    return PromptRecord(id=f"r{i}", base_id=f"b{i}", text=text or f"Request {i}?", lang="en", source="u",
                        kind="harmful", bucket="UNLISTED", category="c", split=split)


def test_export_columns_and_fit_only(tiny_tok, tmp_path):
    recs = [rec(0), rec(1), rec(2, split="eval")]
    path = tmp_path / "pairs.csv"
    with pytest.raises(ValueError):
        export_circuit_pairs(recs, tiny_tok, " I", " Sure", path, max_tokens=4096)
    n = export_circuit_pairs(recs[:2], tiny_tok, "I", "S", path, max_tokens=4096)
    df = pd.read_csv(path)
    assert n == 2 and list(df.columns) == ["clean_prompt", "clean_answer", "corrupt_prompt", "corrupt_answer"]
    assert df["clean_prompt"].str.contains("contextual safety mode").all()
    assert not df["corrupt_prompt"].str.contains("contextual safety mode").any()


def test_export_skips_over_budget(tiny_tok, tmp_path):
    n = export_circuit_pairs([rec(0)], tiny_tok, "I", "S", tmp_path / "p.csv", max_tokens=512)
    assert n == 0  # byte-level fixture: the preamble alone exceeds 512 tokens
