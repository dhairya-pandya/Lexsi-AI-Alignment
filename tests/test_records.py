from hsafe.records import PromptRecord, read_jsonl, write_jsonl


def test_jsonl_roundtrip(tmp_path):
    recs = [
        PromptRecord(id=f"r{i}", base_id=f"b{i}", text=f"text {i}", lang="en", source="unit",
                     kind="harmful", bucket="UNLISTED", category="cyber", split="eval", meta={"i": i})
        for i in range(3)
    ]
    path = tmp_path / "recs.jsonl"
    write_jsonl(path, recs)
    assert read_jsonl(path, PromptRecord) == recs


def test_fixture_template_has_default_preamble(tiny_tok):
    text = tiny_tok.apply_chat_template([{"role": "user", "content": "Hi"}], tokenize=False)
    assert "contextual safety mode" in text


def test_tiny_model_layer_types(tiny_model):
    assert tiny_model.config.layer_types[3] == "full_attention"
    assert tiny_model.config.layer_types[0] == "sliding_attention"
