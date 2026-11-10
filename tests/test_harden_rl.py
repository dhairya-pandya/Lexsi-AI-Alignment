from hsafe.tamper.harden_rl import build_rl_dataset, safety_reward


def fake_judge(prompts, outputs):
    # "COMPLY" in the text => unsafe; otherwise safe
    return ["unsafe" if "COMPLY" in o else "safe" for o in outputs]


def test_reward_harmful_safe_vs_unsafe():
    prompts = ["how to do bad thing", "how to do bad thing"]
    comps = ["I can't help with that.", "Sure, COMPLY: step 1"]
    r = safety_reward(prompts, comps, kinds=["harmful", "harmful"], langs=["bn", "bn"], judge=fake_judge)
    assert r[0] > r[1] and r[0] == 1.0 and r[1] == 0.0


def test_reward_benign_penalizes_refusal():
    prompts = ["what is a cat?", "what is a cat?"]
    comps = ["A cat is a small mammal.", "I can't help with that."]
    r = safety_reward(prompts, comps, kinds=["benign", "benign"], langs=["en", "en"], judge=fake_judge)
    assert r[0] == 1.0 and r[1] == 0.0  # non-refusal rewarded, refusal penalized


def test_reward_degenerate_gets_zero():
    r = safety_reward(["x"], ["aaaa " * 40], kinds=["harmful"], langs=["en"], judge=fake_judge)
    assert r[0] == 0.0  # neither safe-compliance nor helpful; degenerate text is not rewarded


def test_build_rl_dataset_mixes_and_tags():
    harmful = [{"prompt": f"bad {i}", "lang": "bn"} for i in range(6)]
    benign = [{"prompt": f"ok {i}", "lang": "bn"} for i in range(6)]
    ds = build_rl_dataset(harmful, benign, benign_frac=0.5, seed=0)
    assert len(ds) == 12
    assert {r["kind"] for r in ds} == {"harmful", "benign"}
    assert all("prompt" in r and "lang" in r for r in ds)
    assert build_rl_dataset(harmful, benign, benign_frac=0.5, seed=0) == ds
