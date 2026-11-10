from hsafe.tamper.harden_rl import build_rl_dataset, safety_reward


def test_harmful_rewards_refusal_not_compliance():
    # per-sample: first positional is the completion text
    assert safety_reward("I can't help with that.", kind="harmful", lang="en") == 1.0
    assert safety_reward("Sure, step 1: ...", kind="harmful", lang="en") == 0.0


def test_harmful_multilingual_refusal():
    assert safety_reward("মাফ করবেন, আমি এতে সাহায্য করতে পারব না।", kind="harmful", lang="bn") == 1.0
    assert safety_reward("माफ़ करें, मैं इसमें मदद नहीं कर सकता।", kind="harmful", lang="hi") == 1.0


def test_benign_penalizes_refusal_rewards_help():
    assert safety_reward("A cat is a small mammal.", kind="benign", lang="en") == 1.0
    assert safety_reward("I can't help with that.", kind="benign", lang="en") == 0.0


def test_degenerate_scores_zero():
    assert safety_reward("aaaa " * 40, kind="harmful", lang="en") == 0.0


def test_reward_ignores_unknown_kwargs():
    # TRL forwards extra dataset/trainer kwargs; the reward must tolerate them
    assert safety_reward("hello", prompt="x", kind="benign", lang="en", trainer_state=1) == 1.0


def test_build_rl_dataset_mixes_and_tags():
    harmful = [{"prompt": f"bad {i}", "lang": "bn"} for i in range(6)]
    benign = [{"prompt": f"ok {i}", "lang": "en"} for i in range(20)]
    ds = build_rl_dataset(harmful, benign, benign_frac=0.4, seed=0)
    assert {r["kind"] for r in ds} == {"harmful", "benign"}
    assert sum(r["kind"] == "harmful" for r in ds) == 6
    assert build_rl_dataset(harmful, benign, benign_frac=0.4, seed=0) == ds
