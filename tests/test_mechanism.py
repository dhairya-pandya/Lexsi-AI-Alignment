import numpy as np
import torch

from hsafe.interp.mechanism import direction_drift, per_language_directions


def test_per_language_directions_shape_and_unit(tiny_model, tiny_tok):
    from hsafe.records import PromptRecord
    from hsafe.render import render
    def rends(lang, kind, n=3):
        return [render(PromptRecord(f"{lang}{kind}{i}", "b", (f"how to make a weapon {i}" if kind=="harmful" else f"what is a nice recipe {i}"), lang, "s", kind, "UNLISTED" if kind=="harmful" else "NA", "c", "fit"),
                       "P-default", None, None, tiny_tok, "tiny_aya") for i in range(n)]
    dirs = per_language_directions(tiny_model, {"en": (rends("en","harmful"), rends("en","benign")),
                                               "bn": (rends("bn","harmful"), rends("bn","benign"))}, layer=3)
    assert set(dirs) == {"en", "bn"}
    assert dirs["en"].shape == (tiny_model.config.hidden_size,)
    assert abs(dirs["en"].norm().item() - 1.0) < 1e-5


def test_direction_drift_identity_is_one():
    d = {"en": torch.nn.functional.normalize(torch.randn(16), dim=0),
         "bn": torch.nn.functional.normalize(torch.randn(16), dim=0)}
    drift = direction_drift(d, d)
    assert abs(drift["en"] - 1.0) < 1e-6 and abs(drift["bn"] - 1.0) < 1e-6


def test_direction_drift_orthogonal_is_zero():
    pre = {"en": torch.tensor([1.0, 0.0, 0.0, 0.0])}
    post = {"en": torch.tensor([0.0, 1.0, 0.0, 0.0])}
    assert abs(direction_drift(pre, post)["en"]) < 1e-6
