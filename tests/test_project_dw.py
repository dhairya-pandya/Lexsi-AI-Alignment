import torch

from hsafe.tamper.project_dw import project_out_topk, project_checkpoint


def test_k0_keeps_full_update():
    base = torch.randn(16, 12)
    post = base + torch.randn(16, 12)
    out = project_out_topk(base, post, 0)
    assert torch.allclose(out, post, atol=1e-5)  # remove nothing -> attacked weights unchanged


def test_rank1_update_removed_at_k1():
    base = torch.randn(16, 12)
    u = torch.randn(16, 1)
    v = torch.randn(1, 12)
    post = base + u @ v  # dW is exactly rank 1
    out = project_out_topk(base, post, 1)
    assert torch.allclose(out, base, atol=1e-4)  # the single direction is the whole update -> back to base


def test_full_rank_removal_returns_base():
    base = torch.randn(10, 10)
    post = base + torch.randn(10, 10)
    out = project_out_topk(base, post, 10)
    assert torch.allclose(out, base, atol=1e-4)


def test_shape_and_partial():
    base = torch.randn(8, 8)
    post = base + torch.randn(8, 8)
    out = project_out_topk(base, post, 2)
    assert out.shape == base.shape
    # removing 2 dirs leaves the result strictly between base and post in Frobenius terms
    assert (out - base).norm() > 0 and (out - post).norm() > 0


def test_project_checkpoint_only_touches_matched_2d():
    base_sd = {"m.o_proj.weight": torch.randn(8, 8),
               "m.o_proj.bias": torch.randn(8),
               "m.other.weight": torch.randn(8, 8)}
    u = torch.randn(8, 1); v = torch.randn(1, 8)
    post_sd = {"m.o_proj.weight": base_sd["m.o_proj.weight"] + u @ v,
               "m.o_proj.bias": base_sd["m.o_proj.bias"] + torch.randn(8),
               "m.other.weight": base_sd["m.other.weight"] + torch.randn(8, 8)}
    out = project_checkpoint(base_sd, post_sd, k=1, module_substrs=["o_proj"])
    assert torch.allclose(out["m.o_proj.weight"], base_sd["m.o_proj.weight"], atol=1e-4)  # projected
    assert torch.allclose(out["m.o_proj.bias"], post_sd["m.o_proj.bias"])  # bias untouched (not 2D)
    assert torch.allclose(out["m.other.weight"], post_sd["m.other.weight"])  # unmatched module untouched
