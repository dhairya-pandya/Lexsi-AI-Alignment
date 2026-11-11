import torch

from hsafe.tamper.anchor_subspace import orthonormal_basis, proj_coords, anchor_penalty


def test_orthonormal_basis_is_orthonormal():
    dirs = torch.randn(3, 20)  # 3 directions in d=20
    Q = orthonormal_basis(dirs)
    assert Q.shape == (20, 3)
    gram = Q.T @ Q
    assert torch.allclose(gram, torch.eye(3), atol=1e-5)  # columns orthonormal


def test_orthonormal_basis_spans_inputs():
    dirs = torch.randn(2, 12)
    Q = orthonormal_basis(dirs)
    # each input direction lies in the column span: reconstruct from projection, residual ~0
    for r in dirs:
        recon = Q @ (Q.T @ r)
        assert torch.allclose(recon, r, atol=1e-4)


def test_orthonormal_basis_drops_dependent_direction():
    r = torch.randn(10)
    dirs = torch.stack([r, 2.0 * r, torch.randn(10)])  # rank 2, not 3
    Q = orthonormal_basis(dirs)
    assert Q.shape[1] == 2  # degenerate direction dropped


def test_proj_coords_matches_manual():
    Q = orthonormal_basis(torch.randn(3, 16))
    h = torch.randn(5, 16)
    coords = proj_coords(h, Q)
    assert coords.shape == (5, 3)
    assert torch.allclose(coords, h @ Q, atol=1e-5)


def test_anchor_penalty_zero_when_above_target():
    Q = orthonormal_basis(torch.randn(2, 8))
    h = torch.randn(4, 8)
    coords = proj_coords(h, Q)
    low_target = coords.min(0).values - 1.0  # every row's coord already above target
    mask = torch.ones(4, dtype=torch.bool)
    assert anchor_penalty(h, Q, low_target, mask).item() == 0.0  # relu hinge inactive


def test_anchor_penalty_hinges_only_on_deficit():
    Q = torch.zeros(4, 1); Q[0, 0] = 1.0  # single direction = first coordinate
    h = torch.tensor([[1.0, 0, 0, 0], [5.0, 0, 0, 0]])  # coords 1.0 and 5.0
    target = torch.tensor([3.0])
    mask = torch.ones(2, dtype=torch.bool)
    # row0 deficit (3-1)=2 -> 4 ; row1 above target -> 0 ; mean over masked rows = 2.0
    assert torch.isclose(anchor_penalty(h, Q, target, mask), torch.tensor(2.0), atol=1e-5)


def test_anchor_penalty_respects_mask():
    Q = torch.zeros(3, 1); Q[0, 0] = 1.0
    h = torch.tensor([[0.0, 0, 0], [0.0, 0, 0]])  # both have coord 0
    target = torch.tensor([2.0])  # deficit 2 -> 4 each
    only_first = torch.tensor([True, False])
    assert torch.isclose(anchor_penalty(h, Q, target, only_first), torch.tensor(4.0), atol=1e-5)
    none = torch.zeros(2, dtype=torch.bool)
    assert anchor_penalty(h, Q, target, none).item() == 0.0  # no masked rows -> no penalty
