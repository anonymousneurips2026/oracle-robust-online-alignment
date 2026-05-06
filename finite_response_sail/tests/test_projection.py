"""L2 projection Pi_{||theta|| <= R}."""
import pytest

torch = pytest.importorskip("torch")

from finite_response_sail.policy import project_l2, project_l2_


def test_identity_inside_ball():
    torch.manual_seed(0)
    theta = torch.randn(32) * 0.1
    assert theta.norm().item() < 10.0
    out = project_l2(theta, radius=10.0)
    assert torch.allclose(out, theta)


def test_shrinks_to_boundary_outside_ball():
    theta = torch.ones(16) * 5.0       # ||theta|| = 20.0
    R = 3.0
    out = project_l2(theta, radius=R)
    assert pytest.approx(out.norm().item(), abs=1e-6) == R
    expected = theta * (R / theta.norm())
    assert torch.allclose(out, expected, atol=1e-6)


def test_inplace_projection_updates_parameter():
    p = torch.nn.Parameter(torch.ones(8) * 2.0)   # ||p|| = sqrt(32) ~ 5.66
    R = 1.0
    project_l2_(p, radius=R)
    assert pytest.approx(p.data.norm().item(), abs=1e-6) == R


def test_inplace_projection_noop_inside_ball():
    p = torch.nn.Parameter(torch.zeros(4))
    project_l2_(p, radius=1.0)
    assert p.data.norm().item() == 0.0
