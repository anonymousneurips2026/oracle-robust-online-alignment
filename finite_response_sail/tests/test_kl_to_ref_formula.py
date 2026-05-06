"""KL_to_ref formula closed-form check.

KL_to_ref = E_x sum_i pi_theta_i (log pi_theta_i - log pi_ref_i)
with pi_theta = softmax(ref + theta @ psi), pi_ref = softmax(ref).
"""
import torch
import torch.nn.functional as F

from finite_response_sail.policy import kl_to_ref


def test_zero_theta_gives_zero_kl():
    torch.manual_seed(0)
    psi = torch.randn(3, 5, 4)
    ref = torch.randn(3, 5)
    theta = torch.zeros(4)
    kl = kl_to_ref(theta, psi, ref).mean().item()
    assert abs(kl) < 1e-6


def test_matches_closed_form():
    torch.manual_seed(1)
    psi = torch.randn(2, 4, 3)
    ref = torch.randn(2, 4)
    theta = torch.randn(3)

    score = torch.einsum("bkd,d->bk", psi, theta)
    log_pi_th = F.log_softmax(ref + score, dim=-1)
    log_pi_ref = F.log_softmax(ref, dim=-1)
    expected = ((log_pi_th.exp()) * (log_pi_th - log_pi_ref)).sum(dim=-1)

    got = kl_to_ref(theta, psi, ref)
    assert torch.allclose(got, expected, atol=1e-7), \
        f"max abs diff = {(got - expected).abs().max().item():.3e}"


def test_kl_nonnegative():
    torch.manual_seed(2)
    psi = torch.randn(4, 6, 5)
    ref = torch.randn(4, 6)
    theta = torch.randn(5) * 3.0
    vals = kl_to_ref(theta, psi, ref)
    assert (vals >= -1e-7).all(), \
        f"KL went negative: min = {vals.min().item():.3e}"
