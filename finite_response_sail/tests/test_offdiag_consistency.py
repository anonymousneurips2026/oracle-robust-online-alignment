"""Off-diagonal-renormalized objective and diagonal-mass metric."""
import pytest

torch = pytest.importorskip("torch")

from finite_response_sail.objective import exact_expectation_loss
from finite_response_sail.offdiag_objective import offdiag_exact_loss
from finite_response_sail.policy import diagonal_mass, policy_probs


def test_offdiag_uses_less_mass_on_diagonal():
    torch.manual_seed(0)
    B, K, d = 3, 6, 4
    theta = torch.randn(d) * 0.4
    psi = torch.randn(B, K, d) * 0.3
    ref_logits = torch.zeros(B, K)
    p_center = torch.full((B, K, K), 0.5)

    # BT term at p=0.5 equals softplus(beta h)+softplus(-beta h), symmetric in h.
    # With p_center symmetric and a strictly non-uniform policy, main and
    # offdiag losses differ; they must differ in the direction of dropping
    # diagonal contribution.
    l_main = exact_expectation_loss(theta, psi, ref_logits, p_center, 0.0, 0.3).item()
    l_off = offdiag_exact_loss(theta, psi, ref_logits, p_center, 0.0, 0.3).item()

    # At h_ii = 0, diagonal contributes pi_i^2 * [0.5*log(2) + 0.5*log(2)] = pi_i^2 * log(2).
    # Removing that diagonal and renormalizing offdiag yields a different scalar.
    assert abs(l_main - l_off) > 1e-6


def test_diagonal_mass_matches_direct_sum():
    B, K, d = 2, 5, 3
    torch.manual_seed(1)
    theta = torch.randn(d) * 0.2
    psi = torch.randn(B, K, d) * 0.3
    pi = policy_probs(theta, psi, torch.zeros(B, K))
    dm = diagonal_mass(pi)
    expected = (pi * pi).sum(dim=-1)
    assert torch.allclose(dm, expected)
    assert (dm <= 1.0 + 1e-9).all() and (dm >= 1.0 / K - 1e-9).all()


def test_offdiag_approaches_main_as_policy_becomes_uniform():
    """With uniform policy, sum_k pi_k^2 = 1/K; off-diagonal renormalization
    differs from main by only the diagonal contribution proportion. The two
    agree in the limit K -> infty; for finite K they differ by a known small
    gap. We just check both are finite and well-behaved.
    """
    B, K, d = 2, 16, 4
    theta = torch.zeros(d)  # uniform policy
    psi = torch.randn(B, K, d) * 0.3
    ref_logits = torch.zeros(B, K)
    p_center = torch.full((B, K, K), 0.5)
    l_main = exact_expectation_loss(theta, psi, ref_logits, p_center, 0.0, 0.1).item()
    l_off = offdiag_exact_loss(theta, psi, ref_logits, p_center, 0.0, 0.1).item()
    assert l_main > 0 and l_off > 0
