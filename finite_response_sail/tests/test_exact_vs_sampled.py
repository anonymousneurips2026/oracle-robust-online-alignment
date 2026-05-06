"""Monte-Carlo check: score-function-corrected sampled gradient matches the exact
analytic gradient of the all-pair expectation objective.
"""
import pytest

torch = pytest.importorskip("torch")

from finite_response_sail.objective import exact_expectation_loss
from finite_response_sail.sampled_objective import sampled_pair_loss


def _fixture(seed=0, B=4, K=4, d=3):
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(d, generator=g) * 0.3
    psi = torch.randn(B, K, d, generator=g) * 0.5
    ref_logits = torch.randn(B, K, generator=g) * 0.2
    p_center = torch.rand(B, K, K, generator=g) * 0.76 + 0.12   # [0.12, 0.88]
    return theta, psi, ref_logits, p_center


def _analytic_grad(theta, psi, ref_logits, p_center, rho_alg, beta):
    t = theta.detach().clone().requires_grad_(True)
    loss = exact_expectation_loss(t, psi, ref_logits, p_center, rho_alg, beta)
    (grad,) = torch.autograd.grad(loss, t)
    return grad


def _mc_score_function_grad(theta, psi, ref_logits, p_center, rho_alg, beta,
                            n_samples, baseline_value, seed=1234):
    """E over n_samples of the SF-corrected sampled_pair_loss gradient."""
    gen = torch.Generator().manual_seed(seed)
    grad = torch.zeros_like(theta)
    baseline = torch.tensor(baseline_value)
    for _ in range(n_samples):
        t = theta.detach().clone().requires_grad_(True)
        total, _ = sampled_pair_loss(
            t, psi, ref_logits, p_center, rho_alg, beta, baseline, generator=gen
        )
        (g,) = torch.autograd.grad(total, t)
        grad = grad + g
    return grad / n_samples


@pytest.mark.parametrize("rho_alg", [0.0, 0.1])
def test_exact_gradient_matches_mc_score_function(rho_alg):
    torch.manual_seed(0)
    theta, psi, ref_logits, p_center = _fixture(seed=0, B=2, K=4, d=3)
    beta = 0.5

    exact = _analytic_grad(theta, psi, ref_logits, p_center, rho_alg, beta)

    # Use the exact per-pair average as baseline so MC variance is small.
    with torch.no_grad():
        t0 = theta.clone()
        # Average of the raw per-pair loss under pi (no projection): approximates
        # E[sample_loss] and serves as a good control variate.
        loss_avg = exact_expectation_loss(
            t0, psi, ref_logits, p_center, rho_alg=0.0, beta=beta
        ).item()
    baseline_value = loss_avg

    mc = _mc_score_function_grad(
        theta, psi, ref_logits, p_center, rho_alg, beta,
        n_samples=20000, baseline_value=baseline_value, seed=7,
    )

    # Compare per-coordinate; allow a few percent Monte-Carlo slack.
    diff = (exact - mc).abs().max().item()
    scale = exact.abs().max().item() + 1e-6
    assert diff / scale < 0.08, (
        f"rho_alg={rho_alg}: max|exact-mc|={diff:.4e} scale={scale:.4e} "
        f"rel={diff / scale:.4f}"
    )
