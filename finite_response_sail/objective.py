"""Exact all-pair expectation objective for finite-response online SAIL.

L(theta) = E_x sum_{i, j} pi_i pi_j [p_center_ij softplus(-beta h_ij)
                                    + (1 - p_center_ij) softplus(beta h_ij)]
         + rho_alg * beta * E_x sum_{i, j} pi_i pi_j |h_ij|

The expectation over (i, j) ~ pi_theta is exact, so autograd through theta picks
up both the h-term and the score-function contribution through the policy.
"""
import torch
import torch.nn.functional as F


def _score_and_pi(theta, psi, ref_logits):
    score = torch.einsum("bkd,d->bk", psi, theta)
    logits = ref_logits + score
    pi = F.softmax(logits, dim=-1)
    return score, pi


def exact_expectation_loss(theta, psi, ref_logits, p_center, rho_alg, beta):
    """theta [d], psi [B, K, d], ref_logits [B, K], p_center [B, K, K]."""
    score, pi = _score_and_pi(theta, psi, ref_logits)
    h = score.unsqueeze(-1) - score.unsqueeze(-2)       # [B, K, K]
    bh = beta * h
    ell1 = F.softplus(-bh)
    ell0 = F.softplus(bh)
    bt_ij = p_center * ell1 + (1.0 - p_center) * ell0   # [B, K, K]
    w = pi.unsqueeze(-1) * pi.unsqueeze(-2)             # [B, K, K]
    bt_term = (w * bt_ij).sum(dim=(-1, -2))             # [B]
    penalty = rho_alg * beta * (w * h.abs()).sum(dim=(-1, -2))
    return (bt_term + penalty).mean()


@torch.no_grad()
def sensitivity_R(theta, psi, ref_logits):
    """E_{x, i, j ~ pi}[|h_ij|]."""
    score, pi = _score_and_pi(theta, psi, ref_logits)
    h = score.unsqueeze(-1) - score.unsqueeze(-2)
    w = pi.unsqueeze(-1) * pi.unsqueeze(-2)
    return (w * h.abs()).sum(dim=(-1, -2)).mean().item()


@torch.no_grad()
def worst_case_loss(theta, psi, ref_logits, p_star, rho_mis, beta):
    """E_{x, i, j ~ pi} max_{p in [max(0, p_star - rho_mis),
                                  min(1, p_star + rho_mis)]}
                       p*softplus(-beta h) + (1-p)*softplus(beta h).

    The integrand is linear in p, so the max is at an endpoint. No calibration
    clipping: the feasible interval is bounded only by the [0, 1] probability
    simplex.
    """
    score, pi = _score_and_pi(theta, psi, ref_logits)
    h = score.unsqueeze(-1) - score.unsqueeze(-2)
    bh = beta * h
    ell1 = F.softplus(-bh)
    ell0 = F.softplus(bh)
    lo = torch.clamp(p_star - rho_mis, min=0.0)
    hi = torch.clamp(p_star + rho_mis, max=1.0)
    worst_p = torch.where(ell1 > ell0, hi, lo)
    wcl_ij = worst_p * ell1 + (1.0 - worst_p) * ell0
    w = pi.unsqueeze(-1) * pi.unsqueeze(-2)
    return (w * wcl_ij).sum(dim=(-1, -2)).mean().item()


def _expected_scalar(probs, values):
    """E_{i ~ pi}[values_i]. probs: [B, K], values: [B, K]."""
    return (probs * values).sum(dim=-1).mean().item()


@torch.no_grad()
def expected_true_utility(theta, psi, ref_logits, true_rewards):
    _, pi = _score_and_pi(theta, psi, ref_logits)
    return _expected_scalar(pi, true_rewards)


@torch.no_grad()
def expected_bias_attr(theta, psi, ref_logits, bias_attr):
    _, pi = _score_and_pi(theta, psi, ref_logits)
    return _expected_scalar(pi, bias_attr)


@torch.no_grad()
def pair_bias_mass_under_pi(theta, psi, ref_logits, b):
    """E_{x, i, j ~ pi}[|b_ij|]. b: [B, K, K]."""
    _, pi = _score_and_pi(theta, psi, ref_logits)
    w = pi.unsqueeze(-1) * pi.unsqueeze(-2)
    return (w * b.abs()).sum(dim=(-1, -2)).mean().item()


@torch.no_grad()
def signed_pair_bias_under_pi(theta, psi, ref_logits, b):
    """E_{x, i, j ~ pi}[b_ij]. Near zero under antisymmetry + iid i,j."""
    _, pi = _score_and_pi(theta, psi, ref_logits)
    w = pi.unsqueeze(-1) * pi.unsqueeze(-2)
    return (w * b).sum(dim=(-1, -2)).mean().item()


@torch.no_grad()
def expected_true_win_vs_ref(theta, psi, ref_logits, pi_ref, true_rewards):
    """E_{x, i ~ pi_theta, j ~ pi_ref}[win(true_r_i, true_r_j)] with ties = 0.5."""
    _, pi = _score_and_pi(theta, psi, ref_logits)
    r = true_rewards                                   # [B, K]
    gt = (r.unsqueeze(-1) > r.unsqueeze(-2)).float()   # [B, K, K]: r_i > r_j
    eq = (r.unsqueeze(-1) == r.unsqueeze(-2)).float()
    w = pi.unsqueeze(-1) * pi_ref.unsqueeze(-2)        # pi_theta(i) * pi_ref(j)
    return (w * (gt + 0.5 * eq)).sum(dim=(-1, -2)).mean().item()


@torch.no_grad()
def observed_win_vs_ref(theta, psi, ref_logits, pi_ref, p_obs):
    """E_{x, i ~ pi_theta, j ~ pi_ref}[p_obs_ij] -- the biased oracle's
    view of the win rate of pi_theta over pi_ref.
    """
    _, pi = _score_and_pi(theta, psi, ref_logits)
    w = pi.unsqueeze(-1) * pi_ref.unsqueeze(-2)        # [B, K, K]
    return (w * p_obs).sum(dim=(-1, -2)).mean().item()


@torch.no_grad()
def label_entropy(theta, psi, ref_logits, p_obs, eps=1e-12):
    """E_{x, i, j ~ pi_theta}[H(Bernoulli(p_obs_ij))].

    Interprets p_obs_ij as the training-label Bernoulli parameter for pair (i, j).
    Under the spec's stochastic update, this is the entropy of the label drawn each step.
    """
    _, pi = _score_and_pi(theta, psi, ref_logits)
    w = pi.unsqueeze(-1) * pi.unsqueeze(-2)            # [B, K, K]
    p = p_obs.clamp(min=eps, max=1.0 - eps)
    bern_H = -(p * torch.log(p) + (1.0 - p) * torch.log1p(-p))
    return (w * bern_H).sum(dim=(-1, -2)).mean().item()


@torch.no_grad()
def mean_abs_h_sampled(theta, psi, ref_logits, n_pairs=1, generator=None):
    """Monte-Carlo estimate of E_{i,j~pi}[|h_ij|] via actual sampling of (i, j).

    Included alongside `sensitivity_R` (exact expectation) so that metrics.json
    reports both the deterministic and the variance-full stochastic estimator.
    In expectation the two agree; at finite n_pairs the MC estimate has variance.
    """
    score, pi = _score_and_pi(theta, psi, ref_logits)
    B, K = pi.shape
    acc = 0.0
    for _ in range(n_pairs):
        i = torch.multinomial(pi, 1, replacement=True, generator=generator).squeeze(-1)
        j = torch.multinomial(pi, 1, replacement=True, generator=generator).squeeze(-1)
        bidx = torch.arange(B, device=pi.device)
        h_ij = score[bidx, i] - score[bidx, j]
        acc += h_ij.abs().mean().item()
    return acc / n_pairs
