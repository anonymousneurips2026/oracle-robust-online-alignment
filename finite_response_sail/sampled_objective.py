"""Sampled-pair ablation with score-function correction.

sample_loss(i, j) = p_center_ij * softplus(-beta h_ij)
                  + (1 - p_center_ij) * softplus(beta h_ij)
                  + rho_alg * beta * |h_ij|
total_loss        = sample_loss
                  + (sample_loss.detach() - baseline) * (log pi_i + log pi_j)

Baseline is a running EMA maintained by the caller.
"""
import torch
import torch.nn.functional as F


def sampled_pair_loss(
    theta, psi, ref_logits, p_center, rho_alg, beta, baseline, generator=None
):
    """Returns (total_loss, sample_loss_mean_detached).

    baseline: scalar tensor on same device as psi.
    generator: optional torch.Generator for reproducible sampling.
    """
    score = torch.einsum("bkd,d->bk", psi, theta)
    logits = ref_logits + score
    logp = F.log_softmax(logits, dim=-1)
    pi = logp.exp()
    B = pi.shape[0]
    i = torch.multinomial(pi, 1, replacement=True, generator=generator).squeeze(-1)
    j = torch.multinomial(pi, 1, replacement=True, generator=generator).squeeze(-1)
    bidx = torch.arange(B, device=pi.device)
    h = score[bidx, i] - score[bidx, j]
    bh = beta * h
    p_ij = p_center[bidx, i, j]
    ell1 = F.softplus(-bh)
    ell0 = F.softplus(bh)
    sample_loss = p_ij * ell1 + (1.0 - p_ij) * ell0 + rho_alg * beta * h.abs()
    adv = sample_loss.detach() - baseline
    total = sample_loss + adv * (logp[bidx, i] + logp[bidx, j])
    return total.mean(), sample_loss.detach().mean()
