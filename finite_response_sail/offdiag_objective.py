"""Off-diagonal-renormalized ablation: drop i==j from the double sum and
renormalize pi_i pi_j by (1 - sum_k pi_k^2).

Checks that main-objective results aren't driven by duplicate-comparison mass.
"""
from typing import Dict

import torch
import torch.nn.functional as F

from .objective import _score_and_pi


def offdiag_exact_loss(theta, psi, ref_logits, p_center, rho_alg, beta):
    score, pi = _score_and_pi(theta, psi, ref_logits)
    h = score.unsqueeze(-1) - score.unsqueeze(-2)
    bh = beta * h
    ell1 = F.softplus(-bh)
    ell0 = F.softplus(bh)
    bt_ij = p_center * ell1 + (1.0 - p_center) * ell0
    w = pi.unsqueeze(-1) * pi.unsqueeze(-2)
    K = w.shape[-1]
    eye = torch.eye(K, device=w.device, dtype=w.dtype).expand_as(w)
    off = w * (1.0 - eye)
    norm = off.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    wn = off / norm
    bt_term = (wn * bt_ij).sum(dim=(-1, -2))
    penalty = rho_alg * beta * (wn * h.abs()).sum(dim=(-1, -2))
    return (bt_term + penalty).mean()


@torch.no_grad()
def offdiag_denominator_stats(pi: torch.Tensor) -> Dict[str, float]:
    """pi: [B, K]. Per-prompt offdiag denominator = 1 - sum_k pi_k^2.

    Returns {offdiag_denominator_min, offdiag_denominator_mean}. Emits a
    single [warn] line to stderr if the min drops below 1e-6.
    """
    import sys
    denom = 1.0 - (pi * pi).sum(dim=-1)
    dmin = float(denom.min().item())
    dmean = float(denom.mean().item())
    if dmin < 1e-6:
        print(
            f"[warn] offdiag denominator < 1e-6 (min={dmin:.3e}, mean={dmean:.3e})",
            file=sys.stderr,
        )
    return {
        "offdiag_denominator_min": dmin,
        "offdiag_denominator_mean": dmean,
    }
