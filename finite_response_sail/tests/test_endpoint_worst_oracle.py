"""Endpoint worst-case oracle correctness.

Verifies:
  - p_wc[i,j] = p_high if ell_1 >= ell_0 else p_low
  - The alternative endpoint never gives a larger conditional loss.
  - Brute-force grid scan over p in [p_low, p_high] confirms p_wc is the
    pointwise argmax of p * ell_1 + (1-p) * ell_0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from finite_response_sail._true_to_worst_stress_eval import compute_endpoint_worst


def _setup(B=4, K=6, d=8, beta=0.1, rho_bar=0.05, seed=0,
           dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    psi = torch.randn(B, K, d, generator=g, dtype=dtype)
    ref = torch.randn(B, K, generator=g, dtype=dtype) * 0.5
    theta = torch.randn(d, generator=g, dtype=dtype) * 0.3
    p_obs = (0.1 + 0.8 * torch.rand(B, K, K, generator=g, dtype=dtype))
    return theta, psi, ref, p_obs, beta, rho_bar


def test_endpoint_choice_matches_formula():
    theta, psi, ref, p_obs, beta, rho_bar = _setup()
    L_wc, L_obs, R, p_wc, ell1, ell0, pi, h, clip = compute_endpoint_worst(
        theta, psi, ref, p_obs, rho_bar, beta,
    )
    p_low = (p_obs - rho_bar).clamp(min=0.0)
    p_high = (p_obs + rho_bar).clamp(max=1.0)
    expected = torch.where(ell1 >= ell0, p_high, p_low)
    assert torch.allclose(p_wc, expected)


def test_alternative_endpoint_never_larger():
    theta, psi, ref, p_obs, beta, rho_bar = _setup()
    _, _, _, p_wc, ell1, ell0, pi, h, _ = compute_endpoint_worst(
        theta, psi, ref, p_obs, rho_bar, beta,
    )
    p_low = (p_obs - rho_bar).clamp(min=0.0)
    p_high = (p_obs + rho_bar).clamp(max=1.0)

    chosen = p_wc * ell1 + (1.0 - p_wc) * ell0
    high = p_high * ell1 + (1.0 - p_high) * ell0
    low = p_low * ell1 + (1.0 - p_low) * ell0
    # chosen >= max(high, low) pointwise
    other_max = torch.maximum(high, low)
    assert (chosen + 1e-9 >= other_max).all()


def test_brute_force_grid_argmax():
    """For each (b,i,j) compare against a fine grid of p in [p_low, p_high]."""
    theta, psi, ref, p_obs, beta, rho_bar = _setup(B=2, K=4, d=6)
    _, _, _, p_wc, ell1, ell0, _, _, _ = compute_endpoint_worst(
        theta, psi, ref, p_obs, rho_bar, beta,
    )
    p_low = (p_obs - rho_bar).clamp(min=0.0)
    p_high = (p_obs + rho_bar).clamp(max=1.0)
    n_grid = 51
    # Build grid: shape [..., n_grid]
    t = torch.linspace(0.0, 1.0, n_grid, dtype=p_obs.dtype)
    grid = p_low.unsqueeze(-1) + t * (p_high - p_low).unsqueeze(-1)  # [B,K,K,n]
    integrand = grid * ell1.unsqueeze(-1) + (1.0 - grid) * ell0.unsqueeze(-1)
    best_idx = integrand.argmax(dim=-1)
    best_p = grid.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
    # endpoints should match the formula's pick within grid resolution
    chosen_loss = p_wc * ell1 + (1.0 - p_wc) * ell0
    best_loss = best_p * ell1 + (1.0 - best_p) * ell0
    assert (chosen_loss + 1e-9 >= best_loss).all()


def test_unclipped_decomposition_matches():
    """When the uncertainty set lies entirely inside [0,1],
    L_wc == L_obs + rho_bar * beta * R within float64 tol.
    """
    g = torch.Generator().manual_seed(7)
    B, K, d = 3, 5, 4
    psi = torch.randn(B, K, d, generator=g, dtype=torch.float64)
    ref = torch.randn(B, K, generator=g, dtype=torch.float64) * 0.5
    theta = torch.randn(d, generator=g, dtype=torch.float64) * 0.2
    rho_bar = 0.05
    # p_obs strictly in [0.2, 0.8] -> p_obs +- rho_bar ⊂ [0.15, 0.85]
    p_obs = 0.2 + 0.6 * torch.rand(B, K, K, generator=g, dtype=torch.float64)
    L_wc, L_obs, R, p_wc, _, _, _, _, clip = compute_endpoint_worst(
        theta, psi, ref, p_obs, rho_bar, beta=0.1,
    )
    assert clip["clipped_any_mass"] == 0.0
    expected = L_obs + rho_bar * 0.1 * R
    assert abs(L_wc - expected) < 1e-9, (L_wc, expected)
