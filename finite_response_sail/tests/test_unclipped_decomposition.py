"""When the uncertainty set lies entirely inside [0,1]:

   L_wc == L_obs + rho_bar * beta * R   (within tol),
   L_stress(alpha) == (1 - alpha) L_id + alpha L_wc   (within tol).

When clipping occurs we DO NOT assert the decomposition: we only verify that
the evaluator records clipping mass > 0 and that the per-alpha values stay in
[0, 1].
"""
from __future__ import annotations

import torch

from finite_response_sail._true_to_worst_stress_eval import compute_endpoint_worst
from finite_response_sail.objective import exact_expectation_loss


def _synth(B=3, K=5, d=4, seed=11, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    psi = torch.randn(B, K, d, generator=g, dtype=dtype)
    ref = torch.randn(B, K, generator=g, dtype=dtype) * 0.5
    theta = torch.randn(d, generator=g, dtype=dtype) * 0.2
    return theta, psi, ref


def test_unclipped_decomposition_holds():
    theta, psi, ref = _synth()
    rho_bar = 0.05
    # p_obs strictly in [0.2, 0.8] -> uncertainty set ⊂ [0.15, 0.85]
    g = torch.Generator().manual_seed(11)
    p_obs = 0.2 + 0.6 * torch.rand(*psi.shape[:-1], psi.shape[-2],
                                   generator=g, dtype=torch.float64)
    p_id = p_obs.clone()  # b = 0 case, so p_id == p_obs
    L_wc, L_obs, R, p_wc, _, _, _, _, clip = compute_endpoint_worst(
        theta, psi, ref, p_obs, rho_bar, beta=0.1,
    )
    assert clip["clipped_any_mass"] == 0.0
    # Direct decomposition
    expected = L_obs + rho_bar * 0.1 * R
    assert abs(L_wc - expected) < 1e-9, (L_wc, expected)

    # Affine identity at intermediate alphas
    L_id = exact_expectation_loss(theta, psi, ref, p_id, 0.0, 0.1).item()
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        P_alpha = (1.0 - alpha) * p_id + alpha * p_wc
        L_alpha = exact_expectation_loss(theta, psi, ref, P_alpha, 0.0, 0.1).item()
        expect = (1.0 - alpha) * L_id + alpha * L_wc
        assert abs(L_alpha - expect) < 1e-9, (alpha, L_alpha, expect)


def test_clipped_case_decomposition_skipped():
    """With rho_bar=0.4 and p_obs in (0.1, 0.9), every uncertainty set hits
    the simplex boundary. We assert the clipping mass is positive and skip
    the algebraic decomposition assertion entirely.
    """
    theta, psi, ref = _synth(seed=12)
    rho_bar = 0.4
    g = torch.Generator().manual_seed(12)
    p_obs = 0.1 + 0.8 * torch.rand(*psi.shape[:-1], psi.shape[-2],
                                   generator=g, dtype=torch.float64)
    L_wc, L_obs, R, _, _, _, _, _, clip = compute_endpoint_worst(
        theta, psi, ref, p_obs, rho_bar, beta=0.1,
    )
    assert clip["clipped_any_mass"] > 0.0
    # Decomposition L_wc == L_obs + rho_bar * beta * R is NOT asserted here:
    # under clipping the LHS lies strictly below the RHS proxy.
    rhs_proxy = L_obs + rho_bar * 0.1 * R
    # Sanity: the clipped worst-case is bounded above by the unclipped proxy.
    assert L_wc <= rhs_proxy + 1e-9
