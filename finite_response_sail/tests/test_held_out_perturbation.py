"""Unit tests for finite_response_sail.held_out_perturbation."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from finite_response_sail.held_out_perturbation import (
    BIAS_ACTIVE_RATE_FLOOR,
    RNG_NAMESPACE,
    build_axis_aligned_perturbations,
    build_held_out_perturbation,
    evaluate_cell,
    generate_unit_direction,
    held_out_seed,
)
from finite_response_sail.oracle_misspec import compute_p_obs, compute_p_star
from finite_response_sail.offdiag_objective import offdiag_exact_loss


def test_held_out_seed_deterministic():
    """Same (namespace, s, m) -> same int across calls."""
    a = held_out_seed(RNG_NAMESPACE, 3, 7)
    b = held_out_seed(RNG_NAMESPACE, 3, 7)
    assert a == b
    # Different inputs should produce different seeds.
    assert held_out_seed(RNG_NAMESPACE, 3, 8) != a
    assert held_out_seed(RNG_NAMESPACE, 4, 7) != a
    assert held_out_seed("other_namespace", 3, 7) != a


def test_generate_unit_direction_norm_and_repro():
    """||u|| == 1 and reproducible across repeated calls."""
    u1 = generate_unit_direction(s=0, m=1, d=7)
    u2 = generate_unit_direction(s=0, m=1, d=7)
    np.testing.assert_array_equal(u1, u2)
    assert math.isclose(float(np.linalg.norm(u1)), 1.0, abs_tol=1e-12)
    # Different m gives a different direction.
    u3 = generate_unit_direction(s=0, m=2, d=7)
    assert not np.array_equal(u1, u3)


def _toy_inputs(N=4, K=4, d=3, seed=0):
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal((N, K, d)).astype(np.float64)
    # Keep psi roughly bounded so that s_held has reasonable magnitudes.
    psi /= max(1.0, np.abs(psi).max())
    true_reward = rng.standard_normal((N, K)).astype(np.float64)
    p_star = compute_p_star(true_reward, tau=2.0)
    return psi, true_reward, p_star


def test_build_perturbation_invariants():
    """Synthetic small case: b antisymmetric, |b| <= 1, b_ii = 0."""
    psi, true_reward, p_star = _toy_inputs(N=4, K=4, d=3, seed=1)
    pert, retry_log = build_held_out_perturbation(
        psi_np=psi, true_reward_np=true_reward, p_star_np=p_star,
        rho_mis=0.05, s=0, m=1, d=3,
    )
    b = pert.b
    # Antisymmetric across the last two axes
    np.testing.assert_allclose(b + b.transpose(0, 2, 1), 0.0, atol=1e-12)
    # Magnitude bounded by 1
    assert float(np.abs(b).max()) <= 1.0 + 1e-10
    # Diagonal is exactly zero
    diag_max = float(np.abs(np.diagonal(b, axis1=-2, axis2=-1)).max())
    assert diag_max == 0.0
    # ||u|| = 1
    assert math.isclose(float(np.linalg.norm(pert.u)), 1.0, abs_tol=1e-12)
    # bias_active_rate is in [0, 1]
    assert 0.0 <= pert.bias_active_rate <= 1.0


def test_axis_aligned_count():
    """d=7 should produce exactly 7 axis-aligned directions per seed."""
    psi, true_reward, p_star = _toy_inputs(N=3, K=4, d=7, seed=2)
    axes = build_axis_aligned_perturbations(
        psi_np=psi, true_reward_np=true_reward, p_star_np=p_star,
        rho_mis=0.05, s=0, d=7,
    )
    assert len(axes) == 7
    # m indices are -1..-7
    assert sorted(p.m for p in axes) == [-7, -6, -5, -4, -3, -2, -1]
    # Each u is a one-hot vector
    for j, p in enumerate(axes):
        expected = np.zeros(7)
        expected[j] = 1.0
        np.testing.assert_array_equal(p.u, expected)
        # Antisymmetry
        np.testing.assert_allclose(
            p.b + p.b.transpose(0, 2, 1), 0.0, atol=1e-12
        )


def test_alpha_affineness_on_toy():
    """L_offdiag is affine in p_center, so L(0.5*p_obs + 0.5*p_star)
    must equal 0.5*(L(p_star) + L(p_obs))."""
    psi_np, true_reward, p_star = _toy_inputs(N=3, K=4, d=3, seed=3)
    pert, _ = build_held_out_perturbation(
        psi_np=psi_np, true_reward_np=true_reward, p_star_np=p_star,
        rho_mis=0.05, s=0, m=1, d=3,
    )
    device = torch.device("cpu")
    psi_t = torch.from_numpy(psi_np).float()
    ref_logits = torch.zeros(psi_np.shape[0], psi_np.shape[1])
    theta = torch.randn(psi_np.shape[-1])
    beta = 0.1
    p_star_t = torch.from_numpy(p_star).float()
    p_obs_t = torch.from_numpy(pert.p_obs).float()
    p_half_t = 0.5 * (p_star_t + p_obs_t)
    with torch.no_grad():
        L0 = float(offdiag_exact_loss(theta, psi_t, ref_logits,
                                       p_star_t, 0.0, beta).item())
        L1 = float(offdiag_exact_loss(theta, psi_t, ref_logits,
                                       p_obs_t, 0.0, beta).item())
        Lh = float(offdiag_exact_loss(theta, psi_t, ref_logits,
                                       p_half_t, 0.0, beta).item())
    predicted = 0.5 * (L0 + L1)
    assert abs(Lh - predicted) < 1e-6, (Lh, predicted)


def test_retry_logic_raises_when_floor_unattainable():
    """If we set the floor higher than achievable, the retry loop should raise."""
    # With identical reward & bias signals the activation rate can be made low.
    psi, true_reward, p_star = _toy_inputs(N=2, K=3, d=2, seed=4)
    # Monkey-patch the constant by importing the module, swapping, then restoring.
    from finite_response_sail import held_out_perturbation as hop
    old = hop.BIAS_ACTIVE_RATE_FLOOR
    try:
        hop.BIAS_ACTIVE_RATE_FLOOR = 1.5  # impossible
        with pytest.raises(RuntimeError, match="failed bias_active_rate"):
            hop.build_held_out_perturbation(
                psi_np=psi, true_reward_np=true_reward, p_star_np=p_star,
                rho_mis=0.05, s=0, m=1, d=2,
            )
    finally:
        hop.BIAS_ACTIVE_RATE_FLOOR = old
