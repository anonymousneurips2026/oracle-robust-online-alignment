"""Oracle-bound properties for the v3_pairbias oracle.

No probability clipping is used. With tau=2 and bounded rewards in [0, 1]:
    delta_empirical(p_star) >= sigmoid(-2.0) - 1e-6

Other guarantees:
    |p_obs - p_star| <= rho_mis
    p_obs in [0, 1]
    p_obs_ji = 1 - p_obs_ij
    p_obs_ii = 0.5
    b_ii = 0
    b_ji = -b_ij
"""
import math

import numpy as np
import pytest

from finite_response_sail.oracle_misspec import (
    compute_p_obs,
    compute_p_star,
    compute_pairwise_bias,
    empirical_margin_delta,
)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _random_batch(rng, B=8, K=12):
    true_r = rng.uniform(0.0, 1.0, size=(B, K))
    bias = rng.uniform(0.0, 1.0, size=(B, K))
    return true_r, bias


@pytest.mark.parametrize("rho_mis", [0.0, 0.025, 0.05, 0.075, 0.1])
def test_p_obs_bounds_and_symmetry(rho_mis):
    rng = np.random.default_rng(int(round((rho_mis + 1.0) * 1000)))
    true_r, bias = _random_batch(rng)
    p_star = compute_p_star(true_r, tau=2.0)
    b = compute_pairwise_bias(true_r, bias)
    p_obs = compute_p_obs(p_star, b, rho_mis=rho_mis)

    # p_obs inside [0, 1].
    assert p_obs.min() >= 0.0 - 1e-12, f"min={p_obs.min()}"
    assert p_obs.max() <= 1.0 + 1e-12, f"max={p_obs.max()}"

    # |p_obs - p_star| <= rho_mis exactly.
    assert np.abs(p_obs - p_star).max() <= rho_mis + 1e-12

    # Anti-symmetry: p_obs_ji = 1 - p_obs_ij.
    assert np.allclose(p_obs + np.swapaxes(p_obs, -1, -2), 1.0, atol=1e-10)

    # Diagonal: p_obs_ii = 0.5.
    diag = np.diagonal(p_obs, axis1=-2, axis2=-1)
    assert np.allclose(diag, 0.5, atol=1e-12)

    # Antisymmetric b: b_ii = 0, b_ji = -b_ij.
    b_diag = np.diagonal(b, axis1=-2, axis2=-1)
    assert np.allclose(b_diag, 0.0, atol=1e-12)
    assert np.allclose(b + np.swapaxes(b, -1, -2), 0.0, atol=1e-10)


def test_margin_floor_under_tau_2():
    """With tau=2 and rewards in [0, 1], the rewards gap is bounded by 1, so
    p_star is bounded by sigmoid(+/- 2). delta_empirical >= sigmoid(-2) - eps.
    """
    floor = _sigmoid(-2.0) - 1e-6
    rng = np.random.default_rng(0)
    for trial in range(20):
        true_r = rng.uniform(0.0, 1.0, size=(16, 12))
        p_star = compute_p_star(true_r, tau=2.0)
        delta = empirical_margin_delta(p_star)
        assert delta >= floor, f"trial={trial} delta={delta}"
