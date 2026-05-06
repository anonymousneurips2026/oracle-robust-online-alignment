"""P_alpha invariants: identity at endpoints, antisymmetry, range on main grid.

P_alpha = p_star + alpha * rho_mis * b
       = compute_p_obs(p_star, b, alpha * rho_mis)
"""
import numpy as np

from finite_response_sail.oracle_misspec import (
    compute_p_obs, compute_p_star, compute_pairwise_bias,
)


MAIN_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RHO_MIS = 0.05
TAU = 2.0


def _build():
    rng = np.random.default_rng(0)
    r = rng.uniform(0.0, 1.0, size=(4, 8))
    s = rng.uniform(0.0, 1.0, size=(4, 8))
    p_star = compute_p_star(r, tau=TAU)
    b = compute_pairwise_bias(r, s)
    p_obs = compute_p_obs(p_star, b, RHO_MIS)
    return p_star, b, p_obs


def test_alpha0_recovers_p_star():
    p_star, b, _ = _build()
    P0 = compute_p_obs(p_star, b, 0.0 * RHO_MIS)
    assert np.allclose(P0, p_star, atol=1e-12)


def test_alpha1_recovers_p_obs():
    p_star, b, p_obs = _build()
    P1 = compute_p_obs(p_star, b, 1.0 * RHO_MIS)
    assert np.allclose(P1, p_obs, atol=1e-12)


def test_antisymmetry_main_grid():
    p_star, b, _ = _build()
    for alpha in MAIN_GRID:
        Pa = compute_p_obs(p_star, b, alpha * RHO_MIS)
        sym = np.swapaxes(Pa, -1, -2) + Pa - 1.0
        assert np.allclose(sym, 0.0, atol=1e-12), \
            f"antisymmetry violated at alpha={alpha}: max|sym|={np.abs(sym).max():.3e}"
        diag = np.diagonal(Pa, axis1=-2, axis2=-1)
        assert np.allclose(diag, 0.5, atol=1e-12), \
            f"diagonal != 0.5 at alpha={alpha}"


def test_range_main_grid_within_unit():
    p_star, b, _ = _build()
    for alpha in MAIN_GRID:
        Pa = compute_p_obs(p_star, b, alpha * RHO_MIS)
        assert Pa.min() >= 0.0 - 1e-9, f"P_alpha < 0 at alpha={alpha}"
        assert Pa.max() <= 1.0 + 1e-9, f"P_alpha > 1 at alpha={alpha}"
