"""check_margin must refuse rho_mis that leaves p_obs outside [0, 1].

Concrete case: tau=10, rho_mis=0.1, r ~ U[0, 1]. With tau=10 and r in [0, 1],
the minimum p_star can approach sigmoid(-10) ~ 5e-5, so rho_mis = 0.1 is
well beyond the empirical margin and must raise ValueError.
"""
import numpy as np
import pytest

from finite_response_sail.oracle_misspec import (
    check_margin,
    compute_p_star,
)


def test_check_margin_raises_under_tau_10():
    rng = np.random.default_rng(0)
    r = rng.uniform(0.0, 1.0, size=(32, 12))
    p_star = compute_p_star(r, tau=10.0)
    with pytest.raises(ValueError) as excinfo:
        check_margin(p_star, rho_mis=0.1)
    msg = str(excinfo.value)
    assert "p_star margin too small for rho_mis" in msg


def test_check_margin_passes_under_tau_2():
    rng = np.random.default_rng(1)
    r = rng.uniform(0.0, 1.0, size=(32, 12))
    p_star = compute_p_star(r, tau=2.0)
    # Should not raise for rho_mis in the planned v3 grid.
    check_margin(p_star, rho_mis=0.1)
    check_margin(p_star, rho_mis=0.05)
    check_margin(p_star, rho_mis=0.025)
