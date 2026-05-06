"""When no eta_kl matches Robust within tolerance, selection must MATCH_FAIL."""
import math

from finite_response_sail._kl_smoothing_report import GroupAgg, select_match


def _g(eta, val_R):
    return GroupAgg(
        label=f"KL-η={eta}", eta_kl=eta, rho_alg_ratio=0.0,
        n_seeds=10, seeds=list(range(10)),
        val_R=(val_R, 0.0), val_entropy=(2.0, 0.0),
        val_KL_to_ref=(0.0, 0.0),
        R=(val_R, 0.0), E_r_star=(0.2, 0.0),
        entropy=(2.0, 0.0), theta_norm=(4.0, 0.0),
        KL_to_ref=(0.0, 0.0), L_SAIL_Pstar=(0.69, 0.0),
        wcl_rho_mis=(0.7, 0.0), avg_shift_loss=(0.69, 0.0),
        worst_shift_loss=(0.7, 0.0), shift_gap=(0.0, 0.0),
        val_source="val",
    )


def test_match_failed_when_grid_too_far():
    # Robust target R = 0.47; Nominal R = 0.49 -> tol = 0.10*0.02 = 0.002
    # All KL etas produce R far from 0.47.
    kl_groups = [_g(1e-4, 0.49), _g(1e-3, 0.485), _g(1e-2, 0.48)]
    res = select_match(kl_groups, target_value=0.47, target_metric="R",
                       nominal_value=0.49)
    assert res.status == "MATCH_FAILED", \
        f"expected MATCH_FAILED, got {res.status}"
    assert res.recommendation, "recommendation should not be empty when failed"


def test_match_ok_when_grid_brackets():
    kl_groups = [_g(1e-4, 0.49), _g(1e-3, 0.4715), _g(1e-2, 0.45)]
    # Robust=0.47, Nominal=0.49, tol = 0.10*0.02 = 0.002. eta=1e-3 is within.
    res = select_match(kl_groups, target_value=0.47, target_metric="R",
                       nominal_value=0.49)
    assert res.status == "OK", f"expected OK, got {res.status}"
    assert abs(res.chosen_eta - 1e-3) < 1e-12
