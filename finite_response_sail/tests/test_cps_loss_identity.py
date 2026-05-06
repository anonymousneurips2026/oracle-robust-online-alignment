"""CPS L_alpha identity checks against existing v3_pairbias_confirm runs.

Skipped if the cache and confirm results aren't available (CI without data).
"""
import json
import os
import math
from pathlib import Path

import pytest


CONFIRM_DIR = Path("finite_response_sail/results/v3_pairbias_confirm")
ALPHA_TOL = 1e-5
COLLINEAR_TOL = 1e-5


def _first_complete_run():
    if not CONFIRM_DIR.exists():
        return None
    for mp in sorted(CONFIRM_DIR.glob("mode-*/metrics.json")):
        if (mp.parent / "theta.pt").exists() and (mp.parent / "oracle_tensors_test.npz").exists():
            return mp.parent
    return None


@pytest.mark.skipif(_first_complete_run() is None,
                    reason="no complete v3_pairbias_confirm run available")
def test_cps_alpha_identity():
    import torch
    from finite_response_sail import _cps_shift_eval as cps

    run_dir = _first_complete_run()
    assert run_dir is not None
    d = json.loads((run_dir / "metrics.json").read_text())
    cfg = d["config"]
    rec = cps.RunRecord(
        run_dir=run_dir, tag=cfg.get("tag", "test"),
        config=cfg, metrics_test=d["final"]["test"],
        metrics_val=d["final"].get("val", {}),
    )
    data_cache = {}
    res = cps.evaluate_run(rec, [0.0, 1.0], data_cache)
    nominal = float(d["final"]["test"]["nominal_bt_loss"])
    biased = float(d["final"]["test"]["biased_bt_loss"])
    assert abs(res.L_alpha_exact[0] - nominal) <= ALPHA_TOL, \
        f"alpha=0 identity: |L - nominal_bt_loss| = {abs(res.L_alpha_exact[0] - nominal):.3e}"
    assert abs(res.L_alpha_exact[1] - biased) <= ALPHA_TOL, \
        f"alpha=1 identity: |L - biased_bt_loss| = {abs(res.L_alpha_exact[1] - biased):.3e}"


@pytest.mark.skipif(_first_complete_run() is None,
                    reason="no complete v3_pairbias_confirm run available")
def test_cps_alpha_affine_in_alpha():
    """L_alpha is a linear function of alpha for fixed theta — three points colinear."""
    import torch
    from finite_response_sail import _cps_shift_eval as cps

    run_dir = _first_complete_run()
    d = json.loads((run_dir / "metrics.json").read_text())
    cfg = d["config"]
    rec = cps.RunRecord(
        run_dir=run_dir, tag=cfg.get("tag", "test"),
        config=cfg, metrics_test=d["final"]["test"],
        metrics_val=d["final"].get("val", {}),
    )
    data_cache = {}
    grid = [0.0, 0.5, 1.0]
    res = cps.evaluate_run(rec, grid, data_cache)
    L0, Lhalf, L1 = res.L_alpha_exact
    expected_half = 0.5 * (L0 + L1)
    err = abs(Lhalf - expected_half)
    assert err <= COLLINEAR_TOL, \
        f"L_alpha not affine in alpha: L(0.5) - 0.5(L(0)+L(1)) = {err:.3e}"
