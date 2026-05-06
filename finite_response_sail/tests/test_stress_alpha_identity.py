"""Affine-in-alpha identity for the true-to-worst stress evaluator.

For every (theta, p_id, p_obs, b, rho_bar) the stress oracle is

    p_alpha(z) = (1 - alpha) p_id(z) + alpha p_wc(z; rho_bar)

so the integrand p_alpha * ell_1 + (1 - p_alpha) * ell_0 is affine in alpha,
and therefore L_stress(alpha) = (1 - alpha) L_id + alpha L_wc.

We assert ONLY internal identities. Specifically, we do NOT compare against
metrics_test["nominal_bt_loss"] -- that key has not been audited as the exact
loss under p_id, so a drift there would muddle this test's signal.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from finite_response_sail._true_to_worst_stress_eval import (
    AFFINE_TOL, compute_endpoint_worst,
)
from finite_response_sail.objective import exact_expectation_loss


def _synthetic_inputs(B=4, K=6, d=8, beta=0.1, rho_bar=0.05, seed=0,
                      dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    psi = torch.randn(B, K, d, generator=g, dtype=dtype)
    ref_logits = torch.randn(B, K, generator=g, dtype=dtype) * 0.5
    theta = torch.randn(d, generator=g, dtype=dtype) * 0.3
    # Build a valid p_id (Bradley-Terry like) and p_obs inside [0.1, 0.9]
    s = torch.randn(B, K, generator=g, dtype=dtype)
    diff = s.unsqueeze(-1) - s.unsqueeze(-2)
    p_id = torch.sigmoid(diff).clamp(0.15, 0.85)
    # antisymmetrize
    p_id = 0.5 + 0.5 * (p_id - p_id.transpose(-1, -2))
    # Ensure off-diagonal lies in (0.2, 0.8)
    p_id = p_id.clamp(0.2, 0.8)
    p_obs = p_id.clone()  # zero bias
    return theta, psi, ref_logits, p_id, p_obs, beta, rho_bar


def test_alpha_identity_synthetic_unclipped():
    theta, psi, ref, p_id, p_obs, beta, rho_bar = _synthetic_inputs(rho_bar=0.05)
    L_wc, L_obs, R, p_wc, ell1, ell0, pi, h, clip = compute_endpoint_worst(
        theta, psi, ref, p_obs, rho_bar, beta,
    )
    assert clip["clipped_any_mass"] == 0.0

    L_id = exact_expectation_loss(theta, psi, ref, p_id, 0.0, beta).item()

    # alpha=0 -> L_id; alpha=1 -> L_wc.
    P_a0 = (1.0 - 0.0) * p_id + 0.0 * p_wc
    P_a1 = (1.0 - 1.0) * p_id + 1.0 * p_wc
    L_a0 = exact_expectation_loss(theta, psi, ref, P_a0, 0.0, beta).item()
    L_a1 = exact_expectation_loss(theta, psi, ref, P_a1, 0.0, beta).item()
    assert abs(L_a0 - L_id) < 1e-7
    assert abs(L_a1 - L_wc) < 1e-7

    # Affine identity at intermediate alphas (float64 -> tight 1e-7 tolerance)
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        P_a = (1.0 - alpha) * p_id + alpha * p_wc
        L_a = exact_expectation_loss(theta, psi, ref, P_a, 0.0, beta).item()
        expect = (1.0 - alpha) * L_id + alpha * L_wc
        assert abs(L_a - expect) < 1e-7, (alpha, L_a, expect)


def test_alpha_identity_real_run_float32():
    """If a real test run is available, exercise the real-data float32 path
    with the same tolerances the evaluator uses.
    """
    from pathlib import Path
    candidates = [
        "finite_response_sail/results/v3_pairbias_confirm",
        "finite_response_sail/results/v3_pairbias_rhoalg_sweep",
    ]
    found = None
    for c in candidates:
        for mp in Path(c).glob("**/mode-*/metrics.json"):
            tp = mp.parent / "theta.pt"
            np_p = mp.parent / "oracle_tensors_test.npz"
            if tp.exists() and np_p.exists():
                found = mp.parent
                break
        if found is not None:
            break
    if found is None:
        import pytest
        pytest.skip("no real run available for float32 identity test")

    import json
    cfg = json.loads((found / "metrics.json").read_text())["config"]
    npz = np.load(found / "oracle_tensors_test.npz")
    p_id = torch.from_numpy(npz["p_star"].astype(np.float32))
    p_obs = torch.from_numpy(npz["p_obs"].astype(np.float32))

    # Reconstruct psi/ref via the standard loader
    from finite_response_sail.train import _load_split, _build_ref_logits
    cache_dir = Path(cfg["cache_dir"])
    data = _load_split(
        cache_dir=cache_dir, featureset=cfg["featureset"], split="test",
        device=torch.device("cpu"), K=int(cfg["K"]),
        expect_model=str(cfg.get("expect_model") or ""), bias_attr=cfg["bias_attr"],
    )
    ref = _build_ref_logits(cfg["ref"], data["base_logprob_per_token"])
    psi = data["psi"]
    theta = torch.load(found / "theta.pt", map_location="cpu", weights_only=False).float()
    beta = float(cfg["beta"])
    rho_bar = float(cfg["rho_mis"])

    L_wc, _, _, p_wc, _, _, _, _, _ = compute_endpoint_worst(
        theta, psi, ref, p_obs, rho_bar, beta,
    )
    L_id = exact_expectation_loss(theta, psi, ref, p_id, 0.0, beta).item()

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        P_a = (1.0 - alpha) * p_id + alpha * p_wc
        L_a = exact_expectation_loss(theta, psi, ref, P_a, 0.0, beta).item()
        expect = (1.0 - alpha) * L_id + alpha * L_wc
        assert abs(L_a - expect) < AFFINE_TOL, (alpha, L_a, expect)
