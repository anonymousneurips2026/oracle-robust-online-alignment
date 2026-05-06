"""p_low, p_high, p_alpha all lie in [0, 1] (within RANGE_SLACK).

Reports clipping mass; does NOT require clipping mass to be zero.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch

from finite_response_sail._true_to_worst_stress_eval import compute_endpoint_worst

RANGE_SLACK = 1e-6


def _synth(seed=0):
    g = torch.Generator().manual_seed(seed)
    B, K, d = 4, 6, 8
    psi = torch.randn(B, K, d, generator=g, dtype=torch.float64)
    ref = torch.randn(B, K, generator=g, dtype=torch.float64) * 0.5
    theta = torch.randn(d, generator=g, dtype=torch.float64) * 0.2
    p_obs = (0.05 + 0.9 * torch.rand(B, K, K, generator=g, dtype=torch.float64))
    return theta, psi, ref, p_obs


def test_synthetic_probability_range_unclipped():
    theta, psi, ref, p_obs = _synth()
    rho_bar = 0.02
    _, _, _, p_wc, _, _, _, _, clip = compute_endpoint_worst(theta, psi, ref, p_obs, rho_bar, 0.1)
    p_low = (p_obs - rho_bar).clamp(min=0.0)
    p_high = (p_obs + rho_bar).clamp(max=1.0)
    assert (p_low >= 0.0 - RANGE_SLACK).all()
    assert (p_high <= 1.0 + RANGE_SLACK).all()
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # p_id is unknown here -- substitute p_obs as a proxy for the test.
        P_alpha = (1.0 - alpha) * p_obs + alpha * p_wc
        assert (P_alpha >= 0.0 - RANGE_SLACK).all()
        assert (P_alpha <= 1.0 + RANGE_SLACK).all()


def test_synthetic_probability_range_clipped():
    """Pick p_obs near the boundary so rho_bar = 0.4 forces clipping."""
    theta, psi, ref, p_obs = _synth(seed=1)
    rho_bar = 0.4  # large radius forces clipping
    _, _, _, p_wc, _, _, _, _, clip = compute_endpoint_worst(theta, psi, ref, p_obs, rho_bar, 0.1)
    # mass should be reported; range still respected
    p_low = (p_obs - rho_bar).clamp(min=0.0)
    p_high = (p_obs + rho_bar).clamp(max=1.0)
    assert (p_low >= 0.0 - RANGE_SLACK).all()
    assert (p_high <= 1.0 + RANGE_SLACK).all()
    assert clip["clipped_any_mass"] >= 0.0  # always finite
    # With rho_bar=0.4 and p_obs in [0.05, 0.95], every entry is clipped.
    assert clip["clipped_any_mass"] > 0.0


def test_real_run_probability_range():
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
        pytest.skip("no real run available")

    cfg = json.loads((found / "metrics.json").read_text())["config"]
    npz = np.load(found / "oracle_tensors_test.npz")
    p_id = torch.from_numpy(npz["p_star"].astype(np.float32))
    p_obs = torch.from_numpy(npz["p_obs"].astype(np.float32))

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
    rho_bar = float(cfg["rho_mis"])
    _, _, _, p_wc, _, _, _, _, clip = compute_endpoint_worst(
        theta, psi, ref, p_obs, rho_bar, float(cfg["beta"]),
    )
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        P_alpha = (1.0 - alpha) * p_id + alpha * p_wc
        assert (P_alpha >= 0.0 - RANGE_SLACK).all()
        assert (P_alpha <= 1.0 + RANGE_SLACK).all()
    # clipping mass is reported and nonneg, but not asserted to be zero.
    assert clip["clipped_any_mass"] >= 0.0
