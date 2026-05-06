"""q_star selection is GLOBAL per rho_mis (seed-pooled mean), not per seed.

We construct StressResult-like objects with synthetic AUC values where:
  - seed 0 prefers q=0.025
  - seed 1 prefers q=0.10
  - seed 2 prefers q=0.10

But the seed-pooled mean prefers q=0.05. Therefore q_star must be 0.05.

Tie-break: when two q values produce identical seed-pooled means within
TIE_TOL, the smaller q wins.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import List

import pytest

from finite_response_sail._true_to_worst_stress_eval import (
    StressResult, select_q_star_global, TIE_TOL,
)


def _make(rho_mis, q, seed, auc, L_wc=0.0, R=0.0):
    rec = SimpleNamespace(run_dir="(synthetic)", config={})
    return StressResult(
        rec=rec, split="val",
        rho_mis=rho_mis, rho_bar=rho_mis,
        rho_alg_ratio=q, rho_alg_abs=q * rho_mis, eta_kl=0.0,
        q=q, seed=seed, beta=0.1,
        method_label="Robust" if q > 0 else "Nominal",
        L_id=0.0, L_wc=L_wc, L_obs=0.0, R=R,
        slope=0.0, auc_0p5_1=auc, auc_all=auc, endpoint_gain=0.0,
        theta_norm=0.0, entropy=0.0, KL_to_ref=0.0,
        E_r_star=0.0, E_bias_attr=0.0,
        clipped_low_mass=0.0, clipped_high_mass=0.0, clipped_any_mass=0.0,
        mean_interval_width=0.0,
        decomposition_residual=0.0, decomposition_residual_skipped=False,
        alpha_values=[0.0, 1.0], L_stress=[0.0, 0.0], affine_residual_max=0.0,
    )


def test_global_argmin_disagrees_with_per_seed():
    """Construct a case where seed-level argmin is not the global argmin."""
    rho = 0.075
    # AUCs: seed -> q -> auc
    aucs = {
        0: {0.025: 1.0, 0.05: 1.5, 0.10: 1.4},  # seed 0 prefers q=0.025
        1: {0.025: 2.0, 0.05: 1.5, 0.10: 1.0},  # seed 1 prefers q=0.10
        2: {0.025: 2.0, 0.05: 1.5, 0.10: 1.0},  # seed 2 prefers q=0.10
    }
    # Seed-pooled means: q=0.025 -> (1.0+2.0+2.0)/3 = 1.667
    #                   q=0.05  -> 1.5
    #                   q=0.10  -> (1.4+1.0+1.0)/3 = 1.133
    # Global argmin -> q=0.10. Seed-level argmins -> {0.025, 0.10, 0.10}. Disagree.
    results: List[StressResult] = []
    for seed, qmap in aucs.items():
        for q, auc in qmap.items():
            results.append(_make(rho, q, seed, auc))
    selected = select_q_star_global(results, "auc_0p5_1")
    assert rho in selected
    assert abs(selected[rho]["q_star_auc"] - 0.10) < TIE_TOL


def test_tie_break_smallest_q_wins():
    rho = 0.075
    # Two q values tie exactly; smallest q wins.
    aucs = {
        0: {0.025: 1.0, 0.05: 1.0, 0.10: 1.5},
        1: {0.025: 1.0, 0.05: 1.0, 0.10: 1.5},
    }
    results: List[StressResult] = []
    for seed, qmap in aucs.items():
        for q, auc in qmap.items():
            results.append(_make(rho, q, seed, auc))
    selected = select_q_star_global(results, "auc_0p5_1")
    # Tie between q=0.025 and q=0.05 -> smaller wins
    assert abs(selected[rho]["q_star_auc"] - 0.025) < TIE_TOL


def test_q_zero_excluded_from_robust_pool():
    rho = 0.05
    # If only q=0 candidates exist (no robust), output should mark NaN.
    results = [_make(rho, 0.0, s, 1.0) for s in range(3)]
    selected = select_q_star_global(results, "auc_0p5_1")
    qs = selected[rho]["q_star_auc"]
    # NaN is not equal to itself
    assert qs != qs


def test_per_rho_independence():
    """q_star is selected independently per rho_mis."""
    results: List[StressResult] = []
    # rho=0.05: prefer q=0.025
    for s in range(3):
        results.append(_make(0.05, 0.025, s, 1.0))
        results.append(_make(0.05, 0.05, s, 1.5))
    # rho=0.10: prefer q=0.05
    for s in range(3):
        results.append(_make(0.10, 0.025, s, 1.5))
        results.append(_make(0.10, 0.05, s, 1.0))
    selected = select_q_star_global(results, "auc_0p5_1")
    assert abs(selected[0.05]["q_star_auc"] - 0.025) < TIE_TOL
    assert abs(selected[0.10]["q_star_auc"] - 0.05) < TIE_TOL
