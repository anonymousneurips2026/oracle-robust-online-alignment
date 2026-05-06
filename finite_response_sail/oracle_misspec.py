"""Probability-level oracle misspecification (v3_pairbias).

Pure numpy. No torch. No probability clipping: the oracle is the raw
Bradley-Terry p_star plus a *pointwise pairwise* bias term b_ij whose sign is
anti-aligned with the true-reward direction. The bias activates only when the
bias attribute direction conflicts with (or is neutral to) the true-reward
direction, so a high bias attribute cannot co-occur with a high true reward
under the perturbation.

Definitions:
    p_star_ij = sigmoid(tau * (r_i - r_j))
    ds_ij     = s_i - s_j
    dr_ij     = r_i - r_j
    M_x       = max_{u,v} |s_u - s_v| + eps
    b_ij      = sign(ds_ij) * 1[ds_ij * dr_ij <= 0] * |ds_ij| / M_x
              = (ds_ij / M_x) * 1[ds_ij * dr_ij <= 0]
    p_obs_ij  = p_star_ij + rho_mis * b_ij

Guarantees:
    b_ii == 0 ; b_ji == -b_ij ; |b_ij| <= 1
    p_obs_ji == 1 - p_obs_ij ; p_obs_ii == 0.5
    |p_obs_ij - p_star_ij| <= rho_mis
    p_obs in [0, 1] whenever rho_mis < empirical_margin_delta(p_star)

No probability clipping is used anywhere. Instead we enforce the margin
condition at the data-generating boundary: callers must pass rho_mis smaller
than the empirical margin delta of p_star; check_margin() raises if not.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def normalize_per_prompt(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-prompt min-max to [0, 1] along the last axis."""
    x = np.asarray(x, dtype=np.float64)
    lo = x.min(axis=-1, keepdims=True)
    hi = x.max(axis=-1, keepdims=True)
    span = np.maximum(hi - lo, eps)
    return (x - lo) / span


def compute_p_star(true_reward: np.ndarray, tau: float) -> np.ndarray:
    r = np.asarray(true_reward, dtype=np.float64)
    diff = r[..., :, None] - r[..., None, :]
    return _stable_sigmoid(tau * diff)


def compute_pairwise_bias(
    true_reward: np.ndarray,
    bias_attr: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    r = np.asarray(true_reward, dtype=np.float64)
    s = np.asarray(bias_attr, dtype=np.float64)
    if r.shape != s.shape:
        raise ValueError(
            f"true_reward and bias_attr shape mismatch: {r.shape} vs {s.shape}"
        )
    ds = s[..., :, None] - s[..., None, :]
    dr = r[..., :, None] - r[..., None, :]
    abs_ds = np.abs(ds)
    M_x = abs_ds.reshape(*abs_ds.shape[:-2], -1).max(axis=-1, keepdims=False)
    M_x = M_x[..., None, None] + eps
    active = (ds * dr <= 0.0).astype(np.float64)
    return np.sign(ds) * abs_ds / M_x * active


def compute_p_obs(p_star: np.ndarray, b: np.ndarray, rho_mis: float) -> np.ndarray:
    p = np.asarray(p_star, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return p + rho_mis * b


def empirical_margin_delta(p_star: np.ndarray) -> float:
    """min over all entries of min(p, 1-p). No clipping or exclusion."""
    p = np.asarray(p_star, dtype=np.float64)
    return float(np.minimum(p, 1.0 - p).min())


def check_margin(p_star: np.ndarray, rho_mis: float, slack: float = 1e-8) -> None:
    if rho_mis <= 0.0:
        return
    delta = empirical_margin_delta(p_star)
    if rho_mis > delta - slack:
        raise ValueError(
            "p_star margin too small for rho_mis; lower tau or lower rho_mis."
            f" delta_empirical={delta:.5f}, rho_mis={rho_mis:.5f}"
        )


def _offdiag_mask(K: int) -> np.ndarray:
    return 1.0 - np.eye(K, dtype=np.float64)


def _corr_per_prompt_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation per prompt along the last axis; NaN if zero std."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_c = a - a.mean(axis=-1, keepdims=True)
    b_c = b - b.mean(axis=-1, keepdims=True)
    num = (a_c * b_c).sum(axis=-1)
    denom = np.sqrt((a_c ** 2).sum(axis=-1) * (b_c ** 2).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 0, num / denom, np.nan)
    return r


def oracle_diagnostics(
    p_star: np.ndarray,
    p_obs: np.ndarray,
    b: np.ndarray,
    bias_attr: np.ndarray,
    true_reward: np.ndarray,
    pi: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Pure-diagnostic view of the oracle. All returns are finite floats."""
    p_star = np.asarray(p_star, dtype=np.float64)
    p_obs = np.asarray(p_obs, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    bias_attr = np.asarray(bias_attr, dtype=np.float64)
    true_reward = np.asarray(true_reward, dtype=np.float64)

    K = p_star.shape[-1]
    off = _offdiag_mask(K)
    off_bool = off > 0

    diff = p_obs - p_star
    abs_diff = np.abs(diff)

    # Bias geometry flags (off-diagonal only).
    ds = bias_attr[..., :, None] - bias_attr[..., None, :]
    dr = true_reward[..., :, None] - true_reward[..., None, :]
    nonzero_ds = np.abs(ds) > 0
    conflict = (ds * dr) < 0.0
    same_dir = (ds * dr) > 0.0
    active = np.abs(b) > 0.0

    # Broadcast off mask to full shape.
    off_b = np.broadcast_to(off_bool, p_star.shape)

    def _fmean(mask: np.ndarray, values: np.ndarray) -> float:
        vals = values[mask]
        if vals.size == 0:
            return 0.0
        return float(vals.mean())

    def _frate(mask_numer: np.ndarray, mask_denom: np.ndarray) -> float:
        num = int(mask_numer.sum())
        den = int(mask_denom.sum())
        return float(num) / float(den) if den > 0 else 0.0

    # Per-prompt bias/true-reward correlation.
    corr = _corr_per_prompt_np(bias_attr, true_reward)
    corr_valid = corr[~np.isnan(corr)]
    bias_attr_true_corr = float(corr_valid.mean()) if corr_valid.size else 0.0
    bias_attr_true_corr_median = float(np.median(corr_valid)) if corr_valid.size else 0.0

    out: Dict[str, float] = {
        "delta_empirical": empirical_margin_delta(p_star),
        "min_p_star": float(p_star.min()),
        "max_p_star": float(p_star.max()),
        "max_abs_p_obs_minus_p_star": float(abs_diff.max()),
        "mean_abs_p_obs_minus_p_star": float(abs_diff.mean()),
        "max_abs_b": float(np.abs(b).max()),
        "mean_abs_b": float(np.abs(b).mean()),
        "bias_active_rate": _frate(active & off_b, off_b),
        "bias_conflict_rate": _frate(conflict & nonzero_ds & off_b, nonzero_ds & off_b),
        "bias_same_direction_zero_rate": _frate(same_dir & off_b, off_b),
        "bias_attr_true_corr": bias_attr_true_corr,
        "bias_attr_true_corr_median": bias_attr_true_corr_median,
    }

    if pi is not None:
        pi = np.asarray(pi, dtype=np.float64)
        # w_ij = pi_i * pi_j per prompt.
        w = pi[..., :, None] * pi[..., None, :]
        # Average pair-weighted |b| and signed b per prompt, then over prompts.
        pair_mass = (w * np.abs(b)).sum(axis=(-1, -2))
        signed = (w * b).sum(axis=(-1, -2))
        out["pair_bias_mass_under_pi"] = float(pair_mass.mean())
        out["signed_pair_bias_under_pi"] = float(signed.mean())
    return out
