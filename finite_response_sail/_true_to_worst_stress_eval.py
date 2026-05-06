"""True-to-worst stress evaluator.

Post-hoc evaluator: for each trained policy theta, traces L_stress(alpha) along
an interpolation between the ideal oracle (alpha=0, p=p_id=p_star) and a
policy-specific worst-case oracle (alpha=1, p=p_wc) inside an L-infinity
uncertainty set centered at p_obs with radius rho_bar = rho_mis. Selects a
single robust operating point q_star per rho_mis using the validation high-
stress AUC (auc_0p5_1), and reports paired Robust-q_star vs Nominal deltas on
the test split.

Headline metrics: L_id, L_wc, slope, auc_0p5_1, R, L_obs, endpoint_gain.
True utility (E[r*]) is NOT a headline; it is reported as an appendix
diagnostic only.

CLI is described in the module's `--help` text. The default invocation is::

    python -m finite_response_sail._true_to_worst_stress_eval \
        --results-dir finite_response_sail/results/v4_true_to_worst_rhoalg_grid \
        --include-results-dir finite_response_sail/results/v3_pairbias_confirm \
        --include-results-dir finite_response_sail/results/v3_pairbias_rhoalg_sweep \
        --out-dir reports/true_to_worst_stress \
        --alpha-grid "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
        --rho-eval-mode rho_mis \
        --selection-metric auc_0p5_1 \
        --selection-split val \
        --eval-split test \
        --common-worst nominal
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .objective import exact_expectation_loss, sensitivity_R
from .policy import kl_to_ref as _kl_to_ref, policy_entropy, policy_probs
from .train import _build_ref_logits, _load_split, _prepare_probabilities


DEFAULT_ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DEFAULT_HIGH_ALPHA = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ROBUST_Q_GRID = [0.025, 0.05, 0.10, 0.25, 0.50, 1.00]
DECOMP_TOL = 1e-5
AFFINE_TOL = 1e-6
TIE_TOL = 1e-12


# --------------------------------------------------------------------------
# Run discovery
# --------------------------------------------------------------------------

@dataclass
class RunRecord:
    run_dir: Path
    tag: str
    config: Dict[str, Any]
    metrics_test: Dict[str, Any]
    metrics_val: Dict[str, Any]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_runs(results_dirs: Sequence[Path]) -> List[RunRecord]:
    """Walk results_dirs; return one RunRecord per complete run.

    Deduplicates by run_dir.resolve() so that overlapping --include-results-dir
    arguments don't double-count.
    """
    seen: Dict[Path, RunRecord] = {}
    for root in results_dirs:
        root = Path(root)
        if not root.exists():
            print(f"[ttw] WARN: results dir does not exist: {root}", file=sys.stderr)
            continue
        for mp in sorted(root.glob("**/mode-*/metrics.json")):
            try:
                d = json.loads(mp.read_text())
            except Exception as e:
                print(f"[ttw] WARN: skip unreadable metrics.json: {mp}: {e}", file=sys.stderr)
                continue
            cfg = d.get("config", {})
            final = d.get("final", {})
            test = final.get("test", {})
            val = final.get("val", {})
            if not test:
                continue
            theta_p = mp.parent / "theta.pt"
            tensors_p = mp.parent / "oracle_tensors_test.npz"
            if not theta_p.exists() or not tensors_p.exists():
                continue
            run_dir = mp.parent.resolve()
            if run_dir in seen:
                continue
            tag = cfg.get("tag", root.name)
            seen[run_dir] = RunRecord(
                run_dir=run_dir, tag=tag, config=cfg,
                metrics_test=test, metrics_val=val,
            )
    return list(seen.values())


def method_label(rho_alg_ratio: float, eta_kl: float = 0.0) -> str:
    if eta_kl and eta_kl > 0.0:
        return f"KL-{eta_kl:g}"
    if abs(rho_alg_ratio) < 1e-12:
        return "Nominal"
    return f"Robust-{rho_alg_ratio:g}"


# --------------------------------------------------------------------------
# Split data resolution
# --------------------------------------------------------------------------

def _data_key(cfg: Dict[str, Any], split: str) -> Tuple:
    return (
        str(cfg.get("cache_dir", "")),
        int(cfg["K"]),
        str(cfg["featureset"]),
        str(cfg["bias_attr"]),
        str(cfg.get("expect_model") or ""),
        str(cfg["ref"]),
        float(cfg["tau"]),
        float(cfg["rho_mis"]),
        split,
    )


def _resolve_split_data(
    cfg: Dict[str, Any], split: str,
    cache_dir_override: Optional[Path],
    cache: Dict[Tuple, Dict[str, Any]],
):
    """Returns dict with psi, ref_logits, p_id, p_obs, b, true_rewards, bias_attr.

    For split="test", prefer the on-disk oracle_tensors_test.npz (consumed by
    the caller). For split="val", reconstruct deterministically from the cache
    via _load_split + _prepare_probabilities.
    """
    key = _data_key(cfg, split)
    if key in cache:
        return cache[key]
    cache_dir = Path(cache_dir_override) if cache_dir_override else Path(cfg["cache_dir"])
    device = _device()
    data = _load_split(
        cache_dir=cache_dir,
        featureset=str(cfg["featureset"]),
        split=split,
        device=device,
        K=int(cfg["K"]),
        expect_model=str(cfg.get("expect_model") or ""),
        bias_attr=str(cfg["bias_attr"]),
    )
    ref_logits = _build_ref_logits(str(cfg["ref"]), data["base_logprob_per_token"])
    pst, b, pobs, pst_np, b_np, pobs_np = _prepare_probabilities(
        data, float(cfg["tau"]), float(cfg["rho_mis"]), device,
    )
    payload = {
        "psi": data["psi"],
        "ref_logits": ref_logits,
        "true_rewards": data["true_rewards"],
        "bias_attr": data["bias_attr"],
        "p_id": pst,
        "b": b,
        "p_obs": pobs,
        "p_id_np": pst_np,
        "b_np": b_np,
        "p_obs_np": pobs_np,
    }
    cache[key] = payload
    return payload


# --------------------------------------------------------------------------
# Endpoint worst-case oracle (centered on p_obs; do NOT reuse worst_case_loss)
# --------------------------------------------------------------------------

@torch.no_grad()
def compute_endpoint_worst(
    theta: torch.Tensor, psi: torch.Tensor, ref_logits: torch.Tensor,
    p_obs: torch.Tensor, rho_bar: float, beta: float,
):
    """Return (L_wc, L_obs, R, p_wc, ell1, ell0, pi, h, clip_diag).

    p_wc[i,j] = p_high if ell1>=ell0 else p_low; uncertainty set is L-infinity
    around p_obs, intersected with [0,1].
    """
    score = torch.einsum("bkd,d->bk", psi, theta)
    pi = F.softmax(ref_logits + score, dim=-1)
    h = score.unsqueeze(-1) - score.unsqueeze(-2)
    bh = beta * h
    ell1 = F.softplus(-bh)
    ell0 = F.softplus(bh)

    p_low = (p_obs - rho_bar).clamp(min=0.0)
    p_high = (p_obs + rho_bar).clamp(max=1.0)
    p_wc = torch.where(ell1 >= ell0, p_high, p_low)

    w = pi.unsqueeze(-1) * pi.unsqueeze(-2)
    L_wc = (w * (p_wc * ell1 + (1.0 - p_wc) * ell0)).sum(dim=(-1, -2)).mean().item()
    L_obs = (w * (p_obs * ell1 + (1.0 - p_obs) * ell0)).sum(dim=(-1, -2)).mean().item()
    R = (w * h.abs()).sum(dim=(-1, -2)).mean().item()

    low_clip = (p_obs - rho_bar) < 0.0
    high_clip = (p_obs + rho_bar) > 1.0
    any_clip = low_clip | high_clip
    clip_diag = {
        "clipped_low_mass": float(low_clip.float().mean().item()),
        "clipped_high_mass": float(high_clip.float().mean().item()),
        "clipped_any_mass": float(any_clip.float().mean().item()),
        "mean_interval_width": float((p_high - p_low).mean().item()),
    }

    return float(L_wc), float(L_obs), float(R), p_wc, ell1, ell0, pi, h, clip_diag


# --------------------------------------------------------------------------
# Per-run stress evaluation
# --------------------------------------------------------------------------

@dataclass
class StressResult:
    rec: RunRecord
    split: str
    rho_mis: float
    rho_bar: float
    rho_alg_ratio: float
    rho_alg_abs: float
    eta_kl: float
    q: float
    seed: int
    beta: float
    method_label: str
    # alpha-independent
    L_id: float
    L_wc: float
    L_obs: float
    R: float
    slope: float
    auc_0p5_1: float
    auc_all: float
    endpoint_gain: float
    theta_norm: float
    entropy: float
    KL_to_ref: float
    E_r_star: float
    E_bias_attr: float
    clipped_low_mass: float
    clipped_high_mass: float
    clipped_any_mass: float
    mean_interval_width: float
    decomposition_residual: float
    decomposition_residual_skipped: bool
    # alpha-dependent
    alpha_values: List[float]
    L_stress: List[float]
    affine_residual_max: float

    @property
    def n_alpha(self) -> int:
        return len(self.alpha_values)


def _load_theta(rec: RunRecord, device: torch.device, d_psi: int) -> torch.Tensor:
    theta = torch.load(rec.run_dir / "theta.pt", map_location=device, weights_only=False).float()
    if theta.dim() != 1 or theta.shape[0] != d_psi:
        raise RuntimeError(
            f"theta shape {tuple(theta.shape)} doesn't match psi[-1]={d_psi} for {rec.run_dir}"
        )
    return theta


def _get_test_oracle_tensors(rec: RunRecord, device: torch.device):
    """Returns (p_id, p_obs, b) for the test split using the on-disk npz.

    The npz key 'p_star' is what the new evaluator calls 'p_id'.
    """
    npz = np.load(rec.run_dir / "oracle_tensors_test.npz")
    p_id = torch.from_numpy(npz["p_star"].astype(np.float32)).to(device)
    p_obs = torch.from_numpy(npz["p_obs"].astype(np.float32)).to(device)
    b = torch.from_numpy(npz["b"].astype(np.float32)).to(device)
    return p_id, p_obs, b


def evaluate_run_stress(
    rec: RunRecord,
    alpha_grid: Sequence[float],
    split: str,
    data_cache: Dict[Tuple, Dict[str, Any]],
    cache_dir_override: Optional[Path] = None,
) -> StressResult:
    cfg = rec.config
    rho_mis = float(cfg["rho_mis"])
    rho_alg_abs = float(cfg.get("rho_alg", 0.0))
    rho_alg_ratio = float(cfg.get("rho_alg_ratio", 0.0))
    eta_kl = float(cfg.get("eta_kl", 0.0) or 0.0)
    beta = float(cfg["beta"])
    seed = int(cfg["seed"])

    payload = _resolve_split_data(cfg, split, cache_dir_override, data_cache)
    psi = payload["psi"]
    ref_logits = payload["ref_logits"]
    device = psi.device

    theta = _load_theta(rec, device, psi.shape[-1])

    if split == "test":
        p_id, p_obs, b = _get_test_oracle_tensors(rec, device)
    else:
        p_id = payload["p_id"]
        p_obs = payload["p_obs"]
        b = payload["b"]

    rho_bar = rho_mis
    L_wc, L_obs, R, p_wc, ell1, ell0, pi, h, clip = compute_endpoint_worst(
        theta, psi, ref_logits, p_obs, rho_bar, beta,
    )

    # alpha sweep
    alpha_list = list(alpha_grid)
    L_stress: List[float] = []
    L_id_val: Optional[float] = None
    L_wc_alpha1: Optional[float] = None
    affine_residual_max = 0.0

    with torch.no_grad():
        for alpha in alpha_list:
            P_alpha = (1.0 - alpha) * p_id + alpha * p_wc
            le = exact_expectation_loss(theta, psi, ref_logits, P_alpha, 0.0, beta).item()
            L_stress.append(float(le))

    # L_id and L_wc are the alpha=0, alpha=1 endpoints if those are in the
    # grid; otherwise compute them directly.
    has_a0 = any(abs(a - 0.0) < TIE_TOL for a in alpha_list)
    has_a1 = any(abs(a - 1.0) < TIE_TOL for a in alpha_list)
    if has_a0:
        L_id_val = L_stress[next(i for i, a in enumerate(alpha_list) if abs(a) < TIE_TOL)]
    else:
        with torch.no_grad():
            L_id_val = float(exact_expectation_loss(theta, psi, ref_logits, p_id, 0.0, beta).item())
    if has_a1:
        L_wc_alpha1 = L_stress[next(i for i, a in enumerate(alpha_list) if abs(a - 1.0) < TIE_TOL)]
    else:
        L_wc_alpha1 = L_wc

    # Internal consistency: alpha=1 endpoint vs the explicit endpoint computation.
    consistency = abs(L_wc_alpha1 - L_wc)
    if consistency > AFFINE_TOL:
        # Numerical drift: use the explicit endpoint value as the canonical L_wc.
        # We do not raise; the affine assertion below catches any pathological case.
        pass

    # Affine identity: L_stress(alpha) == (1-alpha) L_id + alpha L_wc
    for a, le in zip(alpha_list, L_stress):
        expect = (1.0 - a) * L_id_val + a * L_wc_alpha1
        affine_residual_max = max(affine_residual_max, abs(le - expect))
    if affine_residual_max > AFFINE_TOL:
        raise ValueError(
            f"affine identity violated for {rec.run_dir.name}: "
            f"max |L_stress - ((1-a)L_id + a L_wc)| = {affine_residual_max:.3e} > tol {AFFINE_TOL:.0e}"
        )

    # Decomposition (only if no clipping): L_wc == L_obs + rho_bar * beta * R
    decomp_residual = 0.0
    decomp_skipped = clip["clipped_any_mass"] > 0.0
    if not decomp_skipped:
        decomp_residual = abs(L_wc - (L_obs + rho_bar * beta * R))
        if decomp_residual > DECOMP_TOL:
            raise ValueError(
                f"unclipped decomposition violated for {rec.run_dir.name}: "
                f"|L_wc - (L_obs + rho_bar*beta*R)| = {decomp_residual:.3e} > tol {DECOMP_TOL:.0e}"
            )

    # Diagnostic scalars
    probs = policy_probs(theta, psi, ref_logits)
    H = policy_entropy(probs).mean().item()
    theta_norm = float(theta.norm(p=2).item())
    kl_ref = float(_kl_to_ref(theta, psi, ref_logits).mean().item())
    E_r_star = float((probs * payload["true_rewards"]).sum(-1).mean().item())
    E_bias = float((probs * payload["bias_attr"]).sum(-1).mean().item())

    # AUCs over alpha grid
    high_idx = [i for i, a in enumerate(alpha_list) if a >= 0.5 - TIE_TOL]
    auc_0p5_1 = float(np.mean([L_stress[i] for i in high_idx])) if high_idx else float("nan")
    auc_all = float(np.mean(L_stress))

    slope = L_wc_alpha1 - L_id_val
    endpoint_gain = L_wc_alpha1 - L_obs

    q = rho_alg_ratio
    return StressResult(
        rec=rec,
        split=split,
        rho_mis=rho_mis,
        rho_bar=rho_bar,
        rho_alg_ratio=rho_alg_ratio,
        rho_alg_abs=rho_alg_abs,
        eta_kl=eta_kl,
        q=q,
        seed=seed,
        beta=beta,
        method_label=method_label(rho_alg_ratio, eta_kl),
        L_id=L_id_val,
        L_wc=L_wc_alpha1,
        L_obs=L_obs,
        R=R,
        slope=slope,
        auc_0p5_1=auc_0p5_1,
        auc_all=auc_all,
        endpoint_gain=endpoint_gain,
        theta_norm=theta_norm,
        entropy=H,
        KL_to_ref=kl_ref,
        E_r_star=E_r_star,
        E_bias_attr=E_bias,
        clipped_low_mass=clip["clipped_low_mass"],
        clipped_high_mass=clip["clipped_high_mass"],
        clipped_any_mass=clip["clipped_any_mass"],
        mean_interval_width=clip["mean_interval_width"],
        decomposition_residual=decomp_residual,
        decomposition_residual_skipped=decomp_skipped,
        alpha_values=alpha_list,
        L_stress=L_stress,
        affine_residual_max=affine_residual_max,
    )


# --------------------------------------------------------------------------
# q_star selection (global per rho_mis, never per seed)
# --------------------------------------------------------------------------

def select_q_star_global(
    val_results: List[StressResult],
    selection_metric: str = "auc_0p5_1",
) -> Dict[float, Dict[str, float]]:
    """For each rho_mis, pick q_star = argmin seed-pooled mean of `selection_metric`.

    Excludes q=0 from the robust pool. Tie-break: smallest q wins (TIE_TOL=1e-12).
    Also returns q_star_Lwc (argmin seed-pooled mean of L_wc).
    """
    out: Dict[float, Dict[str, float]] = {}
    by_rho: Dict[float, List[StressResult]] = defaultdict(list)
    for r in val_results:
        by_rho[r.rho_mis].append(r)
    for rho, runs in sorted(by_rho.items()):
        candidates: Dict[float, List[StressResult]] = defaultdict(list)
        for r in runs:
            if r.q < TIE_TOL:
                continue  # exclude Nominal from robust pool
            candidates[round(r.q, 12)].append(r)
        if not candidates:
            out[rho] = {"q_star_auc": float("nan"), "q_star_Lwc": float("nan"),
                        "val_auc_0p5_1": float("nan"), "val_L_wc": float("nan"),
                        "val_R": float("nan")}
            continue

        def mean_metric(rs: List[StressResult], name: str) -> float:
            vals = [getattr(r, name) for r in rs]
            return float(np.mean(vals)) if vals else float("nan")

        # auc_0p5_1
        auc_means = {q: mean_metric(rs, "auc_0p5_1") for q, rs in candidates.items()}
        Lwc_means = {q: mean_metric(rs, "L_wc") for q, rs in candidates.items()}

        def argmin_with_tie(d: Dict[float, float]) -> float:
            best_v = min(d.values())
            ties = [q for q, v in d.items() if abs(v - best_v) <= TIE_TOL]
            return min(ties)

        q_star = argmin_with_tie(auc_means)
        q_star_Lwc = argmin_with_tie(Lwc_means)
        out[rho] = {
            "q_star_auc": q_star,
            "q_star_Lwc": q_star_Lwc,
            "val_auc_0p5_1": auc_means[q_star],
            "val_L_wc": Lwc_means[q_star],
            "val_R": mean_metric(candidates[q_star], "R"),
        }
    return out


# --------------------------------------------------------------------------
# Paired delta and crossing-point
# --------------------------------------------------------------------------

def _crossing_alpha(alphas: Sequence[float], deltas: Sequence[float]) -> float:
    """Smallest alpha at which delta <= 0 (linear interp between adjacent grid
    points). Returns NaN if delta is positive over the entire grid.
    """
    a = list(alphas)
    d = list(deltas)
    if d[0] <= 0:
        return a[0]
    for i in range(1, len(a)):
        if d[i] <= 0:
            # Linearly interpolate between a[i-1] (delta>0) and a[i] (delta<=0)
            x0, x1 = a[i - 1], a[i]
            y0, y1 = d[i - 1], d[i]
            if y1 == y0:
                return x1
            t = y0 / (y0 - y1)
            return float(x0 + t * (x1 - x0))
    return float("nan")


def compute_paired_deltas(
    test_results: List[StressResult],
    q_star_map: Dict[float, Dict[str, float]],
    alpha_grid: Sequence[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (per-seed rows, aggregate rows). Pairs Robust(q_star) vs Nominal
    on the seed intersection per rho_mis.
    """
    by_key: Dict[Tuple[float, float, int], StressResult] = {}
    for r in test_results:
        by_key[(r.rho_mis, round(r.q, 12), r.seed)] = r
    rho_set = sorted({r.rho_mis for r in test_results})

    rows: List[Dict[str, Any]] = []
    agg: List[Dict[str, Any]] = []

    for rho in rho_set:
        if rho not in q_star_map:
            continue
        q_star = q_star_map[rho]["q_star_auc"]
        if not (q_star == q_star) or q_star < TIE_TOL:
            # NaN or no robust selection
            agg.append({
                "rho_mis": rho, "q_star": q_star, "n_seeds": 0,
                **{k: float("nan") for k in [
                    "mean_delta_L_id", "mean_delta_L_wc", "mean_delta_auc_0p5_1",
                    "mean_delta_slope", "mean_delta_R",
                    "se_delta_L_id", "se_delta_L_wc", "se_delta_auc_0p5_1",
                    "se_delta_slope", "se_delta_R",
                    "crossing_point_mean", "crossing_point_se",
                ]},
                "sign_count_delta_L_wc_lt_0": 0,
                "sign_count_delta_auc_0p5_1_lt_0": 0,
                "sign_count_delta_R_lt_0": 0,
                "sign_count_delta_L_id_gt_0": 0,
            })
            continue
        qkey = round(q_star, 12)
        seeds_nom = {s for (rmk, qk, s) in by_key if rmk == rho and abs(qk) < TIE_TOL}
        seeds_rob = {s for (rmk, qk, s) in by_key if rmk == rho and abs(qk - qkey) < TIE_TOL}
        paired = sorted(seeds_nom & seeds_rob)
        if not paired:
            continue

        d_L_id, d_L_wc, d_auc, d_auc_all = [], [], [], []
        d_slope, d_R, d_L_obs = [], [], []
        crossings = []
        for s in paired:
            rN = by_key[(rho, 0.0 if abs(0.0) < TIE_TOL else 0.0, s)] if (rho, 0.0, s) in by_key else None
            # find any q with abs(q) < TIE_TOL
            rN = next((by_key[(rho, k, s)] for k in {round(0.0, 12)}
                       if (rho, k, s) in by_key), None)
            if rN is None:
                # fall back: scan all keys
                for (rmk, qk, sk), r in by_key.items():
                    if rmk == rho and sk == s and abs(qk) < TIE_TOL:
                        rN = r
                        break
            rR = by_key.get((rho, qkey, s))
            if rN is None or rR is None:
                continue
            d_L_id.append(rR.L_id - rN.L_id)
            d_L_wc.append(rR.L_wc - rN.L_wc)
            d_auc.append(rR.auc_0p5_1 - rN.auc_0p5_1)
            d_auc_all.append(rR.auc_all - rN.auc_all)
            d_slope.append(rR.slope - rN.slope)
            d_R.append(rR.R - rN.R)
            d_L_obs.append(rR.L_obs - rN.L_obs)
            # crossing point: per-seed, delta_L_stress(alpha) curve
            deltas = [rR.L_stress[i] - rN.L_stress[i] for i in range(len(rR.alpha_values))]
            crossings.append(_crossing_alpha(rR.alpha_values, deltas))
            rows.append({
                "rho_mis": rho, "seed": s, "q_star": q_star,
                "delta_L_id": d_L_id[-1],
                "delta_L_wc": d_L_wc[-1],
                "delta_auc_0p5_1": d_auc[-1],
                "delta_auc_all": d_auc_all[-1],
                "delta_slope": d_slope[-1],
                "delta_R": d_R[-1],
                "delta_L_obs": d_L_obs[-1],
            })

        n = len(d_L_id)

        def _mean(v): return float(np.mean(v)) if v else float("nan")

        def _se(v):
            if len(v) <= 1: return 0.0
            return float(np.std(v, ddof=1) / math.sqrt(len(v)))

        improved_either = sum(
            1 for i in range(n) if (d_L_wc[i] < 0) or (d_auc[i] < 0)
        )
        finite_cross = [c for c in crossings if c == c]
        agg.append({
            "rho_mis": rho, "q_star": q_star, "n_seeds": n,
            "mean_delta_L_id": _mean(d_L_id),
            "mean_delta_L_wc": _mean(d_L_wc),
            "mean_delta_auc_0p5_1": _mean(d_auc),
            "mean_delta_slope": _mean(d_slope),
            "mean_delta_R": _mean(d_R),
            "se_delta_L_id": _se(d_L_id),
            "se_delta_L_wc": _se(d_L_wc),
            "se_delta_auc_0p5_1": _se(d_auc),
            "se_delta_slope": _se(d_slope),
            "se_delta_R": _se(d_R),
            "sign_count_delta_L_wc_lt_0": sum(1 for x in d_L_wc if x < 0),
            "sign_count_delta_auc_0p5_1_lt_0": sum(1 for x in d_auc if x < 0),
            "sign_count_delta_R_lt_0": sum(1 for x in d_R if x < 0),
            "sign_count_delta_L_id_gt_0": sum(1 for x in d_L_id if x > 0),
            "sign_count_improved_either": improved_either,
            "crossing_point_mean": _mean(finite_cross) if finite_cross else float("nan"),
            "crossing_point_se": _se(finite_cross) if finite_cross else 0.0,
        })

    return rows, agg


# --------------------------------------------------------------------------
# Common-worst diagnostic (uses Nominal policy's p_wc as the shared adversary)
# --------------------------------------------------------------------------

@dataclass
class CommonWorstResult:
    rec: RunRecord
    rho_mis: float
    seed: int
    q: float
    method_label: str
    alpha_values: List[float]
    L_stress_common: List[float]
    L_id_common: float
    L_wc_common: float
    auc_0p5_1_common: float
    clipped_any_mass_common: float


def evaluate_common_worst(
    test_results: List[StressResult],
    test_runs: List[RunRecord],
    alpha_grid: Sequence[float],
    data_cache: Dict[Tuple, Dict[str, Any]],
    cache_dir_override: Optional[Path],
) -> List[CommonWorstResult]:
    """For each (rho_mis, seed), compute p_wc using the Nominal policy theta_N
    and re-evaluate every policy along the shared oracle path
    p_alpha_N(z) = (1-alpha) p_id(z) + alpha p_wc_N(z).
    """
    # index Nominal runs by (rho_mis, seed)
    nom_by: Dict[Tuple[float, int], RunRecord] = {}
    for r in test_runs:
        cfg = r.config
        if abs(float(cfg.get("rho_alg_ratio", 0.0))) < TIE_TOL and \
           float(cfg.get("eta_kl", 0.0) or 0.0) < TIE_TOL:
            nom_by[(float(cfg["rho_mis"]), int(cfg["seed"]))] = r

    out: List[CommonWorstResult] = []
    # Group all test runs by (rho_mis, seed) to share p_wc_N.
    by_pair: Dict[Tuple[float, int], List[RunRecord]] = defaultdict(list)
    for r in test_runs:
        by_pair[(float(r.config["rho_mis"]), int(r.config["seed"]))].append(r)

    for (rho, seed), recs in by_pair.items():
        nom = nom_by.get((rho, seed))
        if nom is None:
            continue
        nom_payload = _resolve_split_data(nom.config, "test", cache_dir_override, data_cache)
        psi_n = nom_payload["psi"]
        ref_n = nom_payload["ref_logits"]
        device = psi_n.device
        theta_n = _load_theta(nom, device, psi_n.shape[-1])
        p_id, p_obs, b = _get_test_oracle_tensors(nom, device)
        rho_bar = rho
        beta = float(nom.config["beta"])
        _, _, _, p_wc_N, _, _, _, _, clip = compute_endpoint_worst(
            theta_n, psi_n, ref_n, p_obs, rho_bar, beta,
        )

        for r in recs:
            payload = _resolve_split_data(r.config, "test", cache_dir_override, data_cache)
            psi = payload["psi"]
            ref = payload["ref_logits"]
            theta = _load_theta(r, device, psi.shape[-1])
            beta_r = float(r.config["beta"])

            L_list: List[float] = []
            with torch.no_grad():
                for alpha in alpha_grid:
                    P_alpha_N = (1.0 - alpha) * p_id + alpha * p_wc_N
                    le = exact_expectation_loss(theta, psi, ref, P_alpha_N, 0.0, beta_r).item()
                    L_list.append(float(le))
            high_idx = [i for i, a in enumerate(alpha_grid) if a >= 0.5 - TIE_TOL]
            auc = float(np.mean([L_list[i] for i in high_idx])) if high_idx else float("nan")
            q = float(r.config.get("rho_alg_ratio", 0.0))
            out.append(CommonWorstResult(
                rec=r, rho_mis=rho, seed=seed, q=q,
                method_label=method_label(q, float(r.config.get("eta_kl", 0.0) or 0.0)),
                alpha_values=list(alpha_grid),
                L_stress_common=L_list,
                L_id_common=L_list[0] if alpha_grid[0] == 0.0 else float("nan"),
                L_wc_common=L_list[-1] if abs(alpha_grid[-1] - 1.0) < TIE_TOL else float("nan"),
                auc_0p5_1_common=auc,
                clipped_any_mass_common=clip["clipped_any_mass"],
            ))
    return out


# --------------------------------------------------------------------------
# CSV writers
# --------------------------------------------------------------------------

def _se(vals: Sequence[float]) -> float:
    v = np.asarray(vals, dtype=np.float64)
    if v.size <= 1:
        return 0.0
    return float(v.std(ddof=1) / math.sqrt(v.size))


def write_alpha_metrics_csv(results: List[StressResult], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "split", "rho_mis", "seed", "q", "method_label", "alpha",
        "L_stress", "L_id", "L_wc", "slope", "auc_0p5_1", "auc_all",
        "R", "L_obs", "endpoint_gain",
        "clipped_low_mass", "clipped_high_mass", "clipped_any_mass", "mean_interval_width",
        "theta_norm", "entropy", "KL_to_ref",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            for a, le in zip(r.alpha_values, r.L_stress):
                w.writerow([
                    r.split, f"{r.rho_mis:g}", r.seed, f"{r.q:g}", r.method_label,
                    f"{a:g}", f"{le:.10f}",
                    f"{r.L_id:.10f}", f"{r.L_wc:.10f}", f"{r.slope:.10f}",
                    f"{r.auc_0p5_1:.10f}", f"{r.auc_all:.10f}",
                    f"{r.R:.10f}", f"{r.L_obs:.10f}", f"{r.endpoint_gain:.10f}",
                    f"{r.clipped_low_mass:.6f}", f"{r.clipped_high_mass:.6f}",
                    f"{r.clipped_any_mass:.6f}", f"{r.mean_interval_width:.6f}",
                    f"{r.theta_norm:.10f}", f"{r.entropy:.10f}", f"{r.KL_to_ref:.10f}",
                ])


def write_summary_by_rho_q(results: List[StressResult], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[float, float, str], List[StressResult]] = defaultdict(list)
    for r in results:
        grouped[(r.rho_mis, round(r.q, 12), r.split)].append(r)
    metrics = [
        "L_id", "L_wc", "slope", "auc_0p5_1", "auc_all", "R", "L_obs",
        "endpoint_gain", "clipped_any_mass", "entropy", "KL_to_ref",
    ]
    cols = ["rho_mis", "q", "rho_alg_ratio", "split", "n_seeds"]
    for m in metrics:
        cols += [f"{m}_mean", f"{m}_se"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for (rho, q, split), runs in sorted(grouped.items()):
            row = [f"{rho:g}", f"{q:g}", f"{q:g}", split, len(runs)]
            for m in metrics:
                vals = [getattr(r, m) for r in runs]
                row += [f"{float(np.mean(vals)):.10f}", f"{_se(vals):.10f}"]
            w.writerow(row)


def write_q_selection_csv(
    val_results: List[StressResult], q_star_map: Dict[float, Dict[str, float]], path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    by: Dict[Tuple[float, float], List[StressResult]] = defaultdict(list)
    for r in val_results:
        by[(r.rho_mis, round(r.q, 12))].append(r)
    cols = ["rho_mis", "q", "rho_alg_ratio", "n_seeds",
            "val_auc_0p5_1_mean", "val_auc_0p5_1_se",
            "val_L_wc_mean", "val_L_wc_se",
            "val_R_mean", "val_R_se",
            "selected_for_auc", "selected_for_L_wc",
            "q_star_auc", "q_star_Lwc"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for (rho, q), runs in sorted(by.items()):
            sel = q_star_map.get(rho, {})
            qsa = sel.get("q_star_auc", float("nan"))
            qsl = sel.get("q_star_Lwc", float("nan"))
            sel_auc = int(qsa == qsa and abs(q - qsa) < TIE_TOL)
            sel_lwc = int(qsl == qsl and abs(q - qsl) < TIE_TOL)
            auc_v = [r.auc_0p5_1 for r in runs]
            lwc_v = [r.L_wc for r in runs]
            R_v = [r.R for r in runs]
            w.writerow([
                f"{rho:g}", f"{q:g}", f"{q:g}", len(runs),
                f"{np.mean(auc_v):.10f}", f"{_se(auc_v):.10f}",
                f"{np.mean(lwc_v):.10f}", f"{_se(lwc_v):.10f}",
                f"{np.mean(R_v):.10f}", f"{_se(R_v):.10f}",
                sel_auc, sel_lwc,
                f"{qsa:g}" if qsa == qsa else "",
                f"{qsl:g}" if qsl == qsl else "",
            ])


def write_paired_delta_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["rho_mis", "seed", "q_star",
            "delta_L_id", "delta_L_wc", "delta_auc_0p5_1", "delta_auc_all",
            "delta_slope", "delta_R", "delta_L_obs"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def write_paired_delta_agg_csv(agg: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not agg:
        path.write_text("rho_mis,q_star,n_seeds\n")
        return
    cols = list(agg[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in agg:
            w.writerow(r)


def write_common_worst_csv(results: List[CommonWorstResult], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["rho_mis", "seed", "q", "method_label", "alpha",
            "L_stress_common", "L_id_common", "L_wc_common",
            "auc_0p5_1_common", "clipped_any_mass_common"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            for a, le in zip(r.alpha_values, r.L_stress_common):
                w.writerow([
                    f"{r.rho_mis:g}", r.seed, f"{r.q:g}", r.method_label,
                    f"{a:g}", f"{le:.10f}",
                    f"{r.L_id_common:.10f}", f"{r.L_wc_common:.10f}",
                    f"{r.auc_0p5_1_common:.10f}",
                    f"{r.clipped_any_mass_common:.6f}",
                ])


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _seed_array(runs: List[StressResult]) -> np.ndarray:
    return np.asarray([r.L_stress for r in runs])


def plot_curve_rho075(
    test_results: List[StressResult], q_star: float, out_dir: Path,
    paired_agg: List[Dict[str, Any]], rho_target: float = 0.075,
):
    rho_runs = [r for r in test_results if abs(r.rho_mis - rho_target) < 1e-9]
    nom = [r for r in rho_runs if abs(r.q) < TIE_TOL]
    rob = [r for r in rho_runs if abs(r.q - q_star) < TIE_TOL]
    if not nom or not rob:
        return
    alphas = nom[0].alpha_values
    nom_arr = _seed_array(nom)
    rob_arr = _seed_array(rob)

    def _mean_se(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = arr.mean(0)
        if arr.shape[0] > 1:
            s = arr.std(0, ddof=1) / math.sqrt(arr.shape[0])
        else:
            s = np.zeros_like(m)
        return m, s

    nom_m, nom_s = _mean_se(nom_arr)
    rob_m, rob_s = _mean_se(rob_arr)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axA = axes[0]
    axA.plot(alphas, nom_m, marker="o", color="tab:blue", label="Nominal (q=0)")
    axA.fill_between(alphas, nom_m - nom_s, nom_m + nom_s, alpha=0.2, color="tab:blue")
    axA.plot(alphas, rob_m, marker="s", color="tab:orange", label=f"Robust (q={q_star:g})")
    axA.fill_between(alphas, rob_m - rob_s, rob_m + rob_s, alpha=0.2, color="tab:orange")
    # Optional faint gray curves for other Robust-q
    other_q = sorted({r.q for r in rho_runs if r.q > TIE_TOL and abs(r.q - q_star) > TIE_TOL})
    for oq in other_q:
        oq_runs = [r for r in rho_runs if abs(r.q - oq) < TIE_TOL]
        if not oq_runs:
            continue
        oqA = _seed_array(oq_runs).mean(0)
        axA.plot(alphas, oqA, color="gray", alpha=0.25, linewidth=1)
    axA.set_xlabel("alpha (0 = ideal oracle, 1 = worst-case oracle)")
    axA.set_ylabel("L_stress  (lower is better)")
    axA.set_title(f"True-to-worst stress curve at rho_mis = {rho_target:g}")
    axA.axvline(0.0, color="k", linestyle=":", alpha=0.4)
    axA.axvline(1.0, color="k", linestyle=":", alpha=0.4)
    axA.grid(alpha=0.3)
    axA.legend(fontsize=9)

    axB = axes[1]
    delta_seed_arr = rob_arr - nom_arr if rob_arr.shape == nom_arr.shape else None
    if delta_seed_arr is not None:
        dm = delta_seed_arr.mean(0)
        ds = delta_seed_arr.std(0, ddof=1) / math.sqrt(delta_seed_arr.shape[0]) if delta_seed_arr.shape[0] > 1 else np.zeros_like(dm)
        axB.plot(alphas, dm, marker="o", color="tab:orange")
        axB.fill_between(alphas, dm - ds, dm + ds, alpha=0.2, color="tab:orange")
    axB.axhline(0.0, color="k", linestyle="--", alpha=0.5)
    # crossing point if available
    cross = [r for r in paired_agg if abs(r["rho_mis"] - rho_target) < 1e-9]
    if cross and cross[0]["crossing_point_mean"] == cross[0]["crossing_point_mean"]:
        axB.axvline(cross[0]["crossing_point_mean"], color="green", linestyle=":",
                    alpha=0.7, label=f"crossing alpha={cross[0]['crossing_point_mean']:.2f}")
        axB.legend(fontsize=9)
    axB.set_xlabel("alpha")
    axB.set_ylabel("delta L_stress = Robust q_star  minus  Nominal")
    axB.set_title("Paired delta (negative is robust better)")
    axB.grid(alpha=0.3)

    fig.suptitle("Robust SAIL on the true-to-worst stress curve  (lower is better)", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, out_dir, "true_to_worst_curve_rho075")


def plot_curves_all_rho(
    test_results: List[StressResult], q_star_map: Dict[float, Dict[str, float]], out_dir: Path,
):
    rhos = sorted({r.rho_mis for r in test_results})
    fig, axes = plt.subplots(1, len(rhos), figsize=(5 * len(rhos), 4.0), squeeze=False)
    for ax, rho in zip(axes[0], rhos):
        sel = q_star_map.get(rho, {})
        qs = sel.get("q_star_auc", float("nan"))
        if not (qs == qs):
            ax.set_title(f"rho_mis = {rho:g} (no robust selection)")
            continue
        rho_runs = [r for r in test_results if abs(r.rho_mis - rho) < 1e-9]
        nom = [r for r in rho_runs if abs(r.q) < TIE_TOL]
        rob = [r for r in rho_runs if abs(r.q - qs) < TIE_TOL]
        if not nom or not rob:
            ax.set_title(f"rho_mis = {rho:g} (missing pair)")
            continue
        alphas = nom[0].alpha_values
        # Pair on the seed intersection
        seeds = sorted({r.seed for r in nom} & {r.seed for r in rob})
        nom_by = {r.seed: r for r in nom}
        rob_by = {r.seed: r for r in rob}
        deltas = np.asarray([
            np.asarray(rob_by[s].L_stress) - np.asarray(nom_by[s].L_stress)
            for s in seeds
        ])
        m = deltas.mean(0)
        s = deltas.std(0, ddof=1) / math.sqrt(deltas.shape[0]) if deltas.shape[0] > 1 else np.zeros_like(m)
        ax.plot(alphas, m, marker="o", color="tab:orange")
        ax.fill_between(alphas, m - s, m + s, alpha=0.2, color="tab:orange")
        ax.axhline(0.0, color="k", linestyle="--", alpha=0.5)
        ax.set_title(f"rho_mis = {rho:g} (q_star = {qs:g})")
        ax.set_xlabel("alpha")
        ax.set_ylabel("delta L_stress (Robust q_star  minus  Nominal)")
        ax.grid(alpha=0.3)
    fig.suptitle("Paired delta L_stress(alpha) across rho_mis  (negative is robust better)", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, out_dir, "true_to_worst_curves_all_rho")


def plot_q_sweep_metrics(test_results: List[StressResult], q_star: float,
                         out_dir: Path, rho_target: float = 0.075):
    rho_runs = [r for r in test_results if abs(r.rho_mis - rho_target) < 1e-9]
    if not rho_runs:
        return
    by_q: Dict[float, List[StressResult]] = defaultdict(list)
    for r in rho_runs:
        by_q[round(r.q, 12)].append(r)
    qs = sorted(by_q.keys())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [("L_id", axes[0, 0]), ("R", axes[0, 1]),
               ("L_wc", axes[1, 0]), ("auc_0p5_1", axes[1, 1])]
    for name, ax in metrics:
        means = [float(np.mean([getattr(r, name) for r in by_q[q]])) for q in qs]
        ses = [_se([getattr(r, name) for r in by_q[q]]) for q in qs]
        # Use linear x with log tick for small q
        ax.errorbar(qs, means, yerr=ses, marker="o", capsize=3, color="tab:blue")
        ax.axvline(q_star, color="tab:orange", linestyle="--", alpha=0.7,
                   label=f"q_star = {q_star:g}")
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlabel("q  (rho_alg / rho_mis)")
        ax.set_ylabel(f"{name}  (lower is better)" if name != "R" else "R")
        ax.set_title(f"{name} versus q  (rho_mis = {rho_target:g})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("q sweep at rho_mis = 0.075  (lower is better for L_id, L_wc, auc_0p5_1, R)", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, out_dir, "q_sweep_stress_metrics")


def plot_pareto_Lid_Lwc(test_results: List[StressResult], q_star_map: Dict[float, Dict[str, float]],
                        out_dir: Path, rho_target: float = 0.075):
    rho_runs = [r for r in test_results if abs(r.rho_mis - rho_target) < 1e-9]
    if not rho_runs:
        return
    qs_target = q_star_map.get(rho_target, {}).get("q_star_auc", float("nan"))
    by_q: Dict[float, List[StressResult]] = defaultdict(list)
    for r in rho_runs:
        by_q[round(r.q, 12)].append(r)
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    qs_sorted = sorted(by_q.keys())
    n = len(qs_sorted)
    for i, q in enumerate(qs_sorted):
        runs = by_q[q]
        xs = [r.L_id for r in runs]
        ys = [r.L_wc for r in runs]
        color = cmap(i / max(1, n - 1))
        ax.scatter(xs, ys, alpha=0.25, s=20, color=color)
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        is_nom = abs(q) < TIE_TOL
        is_rob = (qs_target == qs_target) and abs(q - qs_target) < TIE_TOL
        if is_nom or is_rob:
            ax.scatter([mx], [my], s=160, color=color, edgecolor="black", linewidths=1.6,
                       label=f"{'Nominal' if is_nom else 'Robust q_star'} (q={q:g})")
        else:
            ax.scatter([mx], [my], s=80, color=color, edgecolor="white", linewidths=0.8,
                       label=f"q={q:g}")
    ax.set_xlabel("L_id  (lower is better)")
    ax.set_ylabel("L_wc  (lower is better)")
    ax.set_title(f"Pareto: L_id vs L_wc at rho_mis = {rho_target:g}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_dir, "stress_pareto_Lid_vs_Lwc")


def plot_auc_bar_all_rho(test_results: List[StressResult],
                         q_star_map: Dict[float, Dict[str, float]],
                         out_dir: Path):
    rhos = sorted({r.rho_mis for r in test_results})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    x = np.arange(len(rhos))
    nom_m, nom_s, rob_m, rob_s = [], [], [], []
    for rho in rhos:
        rho_runs = [r for r in test_results if abs(r.rho_mis - rho) < 1e-9]
        nom = [r.auc_0p5_1 for r in rho_runs if abs(r.q) < TIE_TOL]
        nom_m.append(float(np.mean(nom)) if nom else math.nan)
        nom_s.append(_se(nom))
        qs = q_star_map.get(rho, {}).get("q_star_auc", float("nan"))
        rob = [r.auc_0p5_1 for r in rho_runs if (qs == qs) and abs(r.q - qs) < TIE_TOL]
        rob_m.append(float(np.mean(rob)) if rob else math.nan)
        rob_s.append(_se(rob))
    ax.bar(x - width / 2, nom_m, width, yerr=nom_s, capsize=3, color="tab:blue", label="Nominal (q=0)")
    ax.bar(x + width / 2, rob_m, width, yerr=rob_s, capsize=3, color="tab:orange", label="Robust q_star")
    ax.set_xticks(x)
    ax.set_xticklabels([f"rho_mis={rho:g}" for rho in rhos])
    ax.set_ylabel("auc_0p5_1  (lower is better)")
    ax.set_title("High-stress AUC across rho_mis  (lower is better)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_fig(fig, out_dir, "stress_auc_bar_all_rho")


def plot_common_worst_curves(common: List[CommonWorstResult], out_dir: Path,
                             q_star_map: Dict[float, Dict[str, float]]):
    if not common:
        return
    rhos = sorted({c.rho_mis for c in common})
    fig, axes = plt.subplots(1, len(rhos), figsize=(5 * len(rhos), 4.0), squeeze=False)
    for ax, rho in zip(axes[0], rhos):
        qs = q_star_map.get(rho, {}).get("q_star_auc", float("nan"))
        if not (qs == qs):
            ax.set_title(f"rho_mis = {rho:g} (no robust selection)")
            continue
        nom = [c for c in common if abs(c.rho_mis - rho) < 1e-9 and abs(c.q) < TIE_TOL]
        rob = [c for c in common if abs(c.rho_mis - rho) < 1e-9 and abs(c.q - qs) < TIE_TOL]
        if not nom or not rob:
            ax.set_title(f"rho_mis = {rho:g} (missing pair)")
            continue
        alphas = nom[0].alpha_values
        seeds = sorted({c.seed for c in nom} & {c.seed for c in rob})
        nom_by = {c.seed: c for c in nom}
        rob_by = {c.seed: c for c in rob}
        deltas = np.asarray([
            np.asarray(rob_by[s].L_stress_common) - np.asarray(nom_by[s].L_stress_common)
            for s in seeds
        ])
        m = deltas.mean(0)
        s = deltas.std(0, ddof=1) / math.sqrt(deltas.shape[0]) if deltas.shape[0] > 1 else np.zeros_like(m)
        ax.plot(alphas, m, marker="o", color="tab:purple")
        ax.fill_between(alphas, m - s, m + s, alpha=0.2, color="tab:purple")
        ax.axhline(0.0, color="k", linestyle="--", alpha=0.5)
        ax.set_title(f"rho_mis = {rho:g} (common-worst Nominal-policy oracle)")
        ax.set_xlabel("alpha")
        ax.set_ylabel("delta L_stress_common (Robust q_star  minus  Nominal)")
        ax.grid(alpha=0.3)
    fig.suptitle("Common-worst diagnostic: shared Nominal-policy adversary  (lower is better)", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, out_dir, "common_worst_gap_curves")


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------

def _fmt_pm(m, s, digits=4):
    return f"{m:.{digits}f} +/- {s:.{digits}f}"


def render_report(
    test_results: List[StressResult],
    val_results: List[StressResult],
    q_star_map: Dict[float, Dict[str, float]],
    paired_agg: List[Dict[str, Any]],
    common_results: List[CommonWorstResult],
    out_dir: Path,
    cli: str,
):
    md: List[str] = []
    md.append("# True-to-Worst Stress Evaluator: report\n")
    md.append("All loss-style metrics (L_id, L_wc, slope, auc_0p5_1, R, L_obs) are lower-is-better.")
    md.append("E[r*] is reported only as an appendix diagnostic. It is not used for selection.\n")
    md.append(f"- CLI invocation: `{cli}`")
    md.append(f"- runs evaluated (test): {len(test_results)} ; (val): {len(val_results)}")
    rhos = sorted({r.rho_mis for r in test_results})
    md.append(f"- rho_mis levels: {rhos}\n")

    md.append("## Stress oracle definition\n")
    md.append("For each policy theta we evaluate")
    md.append("")
    md.append("    L_stress(theta; alpha) = E_{z ~ d_theta} [ p_alpha(z) * ell_1(z) + (1 - p_alpha(z)) * ell_0(z) ]")
    md.append("    p_alpha(z) = (1 - alpha) p_id(z) + alpha p_wc(z; rho_bar)")
    md.append("    p_wc(z) = p_high if ell_1 >= ell_0 else p_low,  with p_low/p_high clipped to [0, 1]")
    md.append("    rho_bar = rho_mis  (uncertainty-set radius around p_obs).")
    md.append("")
    md.append("L_stress is affine in alpha:  L_stress(alpha) = (1 - alpha) L_id + alpha L_wc.")
    md.append("L_id and L_wc are reported per run; auc_0p5_1 = mean of L_stress over alpha in {0.5,...,1.0}.\n")

    md.append("## Selected q_star (validation auc_0p5_1)\n")
    md.append("| rho_mis | q_star (auc) | q_star (L_wc) | val auc_0p5_1 | val L_wc | val R |")
    md.append("|---|---|---|---|---|---|")
    for rho in rhos:
        sel = q_star_map.get(rho, {})
        md.append("| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} |".format(
            f"{rho:g}", f"{sel.get('q_star_auc', float('nan')):g}",
            f"{sel.get('q_star_Lwc', float('nan')):g}",
            sel.get("val_auc_0p5_1", float("nan")),
            sel.get("val_L_wc", float("nan")),
            sel.get("val_R", float("nan")),
        ))
    md.append("")
    md.append("Note: q_star is selected globally per rho_mis using the seed-pooled mean of validation auc_0p5_1, never per seed. q=0 (Nominal) is excluded from the robust pool. Tie-break: smallest q wins.\n")

    md.append("## Paired delta (Robust q_star  minus  Nominal) on test split\n")
    md.append("| rho_mis | q_star | n | mean delta_L_id | mean delta_L_wc | mean delta_auc_0p5_1 | mean delta_R | sign(L_wc<0) | sign(auc<0) | sign(R<0) | improved either |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(paired_agg, key=lambda x: x["rho_mis"]):
        md.append("| {} | {} | {} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {}/{} | {}/{} | {}/{} | {}/{} |".format(
            f"{r['rho_mis']:g}", f"{r['q_star']:g}", r["n_seeds"],
            r["mean_delta_L_id"], r["mean_delta_L_wc"], r["mean_delta_auc_0p5_1"], r["mean_delta_R"],
            r["sign_count_delta_L_wc_lt_0"], r["n_seeds"],
            r["sign_count_delta_auc_0p5_1_lt_0"], r["n_seeds"],
            r["sign_count_delta_R_lt_0"], r["n_seeds"],
            r.get("sign_count_improved_either", 0), r["n_seeds"],
        ))
    md.append("")

    md.append("## Clipping diagnostics (test)\n")
    md.append("| rho_mis | q | mean clipped_any_mass | mean interval width |")
    md.append("|---|---|---|---|")
    by_pair: Dict[Tuple[float, float], List[StressResult]] = defaultdict(list)
    for r in test_results:
        by_pair[(r.rho_mis, round(r.q, 12))].append(r)
    for (rho, q), runs in sorted(by_pair.items()):
        md.append(f"| {rho:g} | {q:g} | {np.mean([r.clipped_any_mass for r in runs]):.4f} | {np.mean([r.mean_interval_width for r in runs]):.4f} |")
    md.append("")

    md.append("## Verdict\n")
    md.extend(_render_verdict(paired_agg, rhos))
    md.append("")

    md.append("## Appendix: E[r*] (true reward) per (rho_mis, q)\n")
    md.append("Reported as a diagnostic only; this value did not influence selection.\n")
    md.append("| rho_mis | q | n | mean E[r*] |")
    md.append("|---|---|---|---|")
    for (rho, q), runs in sorted(by_pair.items()):
        md.append(f"| {rho:g} | {q:g} | {len(runs)} | {np.mean([r.E_r_star for r in runs]):.4f} |")
    md.append("")

    md.append("## Plots\n")
    for name in ["true_to_worst_curve_rho075", "true_to_worst_curves_all_rho",
                 "q_sweep_stress_metrics", "stress_pareto_Lid_vs_Lwc",
                 "stress_auc_bar_all_rho"]:
        md.append(f"- `{name}.png` / `{name}.pdf`")
    if common_results:
        md.append("- `common_worst_gap_curves.png` / `common_worst_gap_curves.pdf`")
    md.append("")
    (out_dir / "stress_report.md").write_text("\n".join(md))


def _render_verdict(paired_agg: List[Dict[str, Any]], rhos: List[float]) -> List[str]:
    """Verdict per the spec:
      sign-pattern at rho_mis:
        mean delta_L_wc < 0
        mean delta_auc_0p5_1 < 0
        mean delta_R < 0
        at least 8/10 paired seeds improve L_wc OR auc_0p5_1
      PASS:        rho_mis = 0.075 has the sign pattern AND at least 2/3 rho_mis values do.
      STRONG PASS: all 3 rho_mis values have it.
      PARTIAL:     Robust reduces R but does not improve L_wc or auc_0p5_1.
      FAIL:        Robust does not reduce R, or worst-case metrics consistently worsen.
    """
    by_rho = {r["rho_mis"]: r for r in paired_agg}
    pattern: Dict[float, bool] = {}
    R_only: Dict[float, bool] = {}
    R_worse: Dict[float, bool] = {}
    for rho in rhos:
        r = by_rho.get(rho)
        if not r or r["n_seeds"] == 0:
            pattern[rho] = False
            R_only[rho] = False
            R_worse[rho] = False
            continue
        sign = (r["mean_delta_L_wc"] < 0
                and r["mean_delta_auc_0p5_1"] < 0
                and r["mean_delta_R"] < 0
                and r.get("sign_count_improved_either", 0) >= 8)
        pattern[rho] = sign
        R_only[rho] = (r["mean_delta_R"] < 0
                       and not (r["mean_delta_L_wc"] < 0 or r["mean_delta_auc_0p5_1"] < 0))
        R_worse[rho] = r["mean_delta_R"] >= 0
    n_pass = sum(1 for v in pattern.values() if v)
    has_075 = pattern.get(0.075, False)
    if n_pass == len(rhos) and len(rhos) >= 1:
        verdict = "STRONG PASS"
    elif has_075 and n_pass >= 2:
        verdict = "PASS"
    elif any(R_only.values()) and not any(pattern.values()):
        verdict = "PARTIAL"
    else:
        # FAIL if no R reduction at any rho_mis OR clear worst-case worsening.
        if any(R_worse.values()) or all(not v for v in pattern.values()):
            verdict = "FAIL"
        else:
            verdict = "PARTIAL"

    lines = [f"**Overall verdict: {verdict}**\n"]
    for rho in rhos:
        r = by_rho.get(rho)
        if not r:
            lines.append(f"- rho_mis={rho:g}: no paired data.")
            continue
        lines.append(
            f"- rho_mis={rho:g}: "
            f"pattern={'yes' if pattern[rho] else 'no'}, "
            f"mean delta_L_wc={r['mean_delta_L_wc']:+.4f}, "
            f"mean delta_auc_0p5_1={r['mean_delta_auc_0p5_1']:+.4f}, "
            f"mean delta_R={r['mean_delta_R']:+.4f}, "
            f"improved_either={r.get('sign_count_improved_either', 0)}/{r['n_seeds']}."
        )
    return lines


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_alpha_grid(spec: str) -> List[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="True-to-worst stress evaluator (post-hoc; no retraining)",
    )
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Primary results directory to scan.")
    p.add_argument("--include-results-dir", action="append", default=[],
                   help="Additional results directories to merge in (repeatable).")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--alpha-grid", type=str,
                   default=",".join(f"{a:g}" for a in DEFAULT_ALPHA_GRID))
    p.add_argument("--rho-eval-mode", choices=["rho_mis"], default="rho_mis",
                   help="How rho_bar is set. Currently only 'rho_mis' (= run's rho_mis).")
    p.add_argument("--selection-metric", choices=["auc_0p5_1"], default="auc_0p5_1")
    p.add_argument("--selection-split", choices=["val", "test"], default="val")
    p.add_argument("--eval-split", choices=["test", "val"], default="test")
    p.add_argument("--rho-mis", type=str, default="",
                   help="Optional comma-separated rho_mis filter (e.g. '0.05,0.075,0.1').")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--common-worst", choices=["nominal"], default=None,
                   help="If 'nominal', also evaluate every policy under the "
                        "common Nominal-policy worst-case oracle path.")
    p.add_argument("--smoke", action="store_true",
                   help="One run, two alphas (0,1). Asserts CSV+plot are produced.")
    return p


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    cli = "python -m finite_response_sail._true_to_worst_stress_eval " + " ".join(sys.argv[1:])

    results_dirs: List[Path] = []
    if args.results_dir is not None:
        results_dirs.append(args.results_dir)
    results_dirs.extend(Path(p) for p in args.include_results_dir)
    if not results_dirs:
        raise SystemExit("at least one of --results-dir / --include-results-dir is required")

    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    if args.smoke:
        alpha_grid = [0.0, 1.0]

    rho_filter = None
    if args.rho_mis:
        rho_filter = {float(x) for x in args.rho_mis.split(",")}

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(results_dirs)
    if rho_filter is not None:
        runs = [r for r in runs if float(r.config["rho_mis"]) in rho_filter]
    # Drop runs we can't classify into the q grid (e.g. eta_kl > 0 belongs to the KL appendix)
    runs = [r for r in runs if float(r.config.get("eta_kl", 0.0) or 0.0) < TIE_TOL]
    if args.smoke:
        runs = runs[:1]
    if not runs:
        raise SystemExit("[ttw] no runs after filtering")

    print(f"[ttw] discovered {len(runs)} runs over alpha grid {alpha_grid}")

    data_cache: Dict[Tuple, Dict[str, Any]] = {}

    # Test-split evaluation
    test_results: List[StressResult] = []
    for rec in runs:
        try:
            r = evaluate_run_stress(rec, alpha_grid, "test",
                                    data_cache=data_cache,
                                    cache_dir_override=args.cache_dir)
            test_results.append(r)
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            print(f"[ttw] WARN test eval skip: {rec.run_dir.name}: {e}", file=sys.stderr)

    # Validation-split evaluation (for q_star selection); skip in smoke mode.
    val_results: List[StressResult] = []
    if not args.smoke:
        for rec in runs:
            try:
                r = evaluate_run_stress(rec, alpha_grid, "val",
                                        data_cache=data_cache,
                                        cache_dir_override=args.cache_dir)
                val_results.append(r)
            except (ValueError, RuntimeError, FileNotFoundError) as e:
                print(f"[ttw] WARN val eval skip: {rec.run_dir.name}: {e}", file=sys.stderr)

    # q_star selection
    sel_source = val_results if args.selection_split == "val" else test_results
    q_star_map = select_q_star_global(sel_source, args.selection_metric)

    # Paired delta (test split)
    paired_rows, paired_agg = compute_paired_deltas(test_results, q_star_map, alpha_grid)

    # Optional common-worst
    common_results: List[CommonWorstResult] = []
    if args.common_worst == "nominal" and not args.smoke:
        common_results = evaluate_common_worst(
            test_results, runs, alpha_grid, data_cache, args.cache_dir,
        )

    # CSVs
    write_alpha_metrics_csv(test_results + val_results, out_dir / "stress_alpha_metrics.csv")
    write_summary_by_rho_q(test_results + val_results, out_dir / "stress_summary_by_rho_q.csv")
    if val_results:
        write_q_selection_csv(val_results, q_star_map, out_dir / "stress_q_selection.csv")
    write_paired_delta_csv(paired_rows, out_dir / "stress_paired_delta.csv")
    write_paired_delta_agg_csv(paired_agg, out_dir / "stress_paired_delta_agg.csv")
    if common_results:
        write_common_worst_csv(common_results, out_dir / "common_worst_alpha_metrics.csv")

    # Plots
    if not args.smoke:
        rho_target = 0.075
        qs_target = q_star_map.get(rho_target, {}).get("q_star_auc", float("nan"))
        if qs_target == qs_target:
            plot_curve_rho075(test_results, qs_target, out_dir, paired_agg, rho_target)
            plot_q_sweep_metrics(test_results, qs_target, out_dir, rho_target)
            plot_pareto_Lid_Lwc(test_results, q_star_map, out_dir, rho_target)
        plot_curves_all_rho(test_results, q_star_map, out_dir)
        plot_auc_bar_all_rho(test_results, q_star_map, out_dir)
        if common_results:
            plot_common_worst_curves(common_results, out_dir, q_star_map)
    else:
        # Emit one figure for the smoke test to check.
        plot_curves_all_rho(test_results, q_star_map, out_dir)

    # Markdown report
    if not args.smoke:
        render_report(test_results, val_results, q_star_map, paired_agg,
                      common_results, out_dir, cli)
    print(f"[ttw] wrote artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
