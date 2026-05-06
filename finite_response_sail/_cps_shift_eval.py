"""Controlled Preference Shift (CPS) evaluator.

Pure post-hoc evaluation. No retraining. For each existing run:
  - Load theta.pt, config.json, oracle_tensors_test.npz.
  - Resolve cached test psi/ref_logits via the same loader train.py used.
  - For each alpha in the grid, compute
        P_alpha = p_star + alpha * rho_mis * b
    and
        L_alpha_exact   = exact_expectation_loss(theta, psi, ref, P_alpha, 0, beta)
        L_alpha_offdiag = offdiag_exact_loss   (theta, psi, ref, P_alpha, 0, beta)
  - Run identity checks (alpha=0/1 vs nominal_bt_loss/biased_bt_loss),
    antisymmetry, and main-grid range checks.

CSV outputs (in --out-dir):
    cps_alpha_metrics.csv          (one row per (run, alpha))
    cps_alpha_metrics_offdiag.csv  (appendix; same shape, offdiag loss)
    cps_summary_by_rho_method.csv  (mean/SE per (rho_mis, method))
    cps_paired_delta.csv           (Robust - Nominal per (rho_mis, alpha, seed))
    cps_paired_delta_aggregate.csv (mean/SE/sign-counts per (rho_mis, alpha))
    cps_alpha_metrics.md           (compact markdown rendering)
    cps_report.md                  (full markdown report)

Plot outputs (PNG + PDF, dpi=300):
    cps_loss_curves_by_rho.{png,pdf}
    cps_delta_loss_curves_by_rho.{png,pdf}
    cps_worst_avg_bar.{png,pdf}
    cps_heatmap_delta_L.{png,pdf}
    cps_pareto_true_vs_worst.{png,pdf}

L_alpha is affine in alpha (P_alpha enters BT loss linearly), so the dense
grid is exact endpoint linear interpolation. The summary tables surface
explicit endpoint/slope columns (L_alpha_0, L_alpha_1, shift_slope,
endpoint_worst_loss) and the report says so plainly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .objective import exact_expectation_loss
from .offdiag_objective import offdiag_exact_loss
from .policy import kl_to_ref as _kl_to_ref, policy_entropy, policy_probs, top1_prob
from .train import _build_ref_logits, _load_split
from ._field_resolver import FIELD_MAP, field_map_markdown_table, has_field, resolve

DEFAULT_ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DEFAULT_ALPHA_GRID_STRESS = [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0]

IDENTITY_TOL = 1e-5
ANTISYM_TOL = 1e-6
RANGE_SLACK = 1e-6


# --------------------------------------------------------------------------
# Run discovery and loading
# --------------------------------------------------------------------------

@dataclass
class RunRecord:
    run_dir: Path
    tag: str
    config: Dict[str, Any]
    metrics_test: Dict[str, Any]   # final.test
    metrics_val: Dict[str, Any]    # final.val (may be empty if missing)


def _parse_alpha_grid(spec: str) -> List[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def discover_runs(results_dirs: Sequence[Path]) -> List[RunRecord]:
    """Walk results dirs, return one RunRecord per complete run."""
    out: List[RunRecord] = []
    for root in results_dirs:
        root = Path(root)
        if not root.exists():
            print(f"[cps] WARN: results dir does not exist: {root}", file=sys.stderr)
            continue
        for mp in sorted(root.glob("mode-*/metrics.json")):
            try:
                d = json.loads(mp.read_text())
            except Exception as e:
                print(f"[cps] WARN: skip unreadable metrics.json: {mp}: {e}", file=sys.stderr)
                continue
            cfg = d.get("config", {})
            final = d.get("final", {})
            test = final.get("test", {})
            val = final.get("val", {})
            if not test or "expected_true_utility" not in test:
                continue
            theta_p = mp.parent / "theta.pt"
            tensors_p = mp.parent / "oracle_tensors_test.npz"
            if not theta_p.exists() or not tensors_p.exists():
                continue
            tag = cfg.get("tag", root.name)
            out.append(RunRecord(
                run_dir=mp.parent,
                tag=tag,
                config=cfg,
                metrics_test=test,
                metrics_val=val,
            ))
    return out


def method_label(rho_alg_ratio: float, eta_kl: float = 0.0) -> str:
    if eta_kl and eta_kl > 0.0:
        return f"KL-{eta_kl:g}"
    if abs(rho_alg_ratio) < 1e-12:
        return "Nominal"
    if abs(rho_alg_ratio - 0.025) < 1e-9:
        return "Robust-0.025rho"
    return f"Robust-{rho_alg_ratio:g}rho"


# --------------------------------------------------------------------------
# Test-split data resolution (cached across runs of the same featureset/K)
# --------------------------------------------------------------------------

def _data_key(cfg: Dict[str, Any]) -> Tuple[str, int, str, str, str]:
    return (
        str(cfg.get("cache_dir", "")),
        int(cfg["K"]),
        str(cfg["featureset"]),
        str(cfg["bias_attr"]),
        str(cfg["expect_model"]) if cfg.get("expect_model") else "",
    )


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_test_data(cfg: Dict[str, Any], cache_dir_override: Optional[Path],
                       cache: Dict[Tuple, Tuple[Dict[str, Any], torch.Tensor]]):
    """Returns (data_dict, ref_logits) for the test split.

    Memoizes by (cache_dir, K, featureset, bias_attr, expect_model).
    """
    key = _data_key(cfg)
    if key in cache:
        return cache[key]
    cache_dir = Path(cache_dir_override) if cache_dir_override else Path(cfg["cache_dir"])
    data = _load_split(
        cache_dir=cache_dir,
        featureset=str(cfg["featureset"]),
        split="test",
        device=_device(),
        K=int(cfg["K"]),
        expect_model=str(cfg.get("expect_model") or ""),
        bias_attr=str(cfg["bias_attr"]),
    )
    ref_logits = _build_ref_logits(str(cfg["ref"]), data["base_logprob_per_token"])
    cache[key] = (data, ref_logits)
    return data, ref_logits


# --------------------------------------------------------------------------
# Per-run CPS evaluation
# --------------------------------------------------------------------------

@dataclass
class CPSResult:
    rec: RunRecord
    rho_mis: float
    rho_alg_abs: float
    rho_alg_ratio: float
    eta_kl: float
    method: str
    seed: int
    beta: float
    # alpha-independent
    R: float
    E_r_star: float
    E_bias_attr: float
    entropy: float
    top1: float
    theta_norm: float
    KL_to_ref: float
    # alpha-dependent
    alpha_values: List[float]
    L_alpha_exact: List[float]
    L_alpha_offdiag: List[float]
    # identity check residuals
    id_residual_alpha0: float        # |L_a0 - nominal_bt_loss|
    id_residual_alpha1: float        # |L_a1 - biased_bt_loss|
    antisym_residual: float          # max |P_a[j,i] + P_a[i,j] - 1| over alphas+pairs
    range_min: float                 # min over main grid of P_alpha
    range_max: float                 # max over main grid of P_alpha
    has_alpha0: bool
    has_alpha1: bool

    def shift_slope_exact(self) -> Optional[float]:
        if not (self.has_alpha0 and self.has_alpha1):
            return None
        i0 = self.alpha_values.index(0.0)
        i1 = self.alpha_values.index(1.0)
        return self.L_alpha_exact[i1] - self.L_alpha_exact[i0]

    def L_alpha_0_exact(self) -> Optional[float]:
        if not self.has_alpha0:
            return None
        return self.L_alpha_exact[self.alpha_values.index(0.0)]

    def L_alpha_1_exact(self) -> Optional[float]:
        if not self.has_alpha1:
            return None
        return self.L_alpha_exact[self.alpha_values.index(1.0)]

    def endpoint_worst_loss_exact(self) -> Optional[float]:
        a0 = self.L_alpha_0_exact()
        a1 = self.L_alpha_1_exact()
        if a0 is None or a1 is None:
            return None
        return max(a0, a1)


def evaluate_run(
    rec: RunRecord,
    alpha_grid: Sequence[float],
    data_cache: Dict[Tuple, Tuple],
    cache_dir_override: Optional[Path] = None,
    enforce_range: bool = True,
) -> CPSResult:
    """Run CPS evaluation on one run. Raises ValueError on identity-check failure."""
    cfg = rec.config
    rho_mis = float(cfg["rho_mis"])
    rho_alg = float(cfg.get("rho_alg", 0.0))
    rho_alg_ratio = float(cfg.get("rho_alg_ratio", 0.0))
    eta_kl = float(cfg.get("eta_kl", 0.0) or 0.0)
    beta = float(cfg["beta"])
    seed = int(cfg["seed"])

    data, ref_logits = _resolve_test_data(cfg, cache_dir_override, data_cache)
    psi = data["psi"]
    device = psi.device

    # Load theta and the cached test-split p_star/b (already aligned with psi
    # since both come from the same test-split cache).
    theta = torch.load(rec.run_dir / "theta.pt", map_location=device, weights_only=False).float()
    if theta.dim() != 1 or theta.shape[0] != psi.shape[-1]:
        raise RuntimeError(
            f"theta shape {tuple(theta.shape)} doesn't match psi[-1]={psi.shape[-1]} for {rec.run_dir}"
        )
    npz = np.load(rec.run_dir / "oracle_tensors_test.npz")
    p_star_np = npz["p_star"].astype(np.float64)
    b_np = npz["b"].astype(np.float64)

    # alpha-independent metrics
    R_val = _R_under_pi(theta, psi, ref_logits)
    probs = policy_probs(theta, psi, ref_logits)
    pi_ref = torch.softmax(ref_logits, dim=-1)
    E_r_star = (probs * data["true_rewards"]).sum(-1).mean().item()
    E_bias = (probs * data["bias_attr"]).sum(-1).mean().item()
    H = policy_entropy(probs).mean().item()
    top1 = top1_prob(probs).mean().item()
    theta_norm = float(theta.norm(p=2).item())
    kl_ref = float(_kl_to_ref(theta, psi, ref_logits).mean().item())

    # alpha-dependent: compute P_alpha and the two losses
    alpha_values = list(alpha_grid)
    L_exact: List[float] = []
    L_offdiag: List[float] = []
    range_min = math.inf
    range_max = -math.inf
    antisym_residual = 0.0

    for alpha in alpha_values:
        P_alpha_np = p_star_np + (alpha * rho_mis) * b_np
        P_alpha = torch.from_numpy(P_alpha_np).float().to(device)
        # antisymmetry: P_alpha[j,i] + P_alpha[i,j] - 1 should be ~0
        sym = (P_alpha + P_alpha.transpose(-1, -2) - 1.0).abs().max().item()
        antisym_residual = max(antisym_residual, sym)
        range_min = min(range_min, float(P_alpha.min().item()))
        range_max = max(range_max, float(P_alpha.max().item()))
        with torch.no_grad():
            le = exact_expectation_loss(theta, psi, ref_logits, P_alpha, 0.0, beta).item()
            lo = offdiag_exact_loss(theta, psi, ref_logits, P_alpha, 0.0, beta).item()
        L_exact.append(float(le))
        L_offdiag.append(float(lo))

    # identity checks vs the existing metrics.json fields (exact path only)
    has_a0 = any(abs(a - 0.0) < 1e-12 for a in alpha_values)
    has_a1 = any(abs(a - 1.0) < 1e-12 for a in alpha_values)
    id_a0 = math.nan
    id_a1 = math.nan
    if has_a0:
        nom = float(rec.metrics_test.get("nominal_bt_loss", math.nan))
        id_a0 = abs(L_exact[alpha_values.index(0.0)] - nom)
    if has_a1:
        bi = float(rec.metrics_test.get("biased_bt_loss", math.nan))
        id_a1 = abs(L_exact[alpha_values.index(1.0)] - bi)

    # Range check (no silent clip): main grid must satisfy [0, 1].
    if enforce_range and (range_min < -RANGE_SLACK or range_max > 1.0 + RANGE_SLACK):
        raise ValueError(
            f"P_alpha out of [0,1] on main grid for {rec.run_dir.name} "
            f"(min={range_min:.6e}, max={range_max:.6e}). "
            f"Either lower rho_mis, restrict alpha grid, or pass --no-enforce-range "
            f"to emit a clipped diagnostic instead."
        )
    if antisym_residual > ANTISYM_TOL:
        raise ValueError(
            f"P_alpha antisymmetry violated for {rec.run_dir.name}: "
            f"max residual {antisym_residual:.3e} > tol {ANTISYM_TOL:.0e}"
        )
    if has_a0 and not math.isnan(id_a0) and id_a0 > IDENTITY_TOL:
        raise ValueError(
            f"alpha=0 identity check failed for {rec.run_dir.name}: "
            f"|L_alpha0 - nominal_bt_loss| = {id_a0:.3e} > tol {IDENTITY_TOL:.0e}"
        )
    if has_a1 and not math.isnan(id_a1) and id_a1 > IDENTITY_TOL:
        raise ValueError(
            f"alpha=1 identity check failed for {rec.run_dir.name}: "
            f"|L_alpha1 - biased_bt_loss| = {id_a1:.3e} > tol {IDENTITY_TOL:.0e}"
        )

    return CPSResult(
        rec=rec,
        rho_mis=rho_mis,
        rho_alg_abs=rho_alg,
        rho_alg_ratio=rho_alg_ratio,
        eta_kl=eta_kl,
        method=method_label(rho_alg_ratio, eta_kl),
        seed=seed,
        beta=beta,
        R=R_val,
        E_r_star=E_r_star,
        E_bias_attr=E_bias,
        entropy=H,
        top1=top1,
        theta_norm=theta_norm,
        KL_to_ref=kl_ref,
        alpha_values=alpha_values,
        L_alpha_exact=L_exact,
        L_alpha_offdiag=L_offdiag,
        id_residual_alpha0=float(id_a0),
        id_residual_alpha1=float(id_a1),
        antisym_residual=float(antisym_residual),
        range_min=float(range_min),
        range_max=float(range_max),
        has_alpha0=has_a0,
        has_alpha1=has_a1,
    )


@torch.no_grad()
def _R_under_pi(theta, psi, ref_logits) -> float:
    """E_{x, i, j ~ pi}[|h_ij|] -- the same quantity stored in sensitivity_R."""
    from .objective import sensitivity_R
    return float(sensitivity_R(theta, psi, ref_logits))


# --------------------------------------------------------------------------
# CSV writers
# --------------------------------------------------------------------------

def _se(vals: Sequence[float]) -> float:
    v = np.asarray(vals, dtype=np.float64)
    if v.size <= 1:
        return 0.0
    return float(v.std(ddof=1) / math.sqrt(v.size))


def write_alpha_metrics_csv(results: List[CPSResult], path: Path,
                            loss_field: str = "L_alpha_exact"):
    """Long-format: one row per (run, alpha)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tag", "run_dir", "seed", "rho_mis", "rho_alg_ratio", "rho_alg_abs",
            "eta_kl", "method_label", "alpha", loss_field,
            "L_alpha_0_exact", "L_alpha_1_exact",
            "shift_slope_exact", "endpoint_worst_loss_exact",
            "R", "E_r_star", "E_bias_attr", "entropy", "top1",
            "theta_norm", "KL_to_ref",
            "id_residual_alpha0", "id_residual_alpha1",
            "P_alpha_min_main_grid", "P_alpha_max_main_grid",
            "antisym_residual",
        ])
        for r in results:
            losses = (r.L_alpha_exact if loss_field == "L_alpha_exact"
                      else r.L_alpha_offdiag)
            for a, L in zip(r.alpha_values, losses):
                w.writerow([
                    r.rec.tag, str(r.rec.run_dir), r.seed,
                    f"{r.rho_mis:g}", f"{r.rho_alg_ratio:g}", f"{r.rho_alg_abs:g}",
                    f"{r.eta_kl:g}", r.method, f"{a:g}", f"{L:.10f}",
                    _opt_fmt(r.L_alpha_0_exact()), _opt_fmt(r.L_alpha_1_exact()),
                    _opt_fmt(r.shift_slope_exact()),
                    _opt_fmt(r.endpoint_worst_loss_exact()),
                    f"{r.R:.10f}", f"{r.E_r_star:.10f}", f"{r.E_bias_attr:.10f}",
                    f"{r.entropy:.10f}", f"{r.top1:.10f}",
                    f"{r.theta_norm:.10f}", f"{r.KL_to_ref:.10f}",
                    f"{r.id_residual_alpha0:.3e}", f"{r.id_residual_alpha1:.3e}",
                    f"{r.range_min:.6e}", f"{r.range_max:.6e}",
                    f"{r.antisym_residual:.3e}",
                ])


def _opt_fmt(x):
    if x is None:
        return ""
    return f"{x:.10f}"


def aggregate_summary(results: List[CPSResult], alpha_grid: Sequence[float]
                      ) -> Dict[Tuple[float, str], Dict[str, Tuple[float, float]]]:
    """Group by (rho_mis, method); compute mean/SE of summary metrics."""
    grouped: Dict[Tuple[float, str], List[CPSResult]] = defaultdict(list)
    for r in results:
        grouped[(r.rho_mis, r.method)].append(r)

    out: Dict[Tuple[float, str], Dict[str, Tuple[float, float]]] = {}
    for key, runs in grouped.items():
        rec: Dict[str, List[float]] = defaultdict(list)
        for r in runs:
            avg = float(np.mean(r.L_alpha_exact))
            worst = float(np.max(r.L_alpha_exact))
            l0 = r.L_alpha_0_exact()
            l1 = r.L_alpha_1_exact()
            slope = r.shift_slope_exact()
            epw = r.endpoint_worst_loss_exact()
            l_at_train = l1 if l1 is not None else worst
            rec["avg_shift_loss"].append(avg)
            rec["worst_shift_loss"].append(worst)
            rec["shift_gap"].append(worst - l_at_train)
            if l0 is not None: rec["L_alpha_0"].append(l0)
            if l1 is not None: rec["L_alpha_1"].append(l1)
            if 0.5 in alpha_grid:
                rec["L_alpha_0.5"].append(r.L_alpha_exact[r.alpha_values.index(0.5)])
            if slope is not None: rec["shift_slope"].append(slope)
            if epw is not None: rec["endpoint_worst_loss"].append(epw)
            rec["R"].append(r.R)
            rec["E_r_star"].append(r.E_r_star)
            rec["entropy"].append(r.entropy)
            rec["theta_norm"].append(r.theta_norm)
            rec["KL_to_ref"].append(r.KL_to_ref)
        out[key] = {k: (float(np.mean(v)), _se(v)) for k, v in rec.items()}
        out[key]["__n_seeds"] = (float(len(runs)), 0.0)
    return out


def write_summary_csv(summary, path: Path):
    cols = [
        "L_alpha_0", "L_alpha_0.5", "L_alpha_1",
        "avg_shift_loss", "worst_shift_loss", "shift_slope",
        "endpoint_worst_loss", "shift_gap",
        "R", "E_r_star", "entropy", "theta_norm", "KL_to_ref",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["rho_mis", "method_label", "n_seeds"]
        for c in cols:
            header += [f"{c}_mean", f"{c}_se"]
        w.writerow(header)
        for (rm, method), agg in sorted(summary.items()):
            n = int(agg["__n_seeds"][0])
            row = [f"{rm:g}", method, n]
            for c in cols:
                if c in agg:
                    m, s = agg[c]
                    row += [f"{m:.10f}", f"{s:.10f}"]
                else:
                    row += ["", ""]
            w.writerow(row)


def compute_paired_delta(results: List[CPSResult], baseline_method: str,
                         compare_methods: Sequence[str], alpha_grid: Sequence[float]
                         ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (per-row deltas, aggregate-row deltas).

    Pairs by (rho_mis, seed) on the seed intersection.
    """
    # index: results[(rho_mis, method, seed)] = CPSResult
    by_key: Dict[Tuple[float, str, int], CPSResult] = {}
    for r in results:
        by_key[(r.rho_mis, r.method, r.seed)] = r
    rho_set = sorted({r.rho_mis for r in results})
    rows: List[Dict[str, Any]] = []
    agg_rows: List[Dict[str, Any]] = []
    for rm in rho_set:
        seeds_baseline = {s for (rmk, m, s) in by_key if rmk == rm and m == baseline_method}
        for method in compare_methods:
            if method == baseline_method:
                continue
            seeds_other = {s for (rmk, m, s) in by_key if rmk == rm and m == method}
            paired_seeds = sorted(seeds_baseline & seeds_other)
            for alpha in alpha_grid:
                deltas_L = []
                deltas_R = []
                deltas_E = []
                deltas_H = []
                deltas_TN = []
                for s in paired_seeds:
                    r0 = by_key[(rm, baseline_method, s)]
                    r1 = by_key[(rm, method, s)]
                    if alpha not in r0.alpha_values or alpha not in r1.alpha_values:
                        continue
                    i0 = r0.alpha_values.index(alpha)
                    i1 = r1.alpha_values.index(alpha)
                    dL = r1.L_alpha_exact[i1] - r0.L_alpha_exact[i0]
                    dR = r1.R - r0.R
                    dE = r1.E_r_star - r0.E_r_star
                    dH = r1.entropy - r0.entropy
                    dTN = r1.theta_norm - r0.theta_norm
                    rows.append({
                        "rho_mis": rm, "method": method, "alpha": alpha, "seed": s,
                        "delta_L_alpha": dL, "delta_R": dR,
                        "delta_E_r_star": dE, "delta_entropy": dH,
                        "delta_theta_norm": dTN,
                    })
                    deltas_L.append(dL); deltas_R.append(dR)
                    deltas_E.append(dE); deltas_H.append(dH); deltas_TN.append(dTN)
                if deltas_L:
                    n = len(deltas_L)
                    agg_rows.append({
                        "rho_mis": rm, "method": method, "alpha": alpha,
                        "n_seeds": n,
                        "mean_delta_L_alpha": float(np.mean(deltas_L)),
                        "se_delta_L_alpha": _se(deltas_L),
                        "sign_count_delta_L_alpha_lt_0": int(sum(1 for x in deltas_L if x < 0)),
                        "mean_delta_R": float(np.mean(deltas_R)),
                        "se_delta_R": _se(deltas_R),
                        "mean_delta_E_r_star": float(np.mean(deltas_E)),
                        "se_delta_E_r_star": _se(deltas_E),
                        "mean_delta_entropy": float(np.mean(deltas_H)),
                        "se_delta_entropy": _se(deltas_H),
                        "mean_delta_theta_norm": float(np.mean(deltas_TN)),
                        "se_delta_theta_norm": _se(deltas_TN),
                    })
    return rows, agg_rows


def write_paired_delta_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("rho_mis,method,alpha,seed,delta_L_alpha,delta_R,delta_E_r_star,delta_entropy,delta_theta_norm\n")
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _group_by_rho_method(results: List[CPSResult]
                         ) -> Dict[Tuple[float, str], List[CPSResult]]:
    g: Dict[Tuple[float, str], List[CPSResult]] = defaultdict(list)
    for r in results:
        g[(r.rho_mis, r.method)].append(r)
    return g


def plot_loss_curves_by_rho(results: List[CPSResult], rhos: List[float],
                            methods: List[str], out_dir: Path):
    grouped = _group_by_rho_method(results)
    fig, axes = plt.subplots(1, len(rhos), figsize=(5 * len(rhos), 4.0), squeeze=False)
    colors = {"Nominal": "tab:blue", "Robust-0.025rho": "tab:orange"}
    for ax, rm in zip(axes[0], rhos):
        for method in methods:
            runs = grouped.get((rm, method), [])
            if not runs:
                continue
            alphas = runs[0].alpha_values
            arr = np.asarray([r.L_alpha_exact for r in runs])  # [n_seeds, n_alpha]
            mean = arr.mean(axis=0)
            se = arr.std(axis=0, ddof=1) / math.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
            color = colors.get(method, None)
            ax.plot(alphas, mean, marker="o", label=method, color=color)
            ax.fill_between(alphas, mean - se, mean + se, alpha=0.2, color=color)
        ax.axvline(1.0, color="k", linestyle="--", alpha=0.5)
        ax.text(1.0, ax.get_ylim()[1], " train α=1", va="top", ha="left", fontsize=8)
        ax.set_title(f"ρ_mis = {rm:g}")
        ax.set_xlabel("α (CPS shift)")
        ax.set_ylabel("L_α (exact, lower=better)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Controlled Preference Shift: L_α vs α (linear in α by construction)", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, out_dir, "cps_loss_curves_by_rho")


def plot_delta_loss_curves_by_rho(agg_rows: List[Dict[str, Any]],
                                  rhos: List[float], compare_method: str,
                                  out_dir: Path):
    fig, axes = plt.subplots(1, len(rhos), figsize=(5 * len(rhos), 4.0), squeeze=False)
    for ax, rm in zip(axes[0], rhos):
        sub = [a for a in agg_rows if a["rho_mis"] == rm and a["method"] == compare_method]
        sub.sort(key=lambda r: r["alpha"])
        if not sub:
            ax.set_title(f"ρ_mis = {rm:g} (no data)")
            continue
        alphas = [r["alpha"] for r in sub]
        m = np.asarray([r["mean_delta_L_alpha"] for r in sub])
        s = np.asarray([r["se_delta_L_alpha"] for r in sub])
        ax.plot(alphas, m, marker="o", color="tab:orange")
        ax.fill_between(alphas, m - s, m + s, alpha=0.2, color="tab:orange")
        ax.axhline(0.0, color="k", linestyle="--", alpha=0.5)
        ax.set_title(f"ρ_mis = {rm:g}")
        ax.set_xlabel("α")
        ax.set_ylabel(f"paired ΔL_α = {compare_method} − Nominal")
        ax.grid(alpha=0.3)
    fig.suptitle("Paired shift-loss delta under P_α (negative = robust better)", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, out_dir, "cps_delta_loss_curves_by_rho")


def plot_worst_avg_bar(summary, rhos: List[float], methods: List[str], out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    width = 0.35
    x = np.arange(len(rhos))
    colors = {"Nominal": "tab:blue", "Robust-0.025rho": "tab:orange"}
    for ax, key, title in [
        (axes[0], "worst_shift_loss", "worst_shift_loss ↓"),
        (axes[1], "avg_shift_loss",   "avg_shift_loss ↓"),
    ]:
        for i, method in enumerate(methods):
            means = []
            ses = []
            for rm in rhos:
                agg = summary.get((rm, method), {})
                if key in agg:
                    m, s = agg[key]
                else:
                    m, s = (math.nan, 0.0)
                means.append(m); ses.append(s)
            offset = (i - (len(methods) - 1) / 2.0) * width
            ax.bar(x + offset, means, width, yerr=ses, capsize=3,
                   label=method, color=colors.get(method, None))
        ax.set_xticks(x)
        ax.set_xticklabels([f"ρ_mis={rm:g}" for rm in rhos])
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9)
    fig.suptitle("CPS worst- and avg-shift losses", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, out_dir, "cps_worst_avg_bar")


def plot_heatmap_delta_L(agg_rows: List[Dict[str, Any]], rhos: List[float],
                         alphas: List[float], compare_method: str, out_dir: Path):
    M = np.full((len(rhos), len(alphas)), np.nan)
    for i, rm in enumerate(rhos):
        for j, a in enumerate(alphas):
            for r in agg_rows:
                if r["rho_mis"] == rm and r["method"] == compare_method and abs(r["alpha"] - a) < 1e-9:
                    M[i, j] = r["mean_delta_L_alpha"]
                    break
    finite = M[np.isfinite(M)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    if vmax == 0:
        vmax = 1e-6
    fig, ax = plt.subplots(figsize=(1.2 * len(alphas) + 2, 0.8 * len(rhos) + 2))
    im = ax.imshow(M, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_yticks(range(len(rhos)))
    ax.set_yticklabels([f"{rm:g}" for rm in rhos])
    ax.set_xlabel("α")
    ax.set_ylabel("ρ_mis")
    for i in range(len(rhos)):
        for j in range(len(alphas)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label=f"mean ΔL_α ({compare_method} − Nominal)")
    ax.set_title("Mean paired ΔL_α: robust minus nominal (negative = robust better)")
    fig.tight_layout()
    _save_fig(fig, out_dir, "cps_heatmap_delta_L")


def plot_pareto_true_vs_worst(results: List[CPSResult], rhos: List[float],
                              methods: List[str], out_dir: Path):
    grouped = _group_by_rho_method(results)
    fig, axes = plt.subplots(1, len(rhos), figsize=(5 * len(rhos), 4.0), squeeze=False)
    colors = {"Nominal": "tab:blue", "Robust-0.025rho": "tab:orange"}
    for ax, rm in zip(axes[0], rhos):
        for method in methods:
            runs = grouped.get((rm, method), [])
            if not runs:
                continue
            xs = [r.E_r_star for r in runs]
            ys = [max(r.L_alpha_exact) for r in runs]
            color = colors.get(method, None)
            ax.scatter(xs, ys, alpha=0.35, s=26, color=color)
            ax.scatter([float(np.mean(xs))], [float(np.mean(ys))],
                       s=140, color=color, edgecolor="black",
                       linewidths=1.4, label=f"{method} (mean)")
        ax.set_title(f"ρ_mis = {rm:g}")
        ax.set_xlabel("E[r*] ↑")
        ax.set_ylabel("worst_shift_loss ↓")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("CPS Pareto: utility vs worst-shift loss", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, out_dir, "cps_pareto_true_vs_worst")


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------

def _fmt_pm(m, s, digits=4):
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def render_alpha_metrics_md(results: List[CPSResult], path: Path,
                            max_rows: int = 30):
    """Compact markdown table of the first N (run, alpha) pairs."""
    lines = ["| run | seed | rho_mis | method | alpha | L_α_exact | L_α_offdiag |",
             "|---|---|---|---|---|---|---|"]
    n = 0
    for r in results:
        for a, le, lo in zip(r.alpha_values, r.L_alpha_exact, r.L_alpha_offdiag):
            lines.append(f"| {r.rec.run_dir.name} | {r.seed} | {r.rho_mis:g} | "
                         f"{r.method} | {a:g} | {le:.5f} | {lo:.5f} |")
            n += 1
            if n >= max_rows:
                lines.append(f"| ... ({len(results)} runs total — see CSV) | | | | | | |")
                path.write_text("\n".join(lines) + "\n")
                return
    path.write_text("\n".join(lines) + "\n")


def render_full_report(
    results: List[CPSResult],
    summary,
    paired_rows: List[Dict[str, Any]],
    paired_agg: List[Dict[str, Any]],
    rhos: List[float],
    methods: List[str],
    alpha_grid: List[float],
    path: Path,
    cli_invocation: str,
):
    md: List[str] = []
    md.append("# Controlled Preference Shift (CPS) report\n")
    md.append(f"- Loaded runs: {len(results)}")
    md.append(f"- ρ_mis levels: {rhos}")
    md.append(f"- Methods: {methods}")
    md.append(f"- α grid (main): {alpha_grid}")
    md.append(f"- CLI invocation: `{cli_invocation}`\n")

    md.append("## Loss-function provenance\n")
    md.append(
        "The CPS evaluator reports both `L_alpha_exact` "
        "(`exact_expectation_loss`) and `L_alpha_offdiag` (`offdiag_exact_loss`).\n\n"
        "- **Primary**: `L_alpha_exact`. The existing `nominal_bt_loss` and "
        "`biased_bt_loss` fields in `metrics.json` are computed via "
        "`exact_expectation_loss` (see `evaluate.py:192–193`), so the α=0 / α=1 "
        "identity checks are asserted on this loss only.\n"
        "- **Secondary**: `L_alpha_offdiag`. Matches the training objective "
        "family (`offdiag_exact`). Identity checks are *not* applied — there is "
        "no matching field in `metrics.json` to check against. Reported "
        "alongside in `cps_alpha_metrics_offdiag.csv` for cross-validation.\n"
    )

    md.append("## Affine-in-α structure\n")
    md.append(
        "P_α = p_star + α·ρ_mis·b enters the BT loss linearly through the "
        "`p · softplus(−βh) + (1−p) · softplus(βh)` term, so for any *fixed* "
        "policy θ, **L_α is exactly linear in α** (no nonlinear shift-response "
        "structure to mine). The summary tables therefore include explicit "
        "endpoint/slope columns: `L_alpha_0`, `L_alpha_1`, "
        "`shift_slope = L_alpha_1 − L_alpha_0`, "
        "`endpoint_worst_loss = max(L_alpha_0, L_alpha_1)`. "
        "Over a convex α range, `worst_shift_loss = endpoint_worst_loss`.\n"
    )

    md.append("## Field-name resolver\n")
    md.append(field_map_markdown_table())
    md.append("")

    md.append("## A. Identity-check residuals\n")
    md.append("| residual                                    | max over runs | tol |")
    md.append("|---|---|---|")
    a0 = max((r.id_residual_alpha0 for r in results if not math.isnan(r.id_residual_alpha0)), default=0.0)
    a1 = max((r.id_residual_alpha1 for r in results if not math.isnan(r.id_residual_alpha1)), default=0.0)
    sym = max((r.antisym_residual for r in results), default=0.0)
    rng_lo = min((r.range_min for r in results), default=0.0)
    rng_hi = max((r.range_max for r in results), default=1.0)
    md.append(f"| max \\|L_α(α=0) − nominal_bt_loss\\|        | {a0:.3e} | {IDENTITY_TOL:.0e} |")
    md.append(f"| max \\|L_α(α=1) − biased_bt_loss\\|         | {a1:.3e} | {IDENTITY_TOL:.0e} |")
    md.append(f"| max P_α antisymmetry residual              | {sym:.3e} | {ANTISYM_TOL:.0e} |")
    md.append(f"| min P_α over main α grid                   | {rng_lo:.6f} | ≥ 0 |")
    md.append(f"| max P_α over main α grid                   | {rng_hi:.6f} | ≤ 1 |")
    md.append("")

    md.append("## B. Per-(ρ_mis, method) summary\n")
    md.append("| ρ_mis | method | n | L_α(0) | L_α(1) | slope | avg_shift | worst_shift | shift_gap | R | E[r*] | entropy | KL_to_ref |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (rm, method), agg in sorted(summary.items()):
        n = int(agg["__n_seeds"][0])
        def g(k):
            if k in agg:
                return _fmt_pm(*agg[k])
            return "—"
        md.append("| " + " | ".join([
            f"{rm:g}", method, str(n),
            g("L_alpha_0"), g("L_alpha_1"), g("shift_slope"),
            g("avg_shift_loss"), g("worst_shift_loss"), g("shift_gap"),
            g("R"), g("E_r_star"), g("entropy"), g("KL_to_ref"),
        ]) + " |")
    md.append("")

    md.append("## C. Paired Δ summary (Robust − Nominal)\n")
    md.append("| ρ_mis | α | n | mean ΔL_α | SE | sign(<0) | mean ΔR | mean ΔE[r*] | mean Δentropy |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(paired_agg, key=lambda x: (x["rho_mis"], x["alpha"])):
        if r["method"] != "Robust-0.025rho":
            continue
        md.append("| " + " | ".join([
            f"{r['rho_mis']:g}", f"{r['alpha']:g}", str(r["n_seeds"]),
            f"{r['mean_delta_L_alpha']:+.4f}", f"{r['se_delta_L_alpha']:.4f}",
            f"{r['sign_count_delta_L_alpha_lt_0']}/{r['n_seeds']}",
            f"{r['mean_delta_R']:+.4f}", f"{r['mean_delta_E_r_star']:+.4f}",
            f"{r['mean_delta_entropy']:+.4f}",
        ]) + " |")
    md.append("")

    # Verdict per spec
    md.append("## D. Verdict\n")
    md.extend(_render_verdict(summary, paired_agg, rhos))
    md.append("")

    md.append("## E. Plots\n")
    for name in ["cps_loss_curves_by_rho", "cps_delta_loss_curves_by_rho",
                 "cps_worst_avg_bar", "cps_heatmap_delta_L",
                 "cps_pareto_true_vs_worst"]:
        md.append(f"- `{name}.png` / `{name}.pdf`")
    md.append("")
    path.write_text("\n".join(md))


def _render_verdict(summary, paired_agg, rhos):
    """PASS if Robust reduces mean worst_shift_loss and mean R while E[r*] drop ≤ 1e-3."""
    lines = []
    overall_pass = True
    overall_partial = False
    for rm in rhos:
        n_aggs = [r for r in paired_agg if r["rho_mis"] == rm and r["method"] == "Robust-0.025rho"]
        if not n_aggs:
            lines.append(f"- ρ_mis={rm:g}: PARTIAL — no Robust runs paired with Nominal.")
            overall_partial = True
            continue
        # check seed-level sign counts on key metrics
        # use Robust summary - Nominal summary on aggregated stats
        sN = summary.get((rm, "Nominal"), {})
        sR = summary.get((rm, "Robust-0.025rho"), {})
        if not sN or not sR:
            lines.append(f"- ρ_mis={rm:g}: PARTIAL — Nominal or Robust missing.")
            overall_partial = True
            continue
        d_worst = sR["worst_shift_loss"][0] - sN["worst_shift_loss"][0]
        d_R = sR["R"][0] - sN["R"][0]
        d_E = sR["E_r_star"][0] - sN["E_r_star"][0]
        cond_worst = d_worst < 0
        cond_R = d_R < 0
        cond_util = d_E >= -1e-3
        status = "PASS" if (cond_worst and cond_R and cond_util) else (
            "PARTIAL" if (cond_worst or cond_R) else "FAIL"
        )
        if status != "PASS":
            overall_pass = False
            if status == "PARTIAL":
                overall_partial = True
        lines.append(
            f"- ρ_mis={rm:g}: **{status}** — "
            f"Δworst_shift_loss={d_worst:+.4f}, ΔR={d_R:+.4f}, ΔE[r*]={d_E:+.4f}"
        )
    overall = "PASS" if overall_pass else ("PARTIAL" if overall_partial else "FAIL")
    lines.insert(0, f"**Overall CPS verdict (Robust-0.025ρ vs Nominal): {overall}**\n")
    return lines


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Controlled Preference Shift evaluator (post-hoc; no retraining)",
    )
    p.add_argument("--tag", type=str, default="cps",
                   help="Tag string recorded in CSV `tag` column (informational).")
    p.add_argument("--results-dir", action="append", default=None,
                   help="Results directory to scan for runs. May be repeated; "
                        "alias for --include-existing-results.")
    p.add_argument("--include-existing-results", action="append", default=[],
                   help="Additional results directories to merge in (repeatable).")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output directory for CSVs, plots, and report.md.")
    p.add_argument("--alpha-grid", type=str,
                   default=",".join(f"{a:g}" for a in DEFAULT_ALPHA_GRID))
    p.add_argument("--alpha-grid-stress", type=str, default="",
                   help="Optional stress alpha grid (CSV). Emitted as a separate "
                        "diagnostic CSV; clipping diagnostics allowed here.")
    p.add_argument("--methods", type=str,
                   default="Nominal,Robust-0.025rho",
                   help="Comma-separated method labels to include in plots/summary.")
    p.add_argument("--rho-mis", type=str, default="",
                   help="Optional comma-separated ρ_mis filter (e.g. '0.05,0.075,0.1').")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Override cache_dir from each run's config.json (useful "
                        "when results were produced on a different host).")
    p.add_argument("--no-enforce-range", action="store_true",
                   help="Disable the [0,1] range check on the main grid.")
    p.add_argument("--smoke", action="store_true",
                   help="One run (first found), two alphas (0.0 and 1.0). "
                        "Asserts CSV+plot are produced.")
    return p


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    cli_invocation = "python -m finite_response_sail._cps_shift_eval " + " ".join(sys.argv[1:])

    results_dirs: List[Path] = []
    if args.results_dir:
        results_dirs.extend(Path(p) for p in args.results_dir)
    if args.include_existing_results:
        results_dirs.extend(Path(p) for p in args.include_existing_results)
    if not results_dirs:
        raise SystemExit("at least one of --results-dir / --include-existing-results required")

    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    if args.smoke:
        alpha_grid = [0.0, 1.0]

    rho_filter = None
    if args.rho_mis:
        rho_filter = {float(x) for x in args.rho_mis.split(",")}

    method_filter = [m.strip() for m in args.methods.split(",") if m.strip()]

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(results_dirs)
    if rho_filter is not None:
        runs = [r for r in runs if float(r.config["rho_mis"]) in rho_filter]
    if method_filter:
        runs = [r for r in runs if method_label(
            float(r.config.get("rho_alg_ratio", 0.0)),
            float(r.config.get("eta_kl", 0.0) or 0.0),
        ) in method_filter]

    if args.smoke:
        runs = runs[:1]
    if not runs:
        raise SystemExit("[cps] no runs to evaluate after filtering")

    print(f"[cps] evaluating {len(runs)} runs over alpha grid {alpha_grid}")
    data_cache: Dict[Tuple, Tuple] = {}
    results: List[CPSResult] = []
    for rec in runs:
        try:
            r = evaluate_run(
                rec, alpha_grid, data_cache,
                cache_dir_override=args.cache_dir,
                enforce_range=not args.no_enforce_range,
            )
            results.append(r)
        except ValueError as e:
            raise SystemExit(f"[cps] FAIL: {e}")

    # Optional stress grid
    stress_results: List[CPSResult] = []
    if args.alpha_grid_stress and not args.smoke:
        stress = _parse_alpha_grid(args.alpha_grid_stress)
        for rec in runs:
            try:
                stress_results.append(evaluate_run(
                    rec, stress, data_cache,
                    cache_dir_override=args.cache_dir,
                    enforce_range=False,
                ))
            except ValueError as e:
                print(f"[cps] WARN stress-grid: {e}", file=sys.stderr)

    rhos = sorted({r.rho_mis for r in results})
    methods = method_filter if method_filter else sorted({r.method for r in results})

    # Write CSVs
    write_alpha_metrics_csv(results, out_dir / "cps_alpha_metrics.csv",
                            loss_field="L_alpha_exact")
    write_alpha_metrics_csv(results, out_dir / "cps_alpha_metrics_offdiag.csv",
                            loss_field="L_alpha_offdiag")

    summary = aggregate_summary(results, alpha_grid)
    write_summary_csv(summary, out_dir / "cps_summary_by_rho_method.csv")

    paired_rows, paired_agg = compute_paired_delta(
        results, baseline_method="Nominal",
        compare_methods=[m for m in methods if m != "Nominal"],
        alpha_grid=alpha_grid,
    )
    write_paired_delta_csv(paired_rows, out_dir / "cps_paired_delta.csv")
    # Aggregate sheet
    if paired_agg:
        with open(out_dir / "cps_paired_delta_aggregate.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(paired_agg[0].keys()))
            w.writeheader()
            for r in paired_agg:
                w.writerow(r)

    if stress_results:
        write_alpha_metrics_csv(stress_results,
                                out_dir / "cps_alpha_metrics_stress.csv",
                                loss_field="L_alpha_exact")

    # Plots
    if not args.smoke:
        plot_loss_curves_by_rho(results, rhos, methods, out_dir)
        plot_pareto_true_vs_worst(results, rhos, methods, out_dir)
        plot_worst_avg_bar(summary, rhos, methods, out_dir)
        if paired_agg:
            plot_delta_loss_curves_by_rho(paired_agg, rhos, "Robust-0.025rho", out_dir)
            plot_heatmap_delta_L(paired_agg, rhos, alpha_grid, "Robust-0.025rho", out_dir)
    else:
        # smoke mode: still emit one PNG so the smoke test can check it.
        plot_loss_curves_by_rho(results, rhos, methods, out_dir)

    # Markdown
    render_alpha_metrics_md(results, out_dir / "cps_alpha_metrics.md")
    render_full_report(
        results, summary, paired_rows, paired_agg,
        rhos, methods, alpha_grid,
        out_dir / "cps_report.md",
        cli_invocation,
    )
    print(f"[cps] wrote {out_dir / 'cps_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
