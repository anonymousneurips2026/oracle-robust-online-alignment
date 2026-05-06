"""KL-to-reference smoothing baseline report (Experiment 3).

Loads:
    - confirm runs at rho_mis=0.075 (Nominal & Robust-0.025rho)
    - new KL-smoothing runs at rho_mis=0.075, rho_alg=0, varying eta_kl

For each KL run computes the same CPS metrics, then selects two baselines:
    KL-matchR        eta_kl whose mean val.R is closest to Robust-0.025rho.
    KL-matchEntropy  eta_kl whose mean val.entropy is closest.

Selection is done on `final.val` (held-out) when available; falls back to
`final.train` and marks selection_status=DIAGNOSTIC_VAL_MISSING.

If no eta_kl lands within tolerance of Robust on either dimension, the
report writes selection_status=MATCH_FAILED and emits a recommended
extension to the eta_kl grid (do NOT silently elect a far-away point).

Outputs:
    reports/kl_smoothing/kl_smoothing_table.csv
    reports/kl_smoothing/kl_selected_comparison.csv
    reports/kl_smoothing/kl_paired_delta.csv
    reports/kl_smoothing/kl_smoothing_report.md
    reports/kl_smoothing/{kl_eta_curves, kl_R_vs_worst, kl_true_vs_worst,
                          kl_alpha_curves, kl_selected_bar}.{png,pdf}
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import _cps_shift_eval as cps
from ._field_resolver import field_map_markdown_table

DEFAULT_RHO_MIS = 0.075
ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MATCH_TOL_FRAC = 0.10   # |Δ| ≤ 0.10 · |Robust − Nominal| counts as a match


def _se(vals):
    v = np.asarray(vals, dtype=np.float64)
    if v.size <= 1:
        return 0.0
    return float(v.std(ddof=1) / math.sqrt(v.size))


def _save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def collect_runs(confirm_dir: Path, kl_dir: Optional[Path], rho_mis: float,
                 sweep_dir: Optional[Path] = None,
                 target_robust_ratio: float = 0.025
                 ) -> Tuple[List[cps.RunRecord], List[cps.RunRecord], List[cps.RunRecord], float]:
    """Returns (nominal, robust, kl_runs, resolved_ratio).

    `target_robust_ratio` selects which robust-ratio runs serve as the comparison
    target for KL-matchR / KL-matchEntropy. Defaults to 0.025 (the original
    confirm-experiment robust setting). When the requested ratio is not found in
    confirm_dir, we additionally scan `sweep_dir`. If still not found, we fall
    back to 0.025 with a warning.
    """
    dirs = [confirm_dir]
    if sweep_dir is not None and sweep_dir.exists():
        dirs.append(sweep_dir)
    if kl_dir is not None and kl_dir.exists():
        dirs.append(kl_dir)
    all_runs = cps.discover_runs(dirs)
    nominal: List[cps.RunRecord] = []
    robust: List[cps.RunRecord] = []
    kl: List[cps.RunRecord] = []
    resolved_ratio = float(target_robust_ratio)
    for r in all_runs:
        cfg = r.config
        if abs(float(cfg["rho_mis"]) - rho_mis) > 1e-9:
            continue
        eta = float(cfg.get("eta_kl", 0.0) or 0.0)
        ratio = float(cfg.get("rho_alg_ratio", 0.0))
        if eta > 0.0:
            kl.append(r)
        elif abs(ratio) < 1e-12:
            nominal.append(r)
        elif abs(ratio - resolved_ratio) < 1e-9:
            robust.append(r)
    if not robust and abs(resolved_ratio - 0.025) > 1e-9:
        # fall back to 0.025
        print(f"[kl] WARN: no runs at rho_alg_ratio={resolved_ratio}; "
              f"falling back to 0.025", file=sys.stderr)
        for r in all_runs:
            cfg = r.config
            if abs(float(cfg["rho_mis"]) - rho_mis) > 1e-9:
                continue
            if float(cfg.get("eta_kl", 0.0) or 0.0) > 0.0:
                continue
            if abs(float(cfg.get("rho_alg_ratio", 0.0)) - 0.025) < 1e-9:
                robust.append(r)
        resolved_ratio = 0.025
    return nominal, robust, kl, resolved_ratio


# --------------------------------------------------------------------------
# Aggregation per group
# --------------------------------------------------------------------------

@dataclass
class GroupAgg:
    label: str
    eta_kl: float           # 0.0 for Nominal/Robust
    rho_alg_ratio: float    # 0.0 for Nominal/KL
    n_seeds: int
    seeds: List[int]
    val_R: Tuple[float, float]
    val_entropy: Tuple[float, float]
    val_KL_to_ref: Tuple[float, float]
    R: Tuple[float, float]
    E_r_star: Tuple[float, float]
    entropy: Tuple[float, float]
    theta_norm: Tuple[float, float]
    KL_to_ref: Tuple[float, float]
    L_SAIL_Pstar: Tuple[float, float]
    wcl_rho_mis: Tuple[float, float]
    avg_shift_loss: Tuple[float, float]
    worst_shift_loss: Tuple[float, float]
    shift_gap: Tuple[float, float]
    val_source: str   # "val" or "train" or "missing"


def _val_or_train_metric(rec: cps.RunRecord, key: str) -> Tuple[Optional[float], str]:
    if rec.metrics_val and key in rec.metrics_val:
        return float(rec.metrics_val[key]), "val"
    train = rec.config.get("__train__", None)  # not used; for clarity
    final_train = None
    if hasattr(rec, "metrics_test"):
        # We didn't load final.train into RunRecord; re-read from disk if needed.
        pass
    # Fallback: try reading metrics.json again to get final.train.
    try:
        import json
        d = json.loads((rec.run_dir / "metrics.json").read_text())
        final = d.get("final", {})
        if "val" in final and key in final["val"]:
            return float(final["val"][key]), "val"
        if "train" in final and key in final["train"]:
            return float(final["train"][key]), "train"
    except Exception:
        pass
    return None, "missing"


def aggregate_group(label: str, recs: List[cps.RunRecord],
                    cps_results: List[cps.CPSResult]) -> Optional[GroupAgg]:
    if not recs:
        return None
    seed_to_cps: Dict[int, cps.CPSResult] = {r.seed: r for r in cps_results}
    seeds_recs = [r for r in recs if int(r.config["seed"]) in seed_to_cps]
    if not seeds_recs:
        return None
    val_R: List[float] = []
    val_H: List[float] = []
    val_KL: List[float] = []
    val_sources: List[str] = []
    R_list: List[float] = []
    E_list: List[float] = []
    H_list: List[float] = []
    TN_list: List[float] = []
    KL_list: List[float] = []
    Lsail_list: List[float] = []
    wcl_list: List[float] = []
    avg_list: List[float] = []
    worst_list: List[float] = []
    gap_list: List[float] = []
    seeds_sorted: List[int] = []
    for rec in seeds_recs:
        s = int(rec.config["seed"])
        cr = seed_to_cps[s]
        seeds_sorted.append(s)
        vR, src = _val_or_train_metric(rec, "sensitivity_R")
        vH, _ = _val_or_train_metric(rec, "entropy")
        vKL, _ = _val_or_train_metric(rec, "KL_to_ref")
        if vR is not None: val_R.append(vR)
        if vH is not None: val_H.append(vH)
        if vKL is not None: val_KL.append(vKL)
        val_sources.append(src)
        R_list.append(cr.R)
        E_list.append(cr.E_r_star)
        H_list.append(cr.entropy)
        TN_list.append(cr.theta_norm)
        KL_list.append(cr.KL_to_ref)
        Lsail_list.append(float(rec.metrics_test.get("nominal_bt_loss", math.nan)))
        wcl_list.append(float(rec.metrics_test.get("wcl_true_rho_mis", math.nan)))
        avg_list.append(float(np.mean(cr.L_alpha_exact)))
        worst_list.append(float(np.max(cr.L_alpha_exact)))
        l1 = cr.L_alpha_1_exact()
        gap_list.append(max(cr.L_alpha_exact) - (l1 if l1 is not None else max(cr.L_alpha_exact)))
    val_source = "val" if all(s == "val" for s in val_sources) else (
        "train" if "train" in val_sources else "missing"
    )
    eta = float(seeds_recs[0].config.get("eta_kl", 0.0) or 0.0)
    ratio = float(seeds_recs[0].config.get("rho_alg_ratio", 0.0))
    def pair(xs):
        if not xs:
            return (math.nan, 0.0)
        return (float(np.mean(xs)), _se(xs))
    return GroupAgg(
        label=label,
        eta_kl=eta,
        rho_alg_ratio=ratio,
        n_seeds=len(seeds_sorted),
        seeds=sorted(seeds_sorted),
        val_R=pair(val_R),
        val_entropy=pair(val_H),
        val_KL_to_ref=pair(val_KL),
        R=pair(R_list),
        E_r_star=pair(E_list),
        entropy=pair(H_list),
        theta_norm=pair(TN_list),
        KL_to_ref=pair(KL_list),
        L_SAIL_Pstar=pair(Lsail_list),
        wcl_rho_mis=pair(wcl_list),
        avg_shift_loss=pair(avg_list),
        worst_shift_loss=pair(worst_list),
        shift_gap=pair(gap_list),
        val_source=val_source,
    )


# --------------------------------------------------------------------------
# Match selection
# --------------------------------------------------------------------------

@dataclass
class MatchResult:
    target_metric: str        # "R" or "entropy"
    target_value: float
    chosen_eta: Optional[float]
    chosen_value: float
    abs_delta: float
    tolerance: float
    status: str               # "OK" | "MATCH_FAILED" | "DIAGNOSTIC_VAL_MISSING"
    recommendation: List[float]
    val_source: str


def select_match(
    kl_groups: List[GroupAgg],
    target_value: float,
    target_metric: str,
    nominal_value: float,
) -> MatchResult:
    """target_metric in {'R', 'entropy'}. Uses val-side aggregates when available."""
    tol_basis = abs(target_value - nominal_value)
    tol = max(MATCH_TOL_FRAC * tol_basis, 1e-6)
    candidates: List[Tuple[float, float, str]] = []
    for g in kl_groups:
        v_pair = g.val_R if target_metric == "R" else g.val_entropy
        if not (math.isnan(v_pair[0]) if v_pair[0] is not None else True):
            candidates.append((g.eta_kl, v_pair[0], g.val_source))
    if not candidates:
        return MatchResult(
            target_metric=target_metric, target_value=target_value,
            chosen_eta=None, chosen_value=math.nan, abs_delta=math.nan,
            tolerance=tol, status="DIAGNOSTIC_VAL_MISSING",
            recommendation=[], val_source="missing",
        )
    candidates.sort(key=lambda c: abs(c[1] - target_value))
    eta, val, src = candidates[0]
    delta = abs(val - target_value)
    diagnostic = any(s != "val" for _, _, s in candidates)
    if delta <= tol:
        status = "OK" if not diagnostic else "DIAGNOSTIC_VAL_MISSING"
        return MatchResult(
            target_metric=target_metric, target_value=target_value,
            chosen_eta=eta, chosen_value=val, abs_delta=delta,
            tolerance=tol, status=status, recommendation=[], val_source=src,
        )
    # No match. Recommend extending the grid.
    sorted_etas = sorted(c[0] for c in candidates)
    sorted_vals = {c[0]: c[1] for c in candidates}
    # Bracket: which side of target_value does the grid sit on?
    if all(sorted_vals[e] > target_value for e in sorted_etas):
        # All eta produce too-high val -- need MORE smoothing? Depends on direction.
        rec = [sorted_etas[-1] * 3.0, sorted_etas[-1] * 10.0]
    elif all(sorted_vals[e] < target_value for e in sorted_etas):
        rec = [sorted_etas[0] / 3.0, sorted_etas[0] / 10.0]
    else:
        # Bracketing case: target lies between two grid points. Recommend
        # geometric-mean interpolated points.
        rec = []
        for i in range(len(sorted_etas) - 1):
            e_lo, e_hi = sorted_etas[i], sorted_etas[i + 1]
            v_lo, v_hi = sorted_vals[e_lo], sorted_vals[e_hi]
            if (v_lo - target_value) * (v_hi - target_value) <= 0:
                # target between v_lo and v_hi -- recommend two intermediate etas
                gm = math.sqrt(e_lo * e_hi)
                rec.extend([gm * 0.7, gm * 1.3])
                break
    return MatchResult(
        target_metric=target_metric, target_value=target_value,
        chosen_eta=eta, chosen_value=val, abs_delta=delta,
        tolerance=tol, status="MATCH_FAILED", recommendation=rec, val_source=src,
    )


# --------------------------------------------------------------------------
# CSV / report
# --------------------------------------------------------------------------

ROW_COLUMNS = [
    "n_seeds",
    "val_R_mean", "val_R_se",
    "val_entropy_mean", "val_entropy_se",
    "val_KL_to_ref_mean", "val_KL_to_ref_se",
    "R_test_mean", "R_test_se",
    "E_r_star_mean", "E_r_star_se",
    "entropy_test_mean", "entropy_test_se",
    "theta_norm_mean", "theta_norm_se",
    "KL_to_ref_mean", "KL_to_ref_se",
    "L_SAIL_Pstar_mean", "L_SAIL_Pstar_se",
    "wcl_rho_mis_mean", "wcl_rho_mis_se",
    "avg_shift_loss_mean", "avg_shift_loss_se",
    "worst_shift_loss_mean", "worst_shift_loss_se",
    "shift_gap_mean", "shift_gap_se",
]


def _row_for(g: GroupAgg) -> Dict[str, Any]:
    out = {
        "label": g.label,
        "eta_kl": g.eta_kl,
        "rho_alg_ratio": g.rho_alg_ratio,
        "val_source": g.val_source,
        "n_seeds": g.n_seeds,
    }
    for src, dst_prefix in [
        (g.val_R, "val_R"), (g.val_entropy, "val_entropy"),
        (g.val_KL_to_ref, "val_KL_to_ref"),
        (g.R, "R_test"), (g.E_r_star, "E_r_star"),
        (g.entropy, "entropy_test"), (g.theta_norm, "theta_norm"),
        (g.KL_to_ref, "KL_to_ref"),
        (g.L_SAIL_Pstar, "L_SAIL_Pstar"), (g.wcl_rho_mis, "wcl_rho_mis"),
        (g.avg_shift_loss, "avg_shift_loss"),
        (g.worst_shift_loss, "worst_shift_loss"),
        (g.shift_gap, "shift_gap"),
    ]:
        out[f"{dst_prefix}_mean"] = src[0]
        out[f"{dst_prefix}_se"] = src[1]
    return out


def write_table_csv(groups: List[GroupAgg], path: Path):
    if not groups:
        path.write_text("(no groups)\n")
        return
    cols = ["label", "eta_kl", "rho_alg_ratio", "val_source"] + ROW_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for g in groups:
            w.writerow(_row_for(g))


def write_selected_comparison_csv(
    selected: List[Tuple[GroupAgg, str]], path: Path,
):
    cols = ["label", "selection_status"] + ["eta_kl", "rho_alg_ratio", "val_source"] + ROW_COLUMNS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for g, status in selected:
            row = _row_for(g)
            row["selection_status"] = status
            w.writerow(row)


def compute_paired_delta_robust_vs_kl(
    robust_results: List[cps.CPSResult],
    kl_results: List[cps.CPSResult],
    kl_eta: float,
) -> List[Dict[str, Any]]:
    """Per-seed (Robust − KL) deltas across the seed intersection."""
    by_seed_robust = {r.seed: r for r in robust_results}
    by_seed_kl = {r.seed: r for r in kl_results if abs(r.eta_kl - kl_eta) < 1e-12}
    seeds = sorted(set(by_seed_robust.keys()) & set(by_seed_kl.keys()))
    out = []
    for s in seeds:
        rr = by_seed_robust[s]
        rk = by_seed_kl[s]
        out.append({
            "seed": s,
            "kl_eta": kl_eta,
            "delta_R": rr.R - rk.R,
            "delta_E_r_star": rr.E_r_star - rk.E_r_star,
            "delta_entropy": rr.entropy - rk.entropy,
            "delta_theta_norm": rr.theta_norm - rk.theta_norm,
            "delta_avg_shift_loss": float(np.mean(rr.L_alpha_exact) - np.mean(rk.L_alpha_exact)),
            "delta_worst_shift_loss": float(np.max(rr.L_alpha_exact) - np.max(rk.L_alpha_exact)),
            "delta_KL_to_ref": rr.KL_to_ref - rk.KL_to_ref,
        })
    return out


def write_paired_delta_csv(rows_R: List[Dict[str, Any]], rows_H: List[Dict[str, Any]],
                            path: Path, status_R: str, status_H: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if status_R != "OK" and status_H != "OK":
        path.write_text(
            "# paired delta NOT emitted (selection_status not OK)\n"
            f"# match-R status: {status_R}\n# match-H status: {status_H}\n"
        )
        return
    cols = ["selection", "seed", "kl_eta", "delta_R", "delta_E_r_star",
            "delta_entropy", "delta_theta_norm",
            "delta_avg_shift_loss", "delta_worst_shift_loss",
            "delta_KL_to_ref"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        if status_R == "OK":
            for r in rows_R:
                w.writerow({"selection": "match-R", **r})
        if status_H == "OK":
            for r in rows_H:
                w.writerow({"selection": "match-H", **r})


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_eta_curves(kl_groups: List[GroupAgg], nominal: GroupAgg,
                    robust: GroupAgg, out_dir: Path, robust_label: str = "Robust-0.025rho"):
    panels = [
        ("E_r_star",         "E[r*] ↑"),
        ("R",                "R ↓"),
        ("entropy",          "entropy"),
        ("theta_norm",       "||θ|| ↓"),
        ("worst_shift_loss", "worst_shift_loss ↓"),
        ("avg_shift_loss",   "avg_shift_loss ↓"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    etas = [g.eta_kl for g in kl_groups]
    for ax, (k, name) in zip(axes.flat, panels):
        means = [getattr(g, k)[0] for g in kl_groups]
        ses = [getattr(g, k)[1] for g in kl_groups]
        ax.errorbar(etas, means, yerr=ses, marker="o", capsize=3, label="KL-smoothed")
        if nominal:
            v = getattr(nominal, k)[0]
            ax.axhline(v, color="tab:blue", linestyle="--", alpha=0.7, label="Nominal")
        if robust:
            v = getattr(robust, k)[0]
            ax.axhline(v, color="tab:orange", linestyle="--", alpha=0.7, label=robust_label)
        ax.set_xscale("log")
        ax.set_xlabel("eta_kl (log scale)")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle("KL-smoothing baseline: KL coefficient sweep (rho_mis=0.075)", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out_dir, "kl_eta_curves")


def plot_R_vs_worst(kl_groups, nominal, robust, match_R: MatchResult, out_dir: Path,
                    robust_label: str = "Robust-0.025rho"):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    etas = sorted({g.eta_kl for g in kl_groups})
    for g in kl_groups:
        ax.scatter([g.R[0]], [g.worst_shift_loss[0]], s=60, color="tab:gray",
                   alpha=0.7)
        ax.annotate(f"η={g.eta_kl:g}", (g.R[0], g.worst_shift_loss[0]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    if nominal:
        ax.scatter([nominal.R[0]], [nominal.worst_shift_loss[0]], s=180,
                   color="tab:blue", edgecolor="black", linewidths=1.4,
                   label="Nominal", zorder=5)
    if robust:
        ax.scatter([robust.R[0]], [robust.worst_shift_loss[0]], s=180,
                   color="tab:orange", edgecolor="black", linewidths=1.4,
                   label=robust_label, zorder=5)
    if match_R.chosen_eta is not None:
        # find the chosen group
        for g in kl_groups:
            if abs(g.eta_kl - match_R.chosen_eta) < 1e-12:
                ax.scatter([g.R[0]], [g.worst_shift_loss[0]], s=220,
                           facecolor="none", edgecolor="red", linewidths=2.0,
                           label=f"KL-matchR (η={g.eta_kl:g}, status={match_R.status})",
                           zorder=6)
                break
    ax.set_xlabel("R (test) ↓")
    ax.set_ylabel("worst_shift_loss ↓")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title("KL smoothing: R vs worst-shift loss")
    fig.tight_layout()
    _save_fig(fig, out_dir, "kl_R_vs_worst")


def plot_true_vs_worst(kl_groups, nominal, robust, match_R: MatchResult, out_dir: Path,
                       robust_label: str = "Robust-0.025rho"):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for g in kl_groups:
        ax.scatter([g.E_r_star[0]], [g.worst_shift_loss[0]], s=60,
                   color="tab:gray", alpha=0.7)
        ax.annotate(f"η={g.eta_kl:g}", (g.E_r_star[0], g.worst_shift_loss[0]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    if nominal:
        ax.scatter([nominal.E_r_star[0]], [nominal.worst_shift_loss[0]], s=180,
                   color="tab:blue", edgecolor="black", linewidths=1.4,
                   label="Nominal", zorder=5)
    if robust:
        ax.scatter([robust.E_r_star[0]], [robust.worst_shift_loss[0]], s=180,
                   color="tab:orange", edgecolor="black", linewidths=1.4,
                   label=robust_label, zorder=5)
    if match_R.chosen_eta is not None:
        for g in kl_groups:
            if abs(g.eta_kl - match_R.chosen_eta) < 1e-12:
                ax.scatter([g.E_r_star[0]], [g.worst_shift_loss[0]], s=220,
                           facecolor="none", edgecolor="red", linewidths=2.0,
                           label=f"KL-matchR (status={match_R.status})", zorder=6)
                break
    ax.set_xlabel("E[r*] ↑")
    ax.set_ylabel("worst_shift_loss ↓")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title("KL smoothing: utility vs CPS worst-shift loss")
    fig.tight_layout()
    _save_fig(fig, out_dir, "kl_true_vs_worst")


def plot_alpha_curves(by_label_results: Dict[str, List[cps.CPSResult]],
                      out_dir: Path,
                      robust_label: str = "Robust-0.025rho"):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    color_map = {"Nominal": "tab:blue", robust_label: "tab:orange",
                 "KL-matchR": "tab:green", "KL-matchEntropy": "tab:purple"}
    for label, results in by_label_results.items():
        if not results:
            continue
        alphas = results[0].alpha_values
        arr = np.asarray([r.L_alpha_exact for r in results])
        m = arr.mean(axis=0)
        s = arr.std(axis=0, ddof=1) / math.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(m)
        c = color_map.get(label, None)
        ax.plot(alphas, m, marker="o", label=label, color=c)
        ax.fill_between(alphas, m - s, m + s, alpha=0.18, color=c)
    ax.axvline(1.0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("α (CPS shift)")
    ax.set_ylabel("L_α ↓")
    ax.set_title("KL smoothing comparison: L_α curves at rho_mis=0.075")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, "kl_alpha_curves")


def plot_selected_bar(selected: List[Tuple[GroupAgg, str]], out_dir: Path):
    if not selected:
        return
    panels = [("E_r_star", "E[r*]"),
              ("R", "R"),
              ("entropy", "entropy"),
              ("theta_norm", "||θ||"),
              ("avg_shift_loss", "avg_shift"),
              ("worst_shift_loss", "worst_shift")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    labels = [g.label for g, _ in selected]
    x = np.arange(len(labels))
    for ax, (k, name) in zip(axes.flat, panels):
        means = [getattr(g, k)[0] for g, _ in selected]
        ses = [getattr(g, k)[1] for g, _ in selected]
        bars = ax.bar(x, means, yerr=ses, capsize=3,
                      color=["tab:blue", "tab:orange", "tab:green", "tab:purple"][:len(labels)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_title(name)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("KL smoothing: selected method comparison", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out_dir, "kl_selected_bar")


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------

def render_report_md(
    nominal: Optional[GroupAgg],
    robust: Optional[GroupAgg],
    kl_groups: List[GroupAgg],
    match_R: MatchResult,
    match_H: MatchResult,
    selected: List[Tuple[GroupAgg, str]],
    paired_R: List[Dict[str, Any]],
    paired_H: List[Dict[str, Any]],
    rho_mis: float,
    path: Path,
    cli: str,
):
    md: List[str] = []
    md.append("# KL-to-reference smoothing baseline report\n")
    md.append(f"- ρ_mis = {rho_mis:g}, ρ_alg = 0 (KL-smoothed runs)")
    md.append(f"- KL eta grid present: {[g.eta_kl for g in kl_groups]}")
    md.append(f"- Nominal seeds: {nominal.n_seeds if nominal else 0}; "
              f"Robust seeds: {robust.n_seeds if robust else 0}")
    md.append(f"- CLI: `{cli}`\n")

    md.append("## KL formula\n")
    md.append(
        "    pi_theta = softmax(ref_logits + theta @ psi)\n"
        "    pi_ref   = softmax(ref_logits)\n"
        "    KL_to_ref = E_x sum_i pi_theta_i (log pi_theta_i − log pi_ref_i)\n\n"
        "Training adds `eta_kl * KL_to_ref.mean()` to the BT loss. "
        "No L2 fallback, no entropy-bonus fallback was used.\n"
    )

    md.append("## Selection discipline\n")
    md.append(
        f"`KL-matchR` and `KL-matchEntropy` are picked by minimizing "
        f"`|mean R(eta_kl) − mean R(Robust)|` (and analogously for entropy) on the "
        f"**held-out validation split** (`final.val.sensitivity_R`, "
        f"`final.val.entropy`). Tolerance: "
        f"`tol = max({MATCH_TOL_FRAC} · |Robust − Nominal|, 1e-6)`. "
        f"If no eta lies within tolerance, selection_status = `MATCH_FAILED` and "
        f"the report refuses to claim a comparison.\n"
    )

    md.append("## Match-R selection\n")
    md.append(_render_match(match_R))
    md.append("\n## Match-Entropy selection\n")
    md.append(_render_match(match_H))
    md.append("")

    md.append("## Field-name resolver\n")
    md.append(field_map_markdown_table())
    md.append("")

    md.append("## A. Selected method comparison\n")
    md.append("| label | status | n | E[r*] | R | entropy | ||θ|| | avg_shift | worst_shift |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for g, status in selected:
        md.append("| " + " | ".join([
            g.label, status, str(g.n_seeds),
            f"{g.E_r_star[0]:.4f}±{g.E_r_star[1]:.4f}",
            f"{g.R[0]:.4f}±{g.R[1]:.4f}",
            f"{g.entropy[0]:.4f}±{g.entropy[1]:.4f}",
            f"{g.theta_norm[0]:.3f}±{g.theta_norm[1]:.3f}",
            f"{g.avg_shift_loss[0]:.5f}±{g.avg_shift_loss[1]:.5f}",
            f"{g.worst_shift_loss[0]:.5f}±{g.worst_shift_loss[1]:.5f}",
        ]) + " |")
    md.append("")

    md.append("## B. Paired Δ (Robust − KL)\n")
    if match_R.status == "OK" and paired_R:
        md.append("**Robust − KL-matchR**, per-seed deltas (n="
                  f"{len(paired_R)} seed intersection):\n")
        md.append(_render_paired_table(paired_R))
    else:
        md.append(f"`Robust − KL-matchR` not emitted (status: {match_R.status}).\n")
    if match_H.status == "OK" and paired_H:
        md.append("\n**Robust − KL-matchEntropy**, per-seed deltas (n="
                  f"{len(paired_H)}):\n")
        md.append(_render_paired_table(paired_H))
    else:
        md.append(f"\n`Robust − KL-matchEntropy` not emitted (status: {match_H.status}).\n")

    md.append("\n## C. Verdicts (4 spec questions)\n")
    md.extend(_render_verdicts(nominal, robust, match_R, match_H, paired_R, paired_H))
    md.append("")

    md.append("## D. Plots\n")
    for name in ["kl_eta_curves", "kl_R_vs_worst", "kl_true_vs_worst",
                 "kl_alpha_curves", "kl_selected_bar"]:
        md.append(f"- `{name}.png` / `{name}.pdf`")
    md.append("")
    path.write_text("\n".join(md))


def _render_match(m: MatchResult) -> str:
    out = [
        f"- target metric: **{m.target_metric}**",
        f"- target value (Robust): {m.target_value:.5f}",
        f"- chosen eta_kl: {m.chosen_eta!r}",
        f"- chosen value: {m.chosen_value:.5f}",
        f"- |Δ|: {m.abs_delta:.5f};   tolerance: {m.tolerance:.5f}",
        f"- selection source: {m.val_source}",
        f"- **status: {m.status}**",
    ]
    if m.status == "MATCH_FAILED" and m.recommendation:
        out.append(f"- recommended next eta_kl values to add: {m.recommendation}")
    return "\n".join(out)


def _render_paired_table(rows: List[Dict[str, Any]]) -> str:
    keys = ["delta_R", "delta_E_r_star", "delta_entropy",
            "delta_theta_norm", "delta_avg_shift_loss",
            "delta_worst_shift_loss", "delta_KL_to_ref"]
    out = ["| metric | mean | SE | sign(<0) |", "|---|---|---|---|"]
    for k in keys:
        vals = [r[k] for r in rows]
        m = float(np.mean(vals))
        s = _se(vals)
        sgn = sum(1 for v in vals if v < 0)
        out.append(f"| {k} | {m:+.4f} | {s:.4f} | {sgn}/{len(vals)} |")
    return "\n".join(out)


def _render_verdicts(nominal, robust, match_R, match_H, paired_R, paired_H):
    out = []
    # 1. Did KL-matchR succeed at matching Robust's R?
    if match_R.status == "OK":
        out.append(f"1. **KL-matchR successfully matched Robust's R?** YES "
                   f"(|Δ|={match_R.abs_delta:.5f} ≤ tol={match_R.tolerance:.5f}).")
    else:
        out.append(f"1. **KL-matchR successfully matched Robust's R?** NO "
                   f"(status: {match_R.status}). Comparison below is suspended.")
    # 2. At matched R, does Robust beat KL on worst_shift_loss?
    if match_R.status == "OK" and paired_R:
        d_worst = float(np.mean([r["delta_worst_shift_loss"] for r in paired_R]))
        # delta = Robust - KL, so negative -> Robust BETTER
        verdict = "Robust BETTER" if d_worst < -1e-5 else (
            "KL BETTER" if d_worst > 1e-5 else "≈ equal"
        )
        out.append(f"2. **At matched R, Robust vs KL-matchR on worst_shift_loss:** "
                   f"mean(Robust − KL) = {d_worst:+.5f} → **{verdict}**.")
    else:
        out.append("2. Comparison suspended — see (1).")
    # 3. At matched entropy
    if match_H.status == "OK" and paired_H:
        d_worst = float(np.mean([r["delta_worst_shift_loss"] for r in paired_H]))
        verdict = "Robust BETTER" if d_worst < -1e-5 else (
            "KL BETTER" if d_worst > 1e-5 else "≈ equal"
        )
        out.append(f"3. **At matched entropy, Robust vs KL-matchEntropy on worst_shift_loss:** "
                   f"mean(Robust − KL) = {d_worst:+.5f} → **{verdict}**.")
    else:
        out.append(f"3. **At matched entropy:** comparison suspended (status: {match_H.status}).")
    # 4. Overall framing
    explained = (match_R.status == "OK" and paired_R and
                 abs(float(np.mean([r["delta_worst_shift_loss"] for r in paired_R]))) < 5e-5)
    if explained:
        out.append("4. KL-to-reference smoothing matches the robust effect "
                   "within tolerance; the robust effect may be partially "
                   "explained by generic KL-to-reference smoothing.")
    elif match_R.status == "OK":
        out.append("4. The sensitivity penalty is not equivalent to generic "
                   "KL-to-reference smoothing — Robust differs from KL-matchR "
                   "even after matching R.")
    else:
        out.append("4. Cannot make the smoothing-vs-robust call yet — match selection failed. "
                   "See match-failure recommendation above.")
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="KL-to-ref smoothing baseline report")
    p.add_argument("--confirm-results", type=Path,
                   default=Path("finite_response_sail/results/v3_pairbias_confirm"))
    p.add_argument("--sweep-results", type=Path,
                   default=Path("finite_response_sail/results/v3_pairbias_rhoalg_sweep"),
                   help="Optional: scan rho_alg sweep results too (so target_robust_ratio "
                        "can point to a non-default ratio that lives in the sweep dir).")
    p.add_argument("--kl-results", type=Path,
                   default=Path("finite_response_sail/results/v3_pairbias_kl_smoothing"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("reports/kl_smoothing"))
    p.add_argument("--rho-mis", type=float, default=DEFAULT_RHO_MIS)
    p.add_argument("--target-robust-ratio", type=float, default=0.025,
                   help="rho_alg_ratio of the runs to use as Robust comparison target. "
                        "Default 0.025 (the original confirm setting). Only affects "
                        "selection of KL-matchR / KL-matchEntropy targets; KL training "
                        "is independent.")
    p.add_argument("--cache-dir", type=Path, default=None)
    return p


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    cli = "python -m finite_response_sail._kl_smoothing_report " + " ".join(sys.argv[1:])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    nominal_recs, robust_recs, kl_recs, resolved_ratio = collect_runs(
        args.confirm_results, args.kl_results, args.rho_mis,
        sweep_dir=args.sweep_results,
        target_robust_ratio=args.target_robust_ratio,
    )
    print(f"[kl] runs: nominal={len(nominal_recs)} "
          f"robust(target_ratio={resolved_ratio})={len(robust_recs)} "
          f"kl={len(kl_recs)}")
    robust_label = cps.method_label(resolved_ratio, eta_kl=0.0)

    if not (nominal_recs and robust_recs):
        raise SystemExit(f"[kl] need at least one Nominal and one {robust_label} run")

    # CPS-evaluate everything together
    all_recs = nominal_recs + robust_recs + kl_recs
    by_eta_recs: Dict[float, List[cps.RunRecord]] = defaultdict(list)
    for r in kl_recs:
        by_eta_recs[float(r.config.get("eta_kl", 0.0))].append(r)

    # Run CPS once per group label so we can index back later
    data_cache: Dict[Tuple, Tuple] = {}
    cps_results: List[cps.CPSResult] = []
    for rec in all_recs:
        try:
            cps_results.append(cps.evaluate_run(
                rec, ALPHA_GRID, data_cache, cache_dir_override=args.cache_dir,
            ))
        except ValueError as e:
            print(f"[kl] WARN evaluate_run failed: {e}", file=sys.stderr)

    nominal_cps = [r for r in cps_results if r.method == "Nominal"]
    robust_cps = [r for r in cps_results if r.method == robust_label]
    kl_cps_by_eta: Dict[float, List[cps.CPSResult]] = defaultdict(list)
    for r in cps_results:
        if r.eta_kl > 0.0:
            kl_cps_by_eta[r.eta_kl].append(r)

    nominal = aggregate_group("Nominal", nominal_recs, nominal_cps)
    robust = aggregate_group(robust_label, robust_recs, robust_cps)
    kl_groups: List[GroupAgg] = []
    for eta in sorted(by_eta_recs):
        g = aggregate_group(f"KL-η={eta:g}", by_eta_recs[eta], kl_cps_by_eta[eta])
        if g:
            kl_groups.append(g)

    write_table_csv(([nominal] if nominal else []) + ([robust] if robust else []) + kl_groups,
                    args.out_dir / "kl_smoothing_table.csv")

    # Match selection — uses val.R / val.entropy
    if not robust:
        raise SystemExit("[kl] Robust group missing")
    target_R = robust.val_R[0] if not (robust.val_R[0] is None or
                                        (isinstance(robust.val_R[0], float) and math.isnan(robust.val_R[0]))) else robust.R[0]
    nominal_R = nominal.val_R[0] if (nominal and not math.isnan(nominal.val_R[0])) else (
        nominal.R[0] if nominal else 0.0
    )
    target_H = robust.val_entropy[0] if not math.isnan(robust.val_entropy[0]) else robust.entropy[0]
    nominal_H = nominal.val_entropy[0] if (nominal and not math.isnan(nominal.val_entropy[0])) else (
        nominal.entropy[0] if nominal else 0.0
    )

    match_R = select_match(kl_groups, target_R, "R", nominal_R)
    match_H = select_match(kl_groups, target_H, "entropy", nominal_H)

    # Build selected list
    selected: List[Tuple[GroupAgg, str]] = []
    if nominal: selected.append((nominal, "OK"))
    if robust: selected.append((robust, "OK"))
    if match_R.chosen_eta is not None:
        for g in kl_groups:
            if abs(g.eta_kl - match_R.chosen_eta) < 1e-12:
                # Re-label for the selected table.
                g_renamed = GroupAgg(**{**g.__dict__, "label": "KL-matchR"})
                selected.append((g_renamed, match_R.status))
                break
    else:
        selected.append((GroupAgg(
            label="KL-matchR", eta_kl=math.nan, rho_alg_ratio=0.0,
            n_seeds=0, seeds=[], val_R=(math.nan, 0.0),
            val_entropy=(math.nan, 0.0), val_KL_to_ref=(math.nan, 0.0),
            R=(math.nan, 0.0), E_r_star=(math.nan, 0.0),
            entropy=(math.nan, 0.0), theta_norm=(math.nan, 0.0),
            KL_to_ref=(math.nan, 0.0), L_SAIL_Pstar=(math.nan, 0.0),
            wcl_rho_mis=(math.nan, 0.0), avg_shift_loss=(math.nan, 0.0),
            worst_shift_loss=(math.nan, 0.0), shift_gap=(math.nan, 0.0),
            val_source="missing",
        ), match_R.status))
    if match_H.chosen_eta is not None:
        for g in kl_groups:
            if abs(g.eta_kl - match_H.chosen_eta) < 1e-12:
                g_renamed = GroupAgg(**{**g.__dict__, "label": "KL-matchEntropy"})
                selected.append((g_renamed, match_H.status))
                break
    else:
        selected.append((GroupAgg(
            label="KL-matchEntropy", eta_kl=math.nan, rho_alg_ratio=0.0,
            n_seeds=0, seeds=[], val_R=(math.nan, 0.0),
            val_entropy=(math.nan, 0.0), val_KL_to_ref=(math.nan, 0.0),
            R=(math.nan, 0.0), E_r_star=(math.nan, 0.0),
            entropy=(math.nan, 0.0), theta_norm=(math.nan, 0.0),
            KL_to_ref=(math.nan, 0.0), L_SAIL_Pstar=(math.nan, 0.0),
            wcl_rho_mis=(math.nan, 0.0), avg_shift_loss=(math.nan, 0.0),
            worst_shift_loss=(math.nan, 0.0), shift_gap=(math.nan, 0.0),
            val_source="missing",
        ), match_H.status))

    write_selected_comparison_csv(selected, args.out_dir / "kl_selected_comparison.csv")

    # Paired deltas (only when status OK)
    paired_R_rows: List[Dict[str, Any]] = []
    paired_H_rows: List[Dict[str, Any]] = []
    if match_R.status == "OK":
        paired_R_rows = compute_paired_delta_robust_vs_kl(
            robust_cps, [r for rs in kl_cps_by_eta.values() for r in rs],
            match_R.chosen_eta,
        )
    if match_H.status == "OK":
        paired_H_rows = compute_paired_delta_robust_vs_kl(
            robust_cps, [r for rs in kl_cps_by_eta.values() for r in rs],
            match_H.chosen_eta,
        )
    write_paired_delta_csv(paired_R_rows, paired_H_rows,
                           args.out_dir / "kl_paired_delta.csv",
                           match_R.status, match_H.status)

    # Plots
    if kl_groups:
        plot_eta_curves(kl_groups, nominal, robust, args.out_dir, robust_label=robust_label)
        plot_R_vs_worst(kl_groups, nominal, robust, match_R, args.out_dir,
                        robust_label=robust_label)
        plot_true_vs_worst(kl_groups, nominal, robust, match_R, args.out_dir,
                           robust_label=robust_label)
        by_label_results: Dict[str, List[cps.CPSResult]] = {
            "Nominal": nominal_cps,
            robust_label: robust_cps,
        }
        if match_R.chosen_eta is not None:
            by_label_results["KL-matchR"] = kl_cps_by_eta.get(match_R.chosen_eta, [])
        if match_H.chosen_eta is not None and match_H.chosen_eta != match_R.chosen_eta:
            by_label_results["KL-matchEntropy"] = kl_cps_by_eta.get(match_H.chosen_eta, [])
        plot_alpha_curves(by_label_results, args.out_dir, robust_label=robust_label)
        plot_selected_bar(selected, args.out_dir)

    render_report_md(
        nominal, robust, kl_groups,
        match_R, match_H, selected,
        paired_R_rows, paired_H_rows,
        args.rho_mis,
        args.out_dir / "kl_smoothing_report.md",
        cli,
    )
    print(f"[kl] wrote {args.out_dir / 'kl_smoothing_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
