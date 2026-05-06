"""rho_alg sweep report (Experiment 2).

Combines `v3_pairbias_confirm` (ratios 0, 0.025) with new sweep results
(default ratios 0.005, 0.01, 0.05, 0.1) at rho_mis=0.075 and produces:

    reports/rhoalg_sweep/rhoalg_table.csv
    reports/rhoalg_sweep/rhoalg_table.md
    reports/rhoalg_sweep/rhoalg_paired_delta.csv
    reports/rhoalg_sweep/rhoalg_report.md
    reports/rhoalg_sweep/cps_eval/...     (CSV + plots from CPS evaluator)
    reports/rhoalg_sweep/{rhoalg_ratio_curves,
                           rhoalg_true_vs_R_pareto,
                           rhoalg_true_vs_worst_pareto,
                           rhoalg_delta_bar,
                           rhoalg_alpha_curves_selected}.{png,pdf}

Paired comparisons use the seed intersection across all ratios; missing
seeds and missing ratios are reported (partial report rather than crash).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import _cps_shift_eval as cps
from ._field_resolver import field_map_markdown_table

DEFAULT_RHO_MIS = 0.075
SELECTED_RATIOS_FOR_ALPHA_PLOT = [0.0, 0.01, 0.025, 0.05, 0.1]
ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
UTILITY_TOL = 1e-3   # E[r*] drop tolerance vs Nominal


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


def _ratio_label(ratio: float) -> str:
    if abs(ratio) < 1e-12:
        return "Nominal (ratio=0)"
    if abs(ratio - 0.025) < 1e-9:
        return "Robust-0.025rho"
    return f"ratio={ratio:g}"


def collect_runs_per_ratio(
    confirm_results: Path, sweep_results: Optional[Path],
    rho_mis: float,
) -> Dict[float, List[cps.RunRecord]]:
    """Returns {ratio: [RunRecord, ...]} for the target rho_mis."""
    dirs = [confirm_results]
    if sweep_results is not None and sweep_results.exists():
        dirs.append(sweep_results)
    all_runs = cps.discover_runs(dirs)
    by_ratio: Dict[float, List[cps.RunRecord]] = defaultdict(list)
    for r in all_runs:
        if abs(float(r.config["rho_mis"]) - rho_mis) > 1e-9:
            continue
        if float(r.config.get("eta_kl", 0.0) or 0.0) > 0.0:
            continue   # exclude KL-smoothing runs from rhoalg sweep
        ratio = float(r.config.get("rho_alg_ratio", 0.0))
        by_ratio[ratio].append(r)
    return by_ratio


def evaluate_all(by_ratio: Dict[float, List[cps.RunRecord]],
                 alpha_grid: List[float],
                 cache_dir_override: Optional[Path] = None
                 ) -> List[cps.CPSResult]:
    out: List[cps.CPSResult] = []
    data_cache: Dict[Tuple, Tuple] = {}
    for ratio, recs in sorted(by_ratio.items()):
        for rec in recs:
            try:
                out.append(cps.evaluate_run(
                    rec, alpha_grid, data_cache,
                    cache_dir_override=cache_dir_override,
                ))
            except ValueError as e:
                print(f"[rhoalg] WARN evaluate_run failed: {e}", file=sys.stderr)
    return out


def aggregate_per_ratio(results: List[cps.CPSResult], beta: float
                        ) -> Dict[float, Dict[str, Tuple[float, float]]]:
    """Per ratio: mean/SE of headline metrics including CPS shift summaries."""
    by_ratio: Dict[float, List[cps.CPSResult]] = defaultdict(list)
    for r in results:
        by_ratio[r.rho_alg_ratio].append(r)
    out: Dict[float, Dict[str, Tuple[float, float]]] = {}
    for ratio, runs in by_ratio.items():
        rec: Dict[str, List[float]] = defaultdict(list)
        for r in runs:
            rec["E_r_star"].append(r.E_r_star)
            rec["R"].append(r.R)
            rec["L_SAIL_Pstar"].append(r.rec.metrics_test.get("nominal_bt_loss", math.nan))
            rec["wcl_rho_mis"].append(r.rec.metrics_test.get("wcl_true_rho_mis", math.nan))
            rec["entropy"].append(r.entropy)
            rec["top1"].append(r.top1)
            rec["theta_norm"].append(r.theta_norm)
            rec["KL_to_ref"].append(r.KL_to_ref)
            rec["bad_mass"].append(r.rec.metrics_test.get("bad_bias_candidate_mass", math.nan))
            rec["good_mass"].append(r.rec.metrics_test.get("good_candidate_mass", math.nan))
            rec["avg_shift_loss"].append(float(np.mean(r.L_alpha_exact)))
            rec["worst_shift_loss"].append(float(np.max(r.L_alpha_exact)))
            l1 = r.L_alpha_1_exact()
            if l1 is not None:
                rec["shift_gap"].append(max(r.L_alpha_exact) - l1)
        agg = {k: (float(np.mean(v)), _se(v)) for k, v in rec.items()}
        agg["__n_seeds"] = (float(len(runs)), 0.0)
        out[ratio] = agg
    return out


def write_rhoalg_table_csv(per_ratio, rho_mis: float, path: Path):
    cols = ["E_r_star", "R", "L_SAIL_Pstar", "wcl_rho_mis",
            "avg_shift_loss", "worst_shift_loss", "shift_gap",
            "entropy", "top1", "theta_norm", "KL_to_ref",
            "bad_mass", "good_mass"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["rho_mis", "rho_alg_ratio", "rho_alg_abs", "n_seeds"]
        for c in cols:
            header += [f"{c}_mean", f"{c}_se"]
        w.writerow(header)
        for ratio in sorted(per_ratio):
            agg = per_ratio[ratio]
            n = int(agg["__n_seeds"][0])
            row = [f"{rho_mis:g}", f"{ratio:g}", f"{ratio * rho_mis:g}", n]
            for c in cols:
                if c in agg:
                    m, s = agg[c]
                    row += [f"{m:.10f}", f"{s:.10f}"]
                else:
                    row += ["", ""]
            w.writerow(row)


def compute_paired_deltas(results: List[cps.CPSResult], rho_mis: float, beta: float
                          ) -> Tuple[List[Dict[str, Any]], Dict[float, List[int]]]:
    """Per-seed paired Δ vs ratio=0 (Nominal). Returns (rows, paired_seeds_by_ratio)."""
    by_ratio_seed: Dict[float, Dict[int, cps.CPSResult]] = defaultdict(dict)
    for r in results:
        by_ratio_seed[r.rho_alg_ratio][r.seed] = r
    if 0.0 not in by_ratio_seed:
        return [], {}
    base = by_ratio_seed[0.0]
    base_seeds = set(base.keys())
    rows: List[Dict[str, Any]] = []
    paired_by_ratio: Dict[float, List[int]] = {}
    for ratio, runs in sorted(by_ratio_seed.items()):
        if ratio == 0.0:
            continue
        seeds = sorted(base_seeds & set(runs.keys()))
        paired_by_ratio[ratio] = seeds
        for s in seeds:
            r0 = base[s]
            r1 = runs[s]
            d_E = r1.E_r_star - r0.E_r_star
            d_R = r1.R - r0.R
            d_wcl = (r1.rec.metrics_test.get("wcl_true_rho_mis", math.nan)
                     - r0.rec.metrics_test.get("wcl_true_rho_mis", math.nan))
            d_avg = float(np.mean(r1.L_alpha_exact) - np.mean(r0.L_alpha_exact))
            d_worst = float(np.max(r1.L_alpha_exact) - np.max(r0.L_alpha_exact))
            l0_train = r0.L_alpha_1_exact() or 0.0
            l1_train = r1.L_alpha_1_exact() or 0.0
            d_gap = (max(r1.L_alpha_exact) - l1_train) - (max(r0.L_alpha_exact) - l0_train)
            d_H = r1.entropy - r0.entropy
            d_theta = r1.theta_norm - r0.theta_norm
            d_kl = r1.KL_to_ref - r0.KL_to_ref
            d_nom = (r1.rec.metrics_test.get("nominal_bt_loss", math.nan)
                     - r0.rec.metrics_test.get("nominal_bt_loss", math.nan))
            rows.append({
                "rho_mis": rho_mis,
                "rho_alg_ratio": ratio,
                "rho_alg_abs": ratio * rho_mis,
                "seed": s,
                "delta_E_r_star": d_E,
                "delta_R": d_R,
                "delta_L_SAIL_Pstar": d_nom,
                "delta_wcl_rho_mis": d_wcl,
                "delta_avg_shift_loss": d_avg,
                "delta_worst_shift_loss": d_worst,
                "delta_shift_gap": d_gap,
                "delta_entropy": d_H,
                "delta_theta_norm": d_theta,
                "delta_KL_to_ref": d_kl,
                "decomp_residual": d_wcl - (d_nom + rho_mis * beta * d_R),
            })
    return rows, paired_by_ratio


def write_paired_delta_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(
            "rho_mis,rho_alg_ratio,rho_alg_abs,seed,delta_E_r_star,delta_R,"
            "delta_L_SAIL_Pstar,delta_wcl_rho_mis,delta_avg_shift_loss,"
            "delta_worst_shift_loss,delta_shift_gap,delta_entropy,"
            "delta_theta_norm,delta_KL_to_ref,decomp_residual\n"
        )
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_ratio_curves(per_ratio, out_dir: Path):
    ratios = sorted(per_ratio.keys())
    panels = [("E_r_star", "E[r*] ↑"), ("R", "R ↓"),
              ("worst_shift_loss", "worst_shift_loss ↓"),
              ("avg_shift_loss", "avg_shift_loss ↓")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (k, name) in zip(axes.flat, panels):
        means = [per_ratio[r].get(k, (math.nan, 0.0))[0] for r in ratios]
        ses = [per_ratio[r].get(k, (math.nan, 0.0))[1] for r in ratios]
        ax.errorbar(ratios, means, yerr=ses, marker="o", capsize=3)
        if any(abs(r - 0.025) < 1e-9 for r in ratios):
            ax.axvline(0.025, color="tab:orange", linestyle="--",
                       alpha=0.7, label="confirm setting (0.025)")
            ax.legend(fontsize=8)
        ax.set_xlabel("rho_alg_ratio")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    fig.suptitle(f"ρ_alg sweep (rho_mis=0.075): mean ± SE", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out_dir, "rhoalg_ratio_curves")


def plot_pareto_true_vs_R(results, per_ratio, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    by_ratio: Dict[float, List[cps.CPSResult]] = defaultdict(list)
    for r in results:
        by_ratio[r.rho_alg_ratio].append(r)
    for ratio in sorted(by_ratio):
        runs = by_ratio[ratio]
        xs = [r.R for r in runs]
        ys = [r.E_r_star for r in runs]
        ax.scatter(xs, ys, alpha=0.3, s=22)
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        ax.scatter([mx], [my], s=130, edgecolor="black", linewidths=1.4,
                   label=f"ratio={ratio:g}", zorder=5)
        ax.annotate(f"{ratio:g}", (mx, my), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.set_xlabel("R ↓ (sensitivity_R)")
    ax.set_ylabel("E[r*] ↑")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title("ρ_alg sweep Pareto: utility vs sensitivity (toward upper-left = better)")
    fig.tight_layout()
    _save_fig(fig, out_dir, "rhoalg_true_vs_R_pareto")


def plot_pareto_true_vs_worst(results, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    by_ratio: Dict[float, List[cps.CPSResult]] = defaultdict(list)
    for r in results:
        by_ratio[r.rho_alg_ratio].append(r)
    for ratio in sorted(by_ratio):
        runs = by_ratio[ratio]
        xs = [r.E_r_star for r in runs]
        ys = [max(r.L_alpha_exact) for r in runs]
        ax.scatter(xs, ys, alpha=0.3, s=22)
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        ax.scatter([mx], [my], s=130, edgecolor="black", linewidths=1.4,
                   label=f"ratio={ratio:g}", zorder=5)
        ax.annotate(f"{ratio:g}", (mx, my), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.set_xlabel("E[r*] ↑")
    ax.set_ylabel("worst_shift_loss ↓")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title("ρ_alg sweep Pareto: utility vs CPS worst-shift loss")
    fig.tight_layout()
    _save_fig(fig, out_dir, "rhoalg_true_vs_worst_pareto")


def plot_delta_bar(paired_rows: List[Dict[str, Any]], out_dir: Path):
    by_ratio: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for r in paired_rows:
        by_ratio[r["rho_alg_ratio"]].append(r)
    ratios = sorted(by_ratio)
    if not ratios:
        return
    metrics = [
        ("delta_E_r_star",          "ΔE[r*]"),
        ("delta_R",                 "ΔR"),
        ("delta_wcl_rho_mis",       "Δwcl(ρ_mis)"),
        ("delta_worst_shift_loss",  "Δworst_shift_loss"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ratios))
    width = 0.2
    for i, (k, name) in enumerate(metrics):
        means = [float(np.mean([r[k] for r in by_ratio[ratio]])) for ratio in ratios]
        ses = [_se([r[k] for r in by_ratio[ratio]]) for ratio in ratios]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=ses, capsize=3, label=name)
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:g}" for r in ratios])
    ax.set_xlabel("rho_alg_ratio (vs ratio=0 baseline)")
    ax.set_ylabel("paired Δ")
    ax.set_title("ρ_alg sweep: paired Δ vs Nominal (negative = improvement for losses/R)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save_fig(fig, out_dir, "rhoalg_delta_bar")


def plot_alpha_curves_selected(results, out_dir: Path,
                               selected: List[float] = SELECTED_RATIOS_FOR_ALPHA_PLOT):
    by_ratio: Dict[float, List[cps.CPSResult]] = defaultdict(list)
    for r in results:
        by_ratio[r.rho_alg_ratio].append(r)
    fig, ax = plt.subplots(figsize=(8, 5))
    for ratio in selected:
        runs = by_ratio.get(ratio, [])
        if not runs:
            continue
        alphas = runs[0].alpha_values
        arr = np.asarray([r.L_alpha_exact for r in runs])
        m = arr.mean(axis=0)
        s = arr.std(axis=0, ddof=1) / math.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(m)
        ax.plot(alphas, m, marker="o", label=f"ratio={ratio:g}")
        ax.fill_between(alphas, m - s, m + s, alpha=0.18)
    ax.axvline(1.0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("α (CPS shift)")
    ax.set_ylabel("L_α_exact ↓")
    ax.set_title("ρ_alg sweep: L_α curves at selected ratios (rho_mis=0.075)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, "rhoalg_alpha_curves_selected")


def render_rhoalg_table_md(per_ratio, rho_mis: float, path: Path):
    lines = ["| ratio | rho_alg | n | E[r*] | R | L_SAIL(P*) | wcl(ρ_mis) | avg_shift | worst_shift | entropy | KL_to_ref |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ratio in sorted(per_ratio):
        agg = per_ratio[ratio]
        n = int(agg["__n_seeds"][0])
        def fmt(k):
            if k in agg:
                m, s = agg[k]
                return f"{m:.4f}±{s:.4f}"
            return "—"
        lines.append("| " + " | ".join([
            f"{ratio:g}", f"{ratio * rho_mis:g}", str(n),
            fmt("E_r_star"), fmt("R"), fmt("L_SAIL_Pstar"),
            fmt("wcl_rho_mis"), fmt("avg_shift_loss"),
            fmt("worst_shift_loss"), fmt("entropy"), fmt("KL_to_ref"),
        ]) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _best_ratio(per_ratio, key: str, reverse: bool = False) -> Optional[float]:
    """Return ratio minimizing (or maximizing if reverse) the given key's mean."""
    candidates = [(r, agg[key][0]) for r, agg in per_ratio.items() if key in agg
                  and not math.isnan(agg[key][0])]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=reverse)
    return candidates[0][0]


def _best_under_utility_tolerance(per_ratio, baseline_E: float, key: str = "worst_shift_loss"
                                  ) -> Optional[float]:
    """Ratio with min `key` mean among ratios with E[r*] drop ≤ UTILITY_TOL vs baseline."""
    elig = []
    for r, agg in per_ratio.items():
        if "E_r_star" not in agg or key not in agg:
            continue
        if agg["E_r_star"][0] >= baseline_E - UTILITY_TOL:
            elig.append((r, agg[key][0]))
    if not elig:
        return None
    elig.sort(key=lambda x: x[1])
    return elig[0][0]


def render_rhoalg_report_md(
    per_ratio, paired_rows, paired_by_ratio,
    rho_mis: float, beta: float,
    missing_ratios: List[float],
    path: Path, cli_invocation: str,
):
    md: List[str] = []
    md.append("# ρ_alg sweep report\n")
    md.append(f"- ρ_mis: {rho_mis:g}")
    md.append(f"- Ratios with data: {sorted(per_ratio)}")
    if missing_ratios:
        md.append(f"- **MISSING ratios** (no runs found): {missing_ratios}")
    md.append(f"- Seeds paired (per ratio vs Nominal): "
              + ", ".join(f"ratio={r:g}: n={len(s)}" for r, s in sorted(paired_by_ratio.items())))
    md.append(f"- CLI: `{cli_invocation}`\n")

    md.append("## Field-name resolver\n")
    md.append(field_map_markdown_table())
    md.append("")

    md.append("## A. Per-ratio summary\n")
    md.append("(Mean ± SE over seeds; CPS metrics computed on alpha grid "
              + str(ALPHA_GRID) + ".)\n")
    table_path = path.parent / "rhoalg_table.md"
    md.append(table_path.read_text())
    md.append("")

    md.append("## B. Paired Δ vs Nominal (mean ± SE on seed intersection)\n")
    md.append("| ratio | n | ΔE[r*] | ΔR | Δwcl(ρ_mis) | Δworst_shift | Δavg_shift | Δentropy | ΔKL | decomp residual |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    by_ratio_rows: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for r in paired_rows:
        by_ratio_rows[r["rho_alg_ratio"]].append(r)
    for ratio in sorted(by_ratio_rows):
        rs = by_ratio_rows[ratio]
        n = len(rs)
        def m(k): return float(np.mean([r[k] for r in rs]))
        def s(k): return _se([r[k] for r in rs])
        md.append("| " + " | ".join([
            f"{ratio:g}", str(n),
            f"{m('delta_E_r_star'):+.4f}±{s('delta_E_r_star'):.4f}",
            f"{m('delta_R'):+.4f}±{s('delta_R'):.4f}",
            f"{m('delta_wcl_rho_mis'):+.4f}±{s('delta_wcl_rho_mis'):.4f}",
            f"{m('delta_worst_shift_loss'):+.4f}±{s('delta_worst_shift_loss'):.4f}",
            f"{m('delta_avg_shift_loss'):+.4f}±{s('delta_avg_shift_loss'):.4f}",
            f"{m('delta_entropy'):+.4f}±{s('delta_entropy'):.4f}",
            f"{m('delta_KL_to_ref'):+.4f}±{s('delta_KL_to_ref'):.4f}",
            f"{m('decomp_residual'):.2e}",
        ]) + " |")
    md.append("")
    md.append("`decomp residual` tests `Δwcl(ρ_mis) ≈ ΔL_SAIL(P*) + ρ_mis · β · ΔR` "
              "at the per-seed level.\n")

    # Verdict / answers to the four questions in the spec.
    md.append("## C. Answers to the four sweep questions\n")
    nominal_E = per_ratio.get(0.0, {}).get("E_r_star", (math.nan, 0.0))[0]
    best_worst = _best_ratio(per_ratio, "worst_shift_loss")
    best_R = _best_ratio(per_ratio, "R")
    best_under_tol = _best_under_utility_tolerance(per_ratio, nominal_E)
    md.append(f"1. **Which ratio minimizes worst_shift_loss?** ratio = {best_worst!r}")
    md.append(f"2. **Which ratio minimizes R?** ratio = {best_R!r}")
    md.append(f"3. **Which ratio minimizes worst_shift_loss under E[r*] drop ≤ {UTILITY_TOL}?** "
              f"ratio = {best_under_tol!r}")
    if 0.025 in per_ratio:
        agg25 = per_ratio[0.025]
        e_drop = nominal_E - agg25["E_r_star"][0] if not math.isnan(nominal_E) else math.nan
        wsl25 = agg25.get("worst_shift_loss", (math.nan,))[0]
        wsl_best = per_ratio.get(best_worst, {}).get("worst_shift_loss", (math.nan,))[0] if best_worst is not None else math.nan
        md.append(
            f"4. **Is ratio=0.025 still a reasonable choice?** "
            f"E[r*] drop = {e_drop:+.4f}; worst_shift_loss(0.025) = {wsl25:.4f}, "
            f"best worst_shift_loss = {wsl_best:.4f}."
        )
    md.append("")

    md.append("## D. Plots\n")
    for name in ["rhoalg_ratio_curves", "rhoalg_true_vs_R_pareto",
                 "rhoalg_true_vs_worst_pareto", "rhoalg_delta_bar",
                 "rhoalg_alpha_curves_selected"]:
        md.append(f"- `{name}.png` / `{name}.pdf`")
    md.append("")
    path.write_text("\n".join(md))


def build_parser():
    p = argparse.ArgumentParser(description="rho_alg sweep report")
    p.add_argument("--confirm-results", type=Path,
                   default=Path("finite_response_sail/results/v3_pairbias_confirm"))
    p.add_argument("--sweep-results", type=Path,
                   default=Path("finite_response_sail/results/v3_pairbias_rhoalg_sweep"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("reports/rhoalg_sweep"))
    p.add_argument("--rho-mis", type=float, default=DEFAULT_RHO_MIS)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--expected-ratios", type=str,
                   default="0,0.005,0.01,0.025,0.05,0.1",
                   help="Ratios that *should* be present; missing ones are flagged.")
    return p


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    cli = "python -m finite_response_sail._rhoalg_sweep_report " + " ".join(sys.argv[1:])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_ratio = collect_runs_per_ratio(args.confirm_results, args.sweep_results, args.rho_mis)
    print(f"[rhoalg] ratios present: {sorted(by_ratio)}; "
          f"counts: {[(r, len(rs)) for r, rs in sorted(by_ratio.items())]}")
    if not by_ratio:
        raise SystemExit("[rhoalg] no runs found")

    expected = [float(x) for x in args.expected_ratios.split(",")]
    present = set(by_ratio.keys())
    missing = sorted(r for r in expected if r not in present and not any(abs(r - p) < 1e-9 for p in present))

    results = evaluate_all(by_ratio, ALPHA_GRID, cache_dir_override=args.cache_dir)
    if not results:
        raise SystemExit("[rhoalg] no CPS results")

    # Persist the CPS evaluator outputs into a subdir for downstream consumption.
    cps_dir = args.out_dir / "cps_eval"
    cps.write_alpha_metrics_csv(results, cps_dir / "cps_alpha_metrics.csv",
                                loss_field="L_alpha_exact")
    cps.write_summary_csv(cps.aggregate_summary(results, ALPHA_GRID),
                          cps_dir / "cps_summary_by_rho_method.csv")

    beta = float(results[0].beta)
    per_ratio = aggregate_per_ratio(results, beta)
    paired_rows, paired_by_ratio = compute_paired_deltas(results, args.rho_mis, beta)

    write_rhoalg_table_csv(per_ratio, args.rho_mis, args.out_dir / "rhoalg_table.csv")
    render_rhoalg_table_md(per_ratio, args.rho_mis, args.out_dir / "rhoalg_table.md")
    write_paired_delta_csv(paired_rows, args.out_dir / "rhoalg_paired_delta.csv")

    plot_ratio_curves(per_ratio, args.out_dir)
    plot_pareto_true_vs_R(results, per_ratio, args.out_dir)
    plot_pareto_true_vs_worst(results, args.out_dir)
    plot_delta_bar(paired_rows, args.out_dir)
    plot_alpha_curves_selected(results, args.out_dir)

    render_rhoalg_report_md(
        per_ratio, paired_rows, paired_by_ratio,
        args.rho_mis, beta, missing,
        args.out_dir / "rhoalg_report.md", cli,
    )
    print(f"[rhoalg] wrote {args.out_dir / 'rhoalg_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
