"""KL appendix evaluator for the true-to-worst stress experiment.

Re-evaluates KL-smoothing runs (eta_kl > 0, rho_alg = 0) under the same stress
metric as the main evaluator. Selects an "eta_kl_star" by validation
auc_0p5_1, exactly the way the main evaluator selects q_star, and reports a
comparison table:

    Nominal               (eta_kl = 0)
    Robust q_star         (best rho_alg-ratio from the main grid)
    KL-eta_kl_star        (best KL-smoothing eta from this set of runs)
    KL-matchR             (KL eta whose validation R best matches Robust q_star's R)

The main comparison remains Nominal versus Robust q_star (in the main report).
This appendix is a baseline check: if KL-smoothing matches Robust on stress,
say so honestly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._true_to_worst_stress_eval import (
    DEFAULT_ALPHA_GRID, TIE_TOL,
    RunRecord, StressResult, discover_runs, evaluate_run_stress,
    select_q_star_global, _save_fig, _se,
)


def _parse_alpha_grid(spec: str) -> List[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def kl_method_label(eta_kl: float, rho_alg_ratio: float) -> str:
    if rho_alg_ratio > TIE_TOL:
        return f"Robust-{rho_alg_ratio:g}"
    if eta_kl < TIE_TOL:
        return "Nominal"
    return f"KL-{eta_kl:g}"


def select_kl_star_global(
    val_results: List[StressResult],
    metric: str = "auc_0p5_1",
) -> Dict[float, Dict[str, float]]:
    """Pick eta_kl_star globally per rho_mis. Excludes eta=0. Tie-break: smallest eta.

    Mirrors select_q_star_global for the eta_kl axis.
    """
    out: Dict[float, Dict[str, float]] = {}
    by_rho: Dict[float, List[StressResult]] = defaultdict(list)
    for r in val_results:
        if r.eta_kl > TIE_TOL:
            by_rho[r.rho_mis].append(r)
    for rho, runs in sorted(by_rho.items()):
        cand: Dict[float, List[StressResult]] = defaultdict(list)
        for r in runs:
            cand[round(r.eta_kl, 12)].append(r)
        if not cand:
            out[rho] = {"eta_kl_star": float("nan"),
                        "val_auc_0p5_1": float("nan"),
                        "val_R": float("nan")}
            continue
        means = {e: float(np.mean([getattr(r, metric) for r in rs]))
                 for e, rs in cand.items()}
        best_v = min(means.values())
        ties = [e for e, v in means.items() if abs(v - best_v) <= TIE_TOL]
        e_star = min(ties)
        out[rho] = {
            "eta_kl_star": e_star,
            "val_auc_0p5_1": means[e_star],
            "val_R": float(np.mean([r.R for r in cand[e_star]])),
        }
    return out


def select_kl_match_R(
    val_results: List[StressResult],
    target_R_by_rho: Dict[float, float],
) -> Dict[float, Dict[str, float]]:
    """For each rho_mis, pick the eta_kl whose seed-pooled mean R is closest
    to the Robust q_star's mean R.
    """
    out: Dict[float, Dict[str, float]] = {}
    by_rho: Dict[float, List[StressResult]] = defaultdict(list)
    for r in val_results:
        if r.eta_kl > TIE_TOL:
            by_rho[r.rho_mis].append(r)
    for rho, runs in sorted(by_rho.items()):
        target = target_R_by_rho.get(rho, float("nan"))
        if not (target == target):
            out[rho] = {"eta_kl_matchR": float("nan"), "val_R": float("nan")}
            continue
        cand: Dict[float, List[StressResult]] = defaultdict(list)
        for r in runs:
            cand[round(r.eta_kl, 12)].append(r)
        if not cand:
            out[rho] = {"eta_kl_matchR": float("nan"), "val_R": float("nan")}
            continue
        means = {e: float(np.mean([r.R for r in rs])) for e, rs in cand.items()}
        e_match = min(means.keys(), key=lambda e: abs(means[e] - target))
        out[rho] = {"eta_kl_matchR": e_match, "val_R": means[e_match]}
    return out


def write_kl_appendix_csv(
    test_results: List[StressResult],
    q_star_map: Dict[float, Dict[str, float]],
    eta_star_map: Dict[float, Dict[str, float]],
    eta_match_map: Dict[float, Dict[str, float]],
    path: Path,
):
    """One row per (rho_mis, method). Methods: Nominal, Robust q_star, KL-eta_star, KL-matchR."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["rho_mis", "method_label", "selector", "param_value", "n_seeds",
            "L_id_mean", "L_id_se", "L_wc_mean", "L_wc_se",
            "auc_0p5_1_mean", "auc_0p5_1_se", "R_mean", "R_se",
            "L_obs_mean", "L_obs_se", "entropy_mean", "KL_to_ref_mean"]
    rhos = sorted({r.rho_mis for r in test_results})

    def filt(rho, predicate):
        return [r for r in test_results if abs(r.rho_mis - rho) < 1e-9 and predicate(r)]

    def stat(rs, name):
        v = [getattr(r, name) for r in rs]
        return (float(np.mean(v)) if v else float("nan"), _se(v))

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rho in rhos:
            entries = []
            entries.append((
                "Nominal", "fixed", 0.0,
                filt(rho, lambda r: r.eta_kl < TIE_TOL and r.q < TIE_TOL),
            ))
            qs = q_star_map.get(rho, {}).get("q_star_auc", float("nan"))
            if qs == qs:
                entries.append((
                    f"Robust-{qs:g}", "val auc_0p5_1", qs,
                    filt(rho, lambda r: r.eta_kl < TIE_TOL and abs(r.q - qs) < TIE_TOL),
                ))
            es = eta_star_map.get(rho, {}).get("eta_kl_star", float("nan"))
            if es == es:
                entries.append((
                    f"KL-{es:g}", "val auc_0p5_1", es,
                    filt(rho, lambda r: abs(r.eta_kl - es) < TIE_TOL),
                ))
            em = eta_match_map.get(rho, {}).get("eta_kl_matchR", float("nan"))
            if em == em:
                entries.append((
                    f"KL-matchR ({em:g})", "match Robust R", em,
                    filt(rho, lambda r: abs(r.eta_kl - em) < TIE_TOL),
                ))
            for label, selector, pv, runs in entries:
                if not runs:
                    w.writerow([f"{rho:g}", label, selector, f"{pv:g}", 0,
                                "", "", "", "", "", "", "", "", "", "", "", ""])
                    continue
                row = [f"{rho:g}", label, selector, f"{pv:g}", len(runs)]
                for name in ["L_id", "L_wc", "auc_0p5_1", "R", "L_obs"]:
                    m, s = stat(runs, name)
                    row += [f"{m:.10f}", f"{s:.10f}"]
                ent_m, _ = stat(runs, "entropy")
                kl_m, _ = stat(runs, "KL_to_ref")
                row += [f"{ent_m:.10f}", f"{kl_m:.10f}"]
                w.writerow(row)


def plot_kl_appendix_curves(
    test_results: List[StressResult],
    q_star_map: Dict[float, Dict[str, float]],
    eta_star_map: Dict[float, Dict[str, float]],
    out_dir: Path,
):
    rhos = sorted({r.rho_mis for r in test_results})
    fig, axes = plt.subplots(1, len(rhos), figsize=(5 * len(rhos), 4.0), squeeze=False)
    for ax, rho in zip(axes[0], rhos):
        nom = [r for r in test_results
               if abs(r.rho_mis - rho) < 1e-9 and r.eta_kl < TIE_TOL and r.q < TIE_TOL]
        qs = q_star_map.get(rho, {}).get("q_star_auc", float("nan"))
        rob = [r for r in test_results
               if (qs == qs) and abs(r.rho_mis - rho) < 1e-9
               and r.eta_kl < TIE_TOL and abs(r.q - qs) < TIE_TOL]
        es = eta_star_map.get(rho, {}).get("eta_kl_star", float("nan"))
        kl = [r for r in test_results
              if (es == es) and abs(r.rho_mis - rho) < 1e-9
              and abs(r.eta_kl - es) < TIE_TOL]
        if not nom:
            ax.set_title(f"rho_mis={rho:g} (no nominal)")
            continue
        alphas = nom[0].alpha_values
        for arr, label, color in [
            (nom, "Nominal (q=0)", "tab:blue"),
            (rob, f"Robust q_star = {qs:g}" if qs == qs else "Robust", "tab:orange"),
            (kl, f"KL eta_star = {es:g}" if es == es else "KL", "tab:green"),
        ]:
            if not arr:
                continue
            mat = np.asarray([r.L_stress for r in arr])
            m = mat.mean(0)
            s = (mat.std(0, ddof=1) / math.sqrt(mat.shape[0])
                 if mat.shape[0] > 1 else np.zeros_like(m))
            ax.plot(alphas, m, marker="o", color=color, label=label)
            ax.fill_between(alphas, m - s, m + s, alpha=0.2, color=color)
        ax.set_xlabel("alpha")
        ax.set_ylabel("L_stress  (lower is better)")
        ax.set_title(f"rho_mis = {rho:g}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("KL appendix: KL-to-reference smoothing baseline  (lower is better)", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, out_dir, "kl_appendix_curves")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="KL appendix evaluator for the true-to-worst stress experiment",
    )
    p.add_argument("--include-results-dir", action="append", default=[], required=True,
                   help="Results directories to scan (repeatable). Should include the "
                        "KL-smoothing runs and at least one rho_alg-ratio run dir.")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--alpha-grid", type=str,
                   default=",".join(f"{a:g}" for a in DEFAULT_ALPHA_GRID))
    p.add_argument("--rho-mis", type=str, default="")
    p.add_argument("--cache-dir", type=Path, default=None)
    return p


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    cli = "python -m finite_response_sail._true_to_worst_stress_eval_kl " + " ".join(sys.argv[1:])

    results_dirs = [Path(p) for p in args.include_results_dir]
    alpha_grid = _parse_alpha_grid(args.alpha_grid)

    rho_filter = None
    if args.rho_mis:
        rho_filter = {float(x) for x in args.rho_mis.split(",")}

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(results_dirs)
    if rho_filter is not None:
        runs = [r for r in runs if float(r.config["rho_mis"]) in rho_filter]
    if not runs:
        raise SystemExit("[ttw-kl] no runs after filtering")

    print(f"[ttw-kl] discovered {len(runs)} runs over alpha grid {alpha_grid}")

    data_cache: Dict[Tuple, Dict[str, Any]] = {}

    test_results: List[StressResult] = []
    for rec in runs:
        try:
            test_results.append(evaluate_run_stress(
                rec, alpha_grid, "test",
                data_cache=data_cache, cache_dir_override=args.cache_dir,
            ))
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            print(f"[ttw-kl] WARN test eval skip: {rec.run_dir.name}: {e}", file=sys.stderr)

    val_results: List[StressResult] = []
    for rec in runs:
        try:
            val_results.append(evaluate_run_stress(
                rec, alpha_grid, "val",
                data_cache=data_cache, cache_dir_override=args.cache_dir,
            ))
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            print(f"[ttw-kl] WARN val eval skip: {rec.run_dir.name}: {e}", file=sys.stderr)

    # q_star uses only the rho_alg-ratio (eta_kl=0) runs
    q_star_map = select_q_star_global(
        [r for r in val_results if r.eta_kl < TIE_TOL], "auc_0p5_1",
    )
    # eta_kl_star uses only the KL runs (eta_kl>0)
    eta_star_map = select_kl_star_global(val_results, "auc_0p5_1")

    target_R_by_rho: Dict[float, float] = {}
    for rho, sel in q_star_map.items():
        qs = sel.get("q_star_auc", float("nan"))
        if not (qs == qs):
            continue
        rob_runs = [r for r in val_results
                    if r.eta_kl < TIE_TOL and abs(r.rho_mis - rho) < 1e-9
                    and abs(r.q - qs) < TIE_TOL]
        if rob_runs:
            target_R_by_rho[rho] = float(np.mean([r.R for r in rob_runs]))
    eta_match_map = select_kl_match_R(val_results, target_R_by_rho)

    write_kl_appendix_csv(
        test_results, q_star_map, eta_star_map, eta_match_map,
        out_dir / "kl_appendix_summary.csv",
    )
    plot_kl_appendix_curves(test_results, q_star_map, eta_star_map, out_dir)

    md = ["# KL appendix\n",
          "Lower is better for L_id, L_wc, auc_0p5_1, R.",
          f"- CLI: `{cli}`\n",
          "## Selected baselines\n",
          "| rho_mis | q_star | KL eta_kl_star | KL eta_kl matchR |",
          "|---|---|---|---|"]
    rhos = sorted({r.rho_mis for r in test_results})
    for rho in rhos:
        qs = q_star_map.get(rho, {}).get("q_star_auc", float("nan"))
        es = eta_star_map.get(rho, {}).get("eta_kl_star", float("nan"))
        em = eta_match_map.get(rho, {}).get("eta_kl_matchR", float("nan"))
        md.append(f"| {rho:g} | {qs:g} | {es:g} | {em:g} |")
    md.append("\n## Comparison table\n")
    md.append("See `kl_appendix_summary.csv` for the full table and `kl_appendix_curves.png` for the L_stress curves.\n")
    md.append("Honest reading: if KL-eta_star matches or beats Robust q_star on auc_0p5_1, "
              "KL-to-reference smoothing is a strong heuristic baseline in this finite-response abstraction.\n")
    (out_dir / "kl_appendix.md").write_text("\n".join(md))
    print(f"[ttw-kl] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
