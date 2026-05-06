"""Unified extended-experiment report.

Produces two seed-intersection-restricted tables plus the legacy combined
view, all at rho_mis = 0.075:

    extended_main_table_n10.{csv,md}
        Nominal / Robust-0.025 / rhoalg_best_worst_shift /
        rhoalg_best_under_utility_tolerance
        — restricted to the seed intersection across these 4 methods.

    kl_control_table_seed_intersection.{csv,md}
        Robust-0.025 / KL-matchR / KL-matchEntropy
        — restricted to the seed intersection across these 3 methods.

    extended_main_table.{csv,md}        (legacy combined view)

    extended_report.md                  (top-level summary doc)

When a method's runs aren't available, its row is marked MISSING and
intersections are computed over methods that ARE available.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import _cps_shift_eval as cps

DEFAULT_RHO_MIS = 0.075
ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Column order shared across both tables (keeps the two viewers consistent).
METRIC_COLS = [
    ("E_r_star", "E[r*]"),
    ("R", "R"),
    ("L_SAIL_Pstar", "L_SAIL(P*)"),
    ("wcl_rho_mis", "wcl(ρ_mis)"),
    ("avg_shift_loss", "avg_shift"),
    ("worst_shift_loss", "worst_shift"),
    ("shift_gap", "shift_gap"),
    ("entropy", "entropy"),
    ("theta_norm", "||θ||"),
    ("KL_to_ref", "KL_to_ref"),
]


def _load_csv_dict(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _try_float(s: Any) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


def _se(vals: Sequence[float]) -> float:
    import numpy as np
    v = np.asarray(vals, dtype=np.float64)
    if v.size <= 1:
        return 0.0
    return float(v.std(ddof=1) / math.sqrt(v.size))


# --------------------------------------------------------------------------
# Method discovery
# --------------------------------------------------------------------------

def _filter(recs, *, eta_kl=None, ratio=None):
    """Filter list of cps.RunRecord by eta_kl and rho_alg_ratio (if given)."""
    out = []
    for r in recs:
        cfg = r.config
        if eta_kl is not None:
            if abs(float(cfg.get("eta_kl", 0.0) or 0.0) - eta_kl) > 1e-9:
                continue
        if ratio is not None:
            if abs(float(cfg.get("rho_alg_ratio", 0.0)) - ratio) > 1e-9:
                continue
            if float(cfg.get("eta_kl", 0.0) or 0.0) > 0.0:
                continue
        out.append(r)
    return out


def _discover_methods(
    confirm_dir: Path, rhoalg_dir: Path, kl_dir: Path,
    rho_mis: float,
    kl_match_R_eta: Optional[float],
    kl_match_H_eta: Optional[float],
) -> Tuple[Dict[str, List[cps.RunRecord]], Dict[str, str]]:
    """Returns (label -> [RunRecord], label -> status)."""
    dirs = [d for d in (confirm_dir, rhoalg_dir, kl_dir) if d is not None]
    all_recs = cps.discover_runs([d for d in dirs if d.exists()])
    # Only this rho_mis
    all_recs = [r for r in all_recs if abs(float(r.config["rho_mis"]) - rho_mis) < 1e-9]

    methods: Dict[str, List[cps.RunRecord]] = {}
    statuses: Dict[str, str] = {}

    methods["Nominal"] = _filter(all_recs, ratio=0.0)
    methods["Robust-0.025rho"] = _filter(all_recs, ratio=0.025)

    # rhoalg best by mean worst_shift (resolved later from rhoalg_table.csv)
    # For now, populate slots; the caller will pick the best ratio from
    # the rhoalg_table.csv to avoid recomputation.
    return all_recs, methods, statuses


def _resolve_rhoalg_best(
    rhoalg_table: List[Dict[str, str]], all_recs: List[cps.RunRecord],
    rho_mis: float, utility_tol: float,
) -> Tuple[Dict[str, List[cps.RunRecord]], Dict[str, float]]:
    """Returns method name -> [RunRecord]; method name -> ratio chosen."""
    out: Dict[str, List[cps.RunRecord]] = {}
    ratios: Dict[str, float] = {}
    if not rhoalg_table:
        return out, ratios
    valid = [r for r in rhoalg_table if r.get("worst_shift_loss_mean")]
    if valid:
        best = min(valid, key=lambda r: _try_float(r["worst_shift_loss_mean"]))
        best_ratio = _try_float(best["rho_alg_ratio"])
        out[f"rhoalg_best_worst_shift (ratio={best_ratio:g})"] = _filter(all_recs, ratio=best_ratio)
        ratios[f"rhoalg_best_worst_shift (ratio={best_ratio:g})"] = best_ratio

        nominal_E = None
        for r in rhoalg_table:
            if abs(_try_float(r.get("rho_alg_ratio")) - 0.0) < 1e-9:
                nominal_E = _try_float(r.get("E_r_star_mean"))
                break
        if nominal_E is not None and not math.isnan(nominal_E):
            elig = [r for r in rhoalg_table
                    if r.get("E_r_star_mean") and r.get("worst_shift_loss_mean")
                    and _try_float(r["E_r_star_mean"]) >= nominal_E - utility_tol]
            if elig:
                bestu = min(elig, key=lambda r: _try_float(r["worst_shift_loss_mean"]))
                bestu_ratio = _try_float(bestu["rho_alg_ratio"])
                lbl = f"rhoalg_best_under_utility_tolerance (ratio={bestu_ratio:g})"
                out[lbl] = _filter(all_recs, ratio=bestu_ratio)
                ratios[lbl] = bestu_ratio
    return out, ratios


def _resolve_kl_matches(
    kl_selected: List[Dict[str, str]], all_recs: List[cps.RunRecord],
) -> Tuple[Dict[str, List[cps.RunRecord]], Dict[str, str], Dict[str, float]]:
    out: Dict[str, List[cps.RunRecord]] = {}
    statuses: Dict[str, str] = {}
    etas: Dict[str, float] = {}
    for src_label, our_label in [("KL-matchR", "KL-matchR"),
                                  ("KL-matchEntropy", "KL-matchEntropy")]:
        match = next((r for r in kl_selected if r.get("label") == src_label), None)
        if match is None:
            statuses[our_label] = "MISSING"
            out[our_label] = []
            continue
        eta = _try_float(match.get("eta_kl"))
        status = match.get("selection_status", "OK")
        if math.isnan(eta) or eta <= 0.0:
            statuses[our_label] = status if status != "OK" else "MISSING"
            out[our_label] = []
            continue
        out[our_label] = _filter(all_recs, eta_kl=eta)
        statuses[our_label] = status
        etas[our_label] = eta
    return out, statuses, etas


# --------------------------------------------------------------------------
# Per-method seed-restricted aggregation
# --------------------------------------------------------------------------

def _run_metrics(
    rec: cps.RunRecord, alpha_grid: List[float],
    data_cache: Dict[Tuple, Tuple], cache_dir_override: Optional[Path],
) -> Dict[str, float]:
    """Compute one row of per-run metrics (alpha-aggregated where applicable)."""
    import numpy as np
    cr = cps.evaluate_run(rec, alpha_grid, data_cache,
                          cache_dir_override=cache_dir_override)
    avg_shift = float(np.mean(cr.L_alpha_exact))
    worst_shift = float(np.max(cr.L_alpha_exact))
    l1 = cr.L_alpha_1_exact()
    shift_gap = worst_shift - (l1 if l1 is not None else worst_shift)
    return {
        "seed": cr.seed,
        "E_r_star": cr.E_r_star,
        "R": cr.R,
        "L_SAIL_Pstar": float(rec.metrics_test.get("nominal_bt_loss", math.nan)),
        "wcl_rho_mis": float(rec.metrics_test.get("wcl_true_rho_mis", math.nan)),
        "avg_shift_loss": avg_shift,
        "worst_shift_loss": worst_shift,
        "shift_gap": shift_gap,
        "entropy": cr.entropy,
        "theta_norm": cr.theta_norm,
        "KL_to_ref": cr.KL_to_ref,
    }


def _aggregate_on_seeds(
    methods: Dict[str, List[cps.RunRecord]],
    cache_dir_override: Optional[Path],
) -> Tuple[Dict[str, Dict[str, Tuple[float, float]]], List[int], Dict[str, List[int]]]:
    """For each method, evaluate every run; restrict to seed intersection;
    return aggregated stats.

    Returns:
        agg[method][metric] = (mean, se) on intersection
        intersection_seeds: sorted list of seeds in the intersection
        per_method_seeds: method -> all seeds available (pre-intersection)
    """
    data_cache: Dict[Tuple, Tuple] = {}
    per_method_runs: Dict[str, List[Dict[str, float]]] = {}
    per_method_seeds: Dict[str, List[int]] = {}
    for label, recs in methods.items():
        rows: List[Dict[str, float]] = []
        for rec in recs:
            try:
                rows.append(_run_metrics(rec, ALPHA_GRID, data_cache, cache_dir_override))
            except ValueError as e:
                print(f"[extended] WARN: skip {rec.run_dir.name}: {e}", file=sys.stderr)
        per_method_runs[label] = rows
        per_method_seeds[label] = sorted({int(r["seed"]) for r in rows})
    if not per_method_runs:
        return {}, [], per_method_seeds
    intersection = None
    for label, rows in per_method_runs.items():
        seeds_here = {int(r["seed"]) for r in rows}
        if intersection is None:
            intersection = seeds_here
        else:
            intersection &= seeds_here
    intersection = sorted(intersection or [])

    import numpy as np
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for label, rows in per_method_runs.items():
        sub = [r for r in rows if int(r["seed"]) in intersection]
        agg: Dict[str, Tuple[float, float]] = {}
        for k, _ in METRIC_COLS:
            vals = [r[k] for r in sub if not math.isnan(r[k])]
            if vals:
                agg[k] = (float(np.mean(vals)), _se(vals))
            else:
                agg[k] = (math.nan, 0.0)
        agg["__n_seeds"] = (float(len(sub)), 0.0)
        out[label] = agg
    return out, intersection, per_method_seeds


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _fmt_pm(m: float, s: float, digits: int = 4) -> str:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "—"
    return f"{m:.{digits}f}±{s:.{digits}f}"


def _render_table_md(
    title: str, agg: Dict[str, Dict[str, Tuple[float, float]]],
    method_order: List[str], statuses: Dict[str, str],
    intersection_seeds: List[int], per_method_seeds: Dict[str, List[int]],
    rho_mis: float,
) -> str:
    lines = [f"## {title}\n"]
    lines.append(f"- ρ_mis = {rho_mis:g}")
    lines.append(f"- seed intersection: {intersection_seeds}  (n={len(intersection_seeds)})")
    miss = [(m, sorted(set(per_method_seeds.get(m, [])) - set(intersection_seeds)))
            for m in method_order]
    miss = [(m, s) for m, s in miss if s]
    if miss:
        for m, s in miss:
            lines.append(f"- dropped from `{m}`: {s}")
    lines.append("")
    headers = ["method", "status", "n"] + [name for _, name in METRIC_COLS]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for m in method_order:
        a = agg.get(m, {})
        n = int(a.get("__n_seeds", (0, 0))[0]) if a else 0
        status = statuses.get(m, "OK" if n > 0 else "MISSING")
        if not a or n == 0:
            row = [m, status, "0"] + ["—"] * len(METRIC_COLS)
        else:
            row = [m, status, str(n)] + [
                _fmt_pm(*a.get(k, (math.nan, 0.0)),
                        digits=5 if k in ("avg_shift_loss", "worst_shift_loss",
                                          "shift_gap") else 4)
                for k, _ in METRIC_COLS
            ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _write_table_csv(
    agg: Dict[str, Dict[str, Tuple[float, float]]],
    method_order: List[str], statuses: Dict[str, str],
    intersection_seeds: List[int],
    rho_mis: float, path: Path,
):
    cols = ["method", "status", "n_seeds", "intersection_seeds"]
    for k, _ in METRIC_COLS:
        cols += [f"{k}_mean", f"{k}_se"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m in method_order:
            a = agg.get(m, {})
            n = int(a.get("__n_seeds", (0, 0))[0]) if a else 0
            status = statuses.get(m, "OK" if n > 0 else "MISSING")
            row = [m, status, n, ";".join(str(s) for s in intersection_seeds)]
            for k, _ in METRIC_COLS:
                mn, se = a.get(k, (math.nan, 0.0)) if a else (math.nan, 0.0)
                row += [f"{mn:.10f}" if not math.isnan(mn) else "",
                        f"{se:.10f}" if not math.isnan(se) else ""]
            w.writerow(row)


# --------------------------------------------------------------------------
# Top-level driver
# --------------------------------------------------------------------------

def render_extended_report_md(
    cps_dir: Path, rhoalg_dir: Path, kl_dir: Path,
    main_table_md: str, kl_control_table_md: str,
    main_table_path: Path, kl_control_table_path: Path,
    cli: str, path: Path,
):
    md: List[str] = []
    md.append("# Extended-experiment unified report\n")
    md.append("This report aggregates results from the three CPS / ρ_alg / "
              "KL-smoothing sub-experiments at ρ_mis = 0.075. The two main "
              "tables are restricted to a per-table **seed intersection** so "
              "every method in a table is paired across the same seeds.\n")
    md.append(f"Generated by: `{cli}`\n")

    md.append("## Source artifacts\n")
    for d, name in [(cps_dir, "CPS shift eval"),
                    (rhoalg_dir, "ρ_alg sweep"),
                    (kl_dir, "KL smoothing")]:
        md.append(f"- {name}: `{d}` "
                  f"({'present' if d.exists() else 'MISSING'})")
    md.append("")

    md.append(f"## Table 1 — Confirm + ρ_alg (file: `{main_table_path.name}`)\n")
    md.append(main_table_md)
    md.append("")

    md.append(f"## Table 2 — KL control (file: `{kl_control_table_path.name}`)\n")
    md.append(kl_control_table_md)
    md.append("")

    md.append("## Status legend\n")
    md.append("- `OK`: row populated from the source report\n"
              "- `MISSING`: source CSV had no matching row (e.g., experiment not yet run)\n"
              "- `MATCH_FAILED`: KL-matchR/matchH selection could not find a "
              "satisfying eta_kl within tolerance\n"
              "- `DIAGNOSTIC_VAL_MISSING`: selection used train metrics instead of val\n")

    md.append("## Pointers to per-experiment reports\n")
    md.append(f"- `{cps_dir}/cps_report.md`")
    md.append(f"- `{rhoalg_dir}/rhoalg_report.md`")
    md.append(f"- `{kl_dir}/kl_smoothing_report.md`")
    md.append("")
    path.write_text("\n".join(md))


def build_parser():
    p = argparse.ArgumentParser(description="Unified extended-experiment report")
    p.add_argument("--cps-dir", type=Path, default=Path("reports/cps_shift_eval"))
    p.add_argument("--rhoalg-dir", type=Path, default=Path("reports/rhoalg_sweep"))
    p.add_argument("--kl-dir", type=Path, default=Path("reports/kl_smoothing"))
    p.add_argument("--out-dir", type=Path, default=Path("reports/extended"))
    p.add_argument("--rho-mis", type=float, default=DEFAULT_RHO_MIS)
    p.add_argument("--utility-tol", type=float, default=1e-3)
    p.add_argument("--confirm-results", type=Path,
                   default=Path("finite_response_sail/results/v3_pairbias_confirm"))
    p.add_argument("--sweep-results", type=Path,
                   default=Path("finite_response_sail/results/v3_pairbias_rhoalg_sweep"))
    p.add_argument("--kl-results", type=Path,
                   default=Path("finite_response_sail/results/v3_pairbias_kl_smoothing"))
    p.add_argument("--cache-dir", type=Path, default=None)
    return p


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    cli = ("python -m finite_response_sail._extended_experiment_report "
           + " ".join(sys.argv[1:]))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rho_mis = args.rho_mis

    # Discover all runs across the three results directories
    all_recs = cps.discover_runs([
        d for d in (args.confirm_results, args.sweep_results, args.kl_results)
        if d.exists()
    ])
    all_recs = [r for r in all_recs if abs(float(r.config["rho_mis"]) - rho_mis) < 1e-9]

    # Resolve KL-match selections from the existing KL report (uses val metrics).
    kl_sel = _load_csv_dict(args.kl_dir / "kl_selected_comparison.csv")
    kl_methods, kl_statuses, kl_etas = _resolve_kl_matches(kl_sel, all_recs)

    # Resolve rhoalg best ratios from the existing rhoalg report.
    rh_table = _load_csv_dict(args.rhoalg_dir / "rhoalg_table.csv")
    rh_methods, rh_ratios = _resolve_rhoalg_best(rh_table, all_recs, rho_mis, args.utility_tol)

    # Build the two tables.
    statuses_combined: Dict[str, str] = dict(kl_statuses)
    main_methods: Dict[str, List[cps.RunRecord]] = {}
    main_methods["Nominal"] = _filter(all_recs, ratio=0.0)
    main_methods["Robust-0.025rho"] = _filter(all_recs, ratio=0.025)
    for k, v in rh_methods.items():
        main_methods[k] = v

    kl_table_methods: Dict[str, List[cps.RunRecord]] = {}
    kl_table_methods["Robust-0.025rho"] = _filter(all_recs, ratio=0.025)
    for k, v in kl_methods.items():
        kl_table_methods[k] = v

    print(f"[extended] main-table methods: {[(k, len(v)) for k, v in main_methods.items()]}")
    print(f"[extended] kl-table methods:   {[(k, len(v)) for k, v in kl_table_methods.items()]}")

    main_agg, main_intersect, main_per_seeds = _aggregate_on_seeds(
        main_methods, args.cache_dir,
    )
    kl_agg, kl_intersect, kl_per_seeds = _aggregate_on_seeds(
        kl_table_methods, args.cache_dir,
    )

    # CSVs
    main_csv = args.out_dir / "extended_main_table_n10.csv"
    kl_csv = args.out_dir / "kl_control_table_seed_intersection.csv"
    _write_table_csv(main_agg, list(main_methods), {}, main_intersect,
                     rho_mis, main_csv)
    _write_table_csv(kl_agg, list(kl_table_methods),
                     {**{"Robust-0.025rho": "OK"}, **kl_statuses},
                     kl_intersect, rho_mis, kl_csv)

    # Markdown
    main_md_str = _render_table_md(
        "Confirm + ρ_alg sweep — same-seed comparison",
        main_agg, list(main_methods), {},
        main_intersect, main_per_seeds, rho_mis,
    )
    kl_md_str = _render_table_md(
        "KL control — Robust vs KL-matchR / KL-matchEntropy",
        kl_agg, list(kl_table_methods),
        {"Robust-0.025rho": "OK", **kl_statuses},
        kl_intersect, kl_per_seeds, rho_mis,
    )
    main_md_path = args.out_dir / "extended_main_table_n10.md"
    kl_md_path = args.out_dir / "kl_control_table_seed_intersection.md"
    main_md_path.write_text(main_md_str)
    kl_md_path.write_text(kl_md_str)

    render_extended_report_md(
        args.cps_dir, args.rhoalg_dir, args.kl_dir,
        main_md_str, kl_md_str,
        main_md_path, kl_md_path,
        cli, args.out_dir / "extended_report.md",
    )

    # Keep legacy artifacts (extended_main_table.{csv,md}) so existing
    # downstream consumers don't break: emit them as a copy of the main n10
    # table.
    legacy_csv = args.out_dir / "extended_main_table.csv"
    legacy_md = args.out_dir / "extended_main_table.md"
    legacy_csv.write_text(main_csv.read_text())
    legacy_md.write_text(main_md_str)

    print(f"[extended] wrote {args.out_dir / 'extended_report.md'}")
    print(f"[extended] wrote {main_md_path}")
    print(f"[extended] wrote {kl_md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
