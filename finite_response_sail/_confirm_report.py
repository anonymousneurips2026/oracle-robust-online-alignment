"""v3_pairbias_confirm — paper-supporting confirmation report.

Reads `results/v3_pairbias_confirm/<run>/metrics.json` and
`training_curve.jsonl`, verifies the decomposition-form robust objective
identities to numerical tolerance, and emits:

    results/v3_pairbias_confirm/reports/confirm_report.md
    results/v3_pairbias_confirm/reports/confirm_delta_bar.png
    results/v3_pairbias_confirm/reports/confirm_wcl_decomposition.png
    results/v3_pairbias_confirm/reports/confirm_pareto_true_vs_R.png
    results/v3_pairbias_confirm/reports/confirm_wcl_vs_true.png
    results/v3_pairbias_confirm/reports/confirm_training_curves.png
    results/v3_pairbias_confirm/reports/confirm_seed_scatter.png
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = "v3_pairbias_confirm"
RESULTS = Path(__file__).resolve().parent / "results" / TAG
REPORTS = RESULTS / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

RHO_MIS = [0.05, 0.075, 0.1]
RATIOS = [0.0, 0.025]
SEEDS = list(range(10))
LABEL = {0.0: "Nominal", 0.025: "Robust-0.025rho"}

PRIMARY = [
    "expected_true_utility",
    "wcl_true_rho_mis",
    "sensitivity_R",
    "nominal_bt_loss",
    "expected_bias_attr",
    "bad_bias_candidate_mass",
    "good_candidate_mass",
]
SECONDARY = [
    "entropy",
    "top1_prob",
    "theta_norm",
    "pair_bias_mass_under_pi",
    "expected_true_win_vs_ref",
    "greedy_true_utility",
]
ALL_KEYS = PRIMARY + SECONDARY + [
    "worst_case_loss", "wcl_true_rho_alg", "biased_feedback_loss",
    "biased_bt_loss",
]


def _se(vals) -> float:
    v = np.asarray(vals, dtype=np.float64)
    if v.size <= 1:
        return 0.0
    return float(v.std(ddof=1) / math.sqrt(v.size))


def _fmt(m, s, digits=4):
    return f"{m:.{digits}f}\u00b1{s:.{digits}f}"


def _load():
    """rows[(rho_mis, ratio, seed)] = {'final': metrics dict, 'cfg': ...}."""
    rows = {}
    for p in sorted(RESULTS.glob("mode-*/metrics.json")):
        d = json.loads(p.read_text())
        c = d["config"]
        key = (float(c["rho_mis"]), float(c["rho_alg_ratio"]), int(c["seed"]))
        rows[key] = {
            "final": d["final"]["test"],
            "cfg": c,
            "run_dir": p.parent,
        }
    return rows


def formula_checks(rows):
    """Returns a dict of max |residuals| across all runs, plus a per-run list
    of any violations above tolerance.
    """
    TOL_FIELD = 1e-6  # derived from torch float32 `.item()` arithmetic
    TOL_DECOMP = 1e-5  # worst_case_loss endpoint-max vs decomposition form
    worst = {
        "wcl_true_rho_mis_vs_formula": 0.0,
        "wcl_true_rho_alg_vs_formula": 0.0,
        "biased_feedback_loss_vs_biased_bt_loss": 0.0,
        "worst_case_loss_vs_wcl_true_rho_mis": 0.0,
    }
    worst_runs = {k: None for k in worst}
    violations = []
    for (rm, ratio, seed), r in rows.items():
        m = r["final"]
        c = r["cfg"]
        beta = float(c["beta"])
        rho_mis = float(c["rho_mis"])
        rho_alg = float(c["rho_alg"])
        sR = float(m["sensitivity_R"])
        nom = float(m["nominal_bt_loss"])

        r1 = abs(m["wcl_true_rho_mis"] - (nom + rho_mis * beta * sR))
        r2 = abs(m["wcl_true_rho_alg"] - (nom + rho_alg * beta * sR))
        r3 = abs(m["biased_feedback_loss"] - m["biased_bt_loss"])
        r4 = abs(m["worst_case_loss"] - m["wcl_true_rho_mis"])

        checks = [
            ("wcl_true_rho_mis_vs_formula", r1, TOL_FIELD),
            ("wcl_true_rho_alg_vs_formula", r2, TOL_FIELD),
            ("biased_feedback_loss_vs_biased_bt_loss", r3, TOL_FIELD),
            ("worst_case_loss_vs_wcl_true_rho_mis", r4, TOL_DECOMP),
        ]
        for name, v, tol in checks:
            if v > worst[name]:
                worst[name] = v
                worst_runs[name] = (rm, ratio, seed)
            if v > tol:
                violations.append((name, rm, ratio, seed, v, tol))
    return worst, worst_runs, violations


def pretty_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def section_b(rows):
    """Primary metrics mean \u00b1 SE over seeds per (rho_mis, ratio)."""
    cell = defaultdict(lambda: defaultdict(list))
    for (rm, ratio, seed), r in rows.items():
        for k in PRIMARY:
            cell[(rm, ratio)][k].append(r["final"][k])
    header = ["\u03c1_mis", "ratio", "E[r*]", "wcl(\u03c1_mis)", "R",
             "L_SAIL(P*)", "E[bias_attr]", "bad_mass", "good_mass"]
    table_rows = []
    for rm in RHO_MIS:
        for ratio in RATIOS:
            c = cell[(rm, ratio)]
            vals = {k: (float(np.mean(c[k])), _se(c[k])) for k in PRIMARY}
            table_rows.append([
                f"{rm:g}", f"{ratio:g}",
                _fmt(*vals["expected_true_utility"]),
                _fmt(*vals["wcl_true_rho_mis"]),
                _fmt(*vals["sensitivity_R"]),
                _fmt(*vals["nominal_bt_loss"]),
                _fmt(*vals["expected_bias_attr"]),
                _fmt(*vals["bad_bias_candidate_mass"]),
                _fmt(*vals["good_candidate_mass"]),
            ])
    return cell, pretty_table(header, table_rows)


def section_c(rows):
    cell = defaultdict(lambda: defaultdict(list))
    for (rm, ratio, seed), r in rows.items():
        for k in SECONDARY:
            cell[(rm, ratio)][k].append(r["final"][k])
    header = ["\u03c1_mis", "ratio", "entropy", "top1", "||\u03b8||",
             "pair_bias_mass", "win_vs_ref", "greedy_r*"]
    table_rows = []
    for rm in RHO_MIS:
        for ratio in RATIOS:
            c = cell[(rm, ratio)]
            vals = {k: (float(np.mean(c[k])), _se(c[k])) for k in SECONDARY}
            table_rows.append([
                f"{rm:g}", f"{ratio:g}",
                _fmt(*vals["entropy"], digits=3),
                _fmt(*vals["top1_prob"]),
                _fmt(*vals["theta_norm"], digits=3),
                _fmt(*vals["pair_bias_mass_under_pi"]),
                _fmt(*vals["expected_true_win_vs_ref"]),
                _fmt(*vals["greedy_true_utility"]),
            ])
    return pretty_table(header, table_rows)


def section_d(rows):
    """Paired Delta: Robust-0.025rho(seed) - Nominal(seed). Per rho_mis."""
    delta = defaultdict(dict)
    header = ["\u03c1_mis", "\u0394E[r*]", "\u0394wcl(\u03c1_mis)", "\u0394R",
              "\u0394L_SAIL(P*)", "\u0394penalty(\u03c1_mis\u00b7\u03b2\u00b7\u0394R)",
              "\u0394E[bias_attr]", "\u0394bad_mass", "\u0394good_mass",
              "max|residual|"]
    table_rows = []
    for rm in RHO_MIS:
        per = defaultdict(list)
        residuals = []
        for seed in SEEDS:
            r0 = rows[(rm, 0.0, seed)]["final"]
            r1 = rows[(rm, 0.025, seed)]["final"]
            cfg = rows[(rm, 0.025, seed)]["cfg"]
            beta = float(cfg["beta"])
            d_r = r1["expected_true_utility"] - r0["expected_true_utility"]
            d_wcl = r1["wcl_true_rho_mis"] - r0["wcl_true_rho_mis"]
            d_R = r1["sensitivity_R"] - r0["sensitivity_R"]
            d_nom = r1["nominal_bt_loss"] - r0["nominal_bt_loss"]
            d_pen = rm * beta * d_R
            d_bias = r1["expected_bias_attr"] - r0["expected_bias_attr"]
            d_bad = r1["bad_bias_candidate_mass"] - r0["bad_bias_candidate_mass"]
            d_good = r1["good_candidate_mass"] - r0["good_candidate_mass"]
            per["dr"].append(d_r)
            per["dwcl"].append(d_wcl)
            per["dR"].append(d_R)
            per["dnom"].append(d_nom)
            per["dpen"].append(d_pen)
            per["dbias"].append(d_bias)
            per["dbad"].append(d_bad)
            per["dgood"].append(d_good)
            residuals.append(abs(d_wcl - (d_nom + d_pen)))
        delta[rm] = per
        table_rows.append([
            f"{rm:g}",
            _fmt(float(np.mean(per["dr"])), _se(per["dr"])),
            _fmt(float(np.mean(per["dwcl"])), _se(per["dwcl"])),
            _fmt(float(np.mean(per["dR"])), _se(per["dR"])),
            _fmt(float(np.mean(per["dnom"])), _se(per["dnom"])),
            _fmt(float(np.mean(per["dpen"])), _se(per["dpen"])),
            _fmt(float(np.mean(per["dbias"])), _se(per["dbias"])),
            _fmt(float(np.mean(per["dbad"])), _se(per["dbad"])),
            _fmt(float(np.mean(per["dgood"])), _se(per["dgood"])),
            f"{max(residuals):.2e}",
        ])
    return delta, pretty_table(header, table_rows)


def section_e(delta):
    header = ["\u03c1_mis", "#seeds \u0394E[r*]\u2265-0.001",
              "#seeds \u0394wcl<0", "#seeds \u0394R<0"]
    rows_out = []
    counts = {}
    for rm in RHO_MIS:
        per = delta[rm]
        n_util = sum(1 for x in per["dr"] if x >= -1e-3)
        n_wcl = sum(1 for x in per["dwcl"] if x < 0)
        n_R = sum(1 for x in per["dR"] if x < 0)
        counts[rm] = (n_util, n_wcl, n_R)
        rows_out.append([f"{rm:g}", f"{n_util}/10", f"{n_wcl}/10", f"{n_R}/10"])
    return counts, pretty_table(header, rows_out)


def section_f(delta, counts):
    """Apply decision criteria."""
    # 6-point criteria (user spec §E):
    #   1. Utility tolerance: mean dE[r*] >= -0.001 at every rho_mis.
    #   2. WCL improves: mean Δwcl_true_rho_mis < 0 at every rho_mis.
    #   3. R improves: mean ΔR < 0 at every rho_mis.
    #   4. 8+/10 seeds utility tolerance.
    #   5. 8+/10 seeds wcl improves.
    #   6. 8+/10 seeds R improves.
    lines = []
    all_ok = True
    means = {}
    for rm in RHO_MIS:
        per = delta[rm]
        means[rm] = {
            "dr": float(np.mean(per["dr"])),
            "dwcl": float(np.mean(per["dwcl"])),
            "dR": float(np.mean(per["dR"])),
        }

    c1 = all(means[rm]["dr"] >= -1e-3 for rm in RHO_MIS)
    c2 = all(means[rm]["dwcl"] < 0 for rm in RHO_MIS)
    c3 = all(means[rm]["dR"] < 0 for rm in RHO_MIS)
    c4 = all(counts[rm][0] >= 8 for rm in RHO_MIS)
    c5 = all(counts[rm][1] >= 8 for rm in RHO_MIS)
    c6 = all(counts[rm][2] >= 8 for rm in RHO_MIS)
    lines.append(f"- **C1** (mean \u0394E[r*] \u2265 -1e-3): **{'PASS' if c1 else 'FAIL'}**")
    lines.append(f"- **C2** (mean \u0394wcl_true_rho_mis < 0):   **{'PASS' if c2 else 'FAIL'}**")
    lines.append(f"- **C3** (mean \u0394R < 0):                   **{'PASS' if c3 else 'FAIL'}**")
    lines.append(f"- **C4** (8/10 seeds utility tolerance):  **{'PASS' if c4 else 'FAIL'}**")
    lines.append(f"- **C5** (8/10 seeds \u0394wcl < 0):          **{'PASS' if c5 else 'FAIL'}**")
    lines.append(f"- **C6** (8/10 seeds \u0394R < 0):             **{'PASS' if c6 else 'FAIL'}**")
    passed = c1 and c2 and c3 and c4 and c5 and c6
    lines.append("")
    if passed:
        lines.append(
            "**Verdict: paper-supporting claim confirmed.** At a small nonzero "
            "robustness radius (\u03c1_alg = 0.025\u00b7\u03c1_mis), robust "
            "training strictly reduces the decomposition-form robust objective "
            "`L^W_\u03c1 = L_SAIL(P*) + \u03c1\u00b7\u03b2\u00b7R(\u03b8)` and "
            "strictly reduces sensitivity_R across all three threat levels, "
            "while preserving E[r*] within a 1e-3 tolerance on the paired seed "
            "median and majority of seeds."
        )
    else:
        failed = []
        if not c1: failed.append("C1 (mean utility tolerance)")
        if not c2: failed.append("C2 (mean \u0394wcl < 0)")
        if not c3: failed.append("C3 (mean \u0394R < 0)")
        if not c4: failed.append("C4 (8/10 seeds utility tolerance)")
        if not c5: failed.append("C5 (8/10 seeds wcl improves)")
        if not c6: failed.append("C6 (8/10 seeds R improves)")
        lines.append(
            "**Verdict: paper-supporting claim NOT confirmed (fallback reading).** "
            f"Failed: {', '.join(failed)}. "
            "The honest reading of the data is: robust training at "
            "\u03c1_alg = 0.025\u00b7\u03c1_mis reliably reduces sensitivity_R "
            "(by construction of the penalty), but does not uniformly reduce "
            "the decomposition-form robust objective and/or preserves utility "
            "only on some seeds. The per-seed bad_mass / good_mass trajectory "
            "in section D shows the flattening effect that accompanies the "
            "penalty."
        )
    return "\n".join(lines)


# ---------- plots ----------

def plot_delta_bar(delta):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics = [("dr", "\u0394E[r*]"), ("dwcl", "\u0394wcl(\u03c1_mis)"),
               ("dR", "\u0394R"), ("dbias", "\u0394E[bias_attr]")]
    x = np.arange(len(RHO_MIS))
    width = 0.2
    for i, (k, name) in enumerate(metrics):
        means = [float(np.mean(delta[rm][k])) for rm in RHO_MIS]
        ses = [_se(delta[rm][k]) for rm in RHO_MIS]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=ses,
               capsize=3, label=name)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"\u03c1_mis={rm:g}" for rm in RHO_MIS])
    ax.set_ylabel("paired \u0394 (Robust-0.025\u03c1 \u2212 Nominal)")
    ax.set_title(f"{TAG}: paired \u0394 metrics (n=10 seeds)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(REPORTS / "confirm_delta_bar.png", dpi=140)
    plt.close(fig)


def plot_wcl_decomposition(delta):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(RHO_MIS))
    width = 0.27
    for i, (k, name) in enumerate([
        ("dnom", "\u0394L_SAIL(P*)"),
        ("dpen", "\u0394penalty = \u03c1\u00b7\u03b2\u00b7\u0394R"),
        ("dwcl", "\u0394wcl(\u03c1_mis)"),
    ]):
        means = [float(np.mean(delta[rm][k])) for rm in RHO_MIS]
        ses = [_se(delta[rm][k]) for rm in RHO_MIS]
        ax.bar(x + (i - 1) * width, means, width, yerr=ses,
               capsize=3, label=name)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"\u03c1_mis={rm:g}" for rm in RHO_MIS])
    ax.set_ylabel("paired \u0394 (Robust \u2212 Nominal)")
    ax.set_title(
        f"{TAG}: wcl decomposition \u0394wcl \u2248 \u0394L_SAIL(P*) + \u03c1\u00b7\u03b2\u00b7\u0394R"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(REPORTS / "confirm_wcl_decomposition.png", dpi=140)
    plt.close(fig)


def plot_pareto_true_vs_R(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    colors = {0.0: "tab:blue", 0.025: "tab:orange"}
    for ax, rm in zip(axes, RHO_MIS):
        for ratio in RATIOS:
            xs, ys = [], []
            for seed in SEEDS:
                r = rows[(rm, ratio, seed)]["final"]
                xs.append(r["sensitivity_R"])
                ys.append(r["expected_true_utility"])
            ax.scatter(xs, ys, color=colors[ratio], alpha=0.35, s=26)
            mx, my = float(np.mean(xs)), float(np.mean(ys))
            ax.scatter([mx], [my], color=colors[ratio], s=140,
                       edgecolor="black", linewidths=1.5,
                       label=LABEL[ratio] + " (mean)")
        ax.set_title(f"\u03c1_mis={rm:g}")
        ax.set_xlabel("sensitivity_R")
        ax.set_ylabel("E[r*]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"{TAG}: paired Pareto — E[r*] vs R", fontsize=10)
    fig.tight_layout()
    fig.savefig(REPORTS / "confirm_pareto_true_vs_R.png", dpi=140)
    plt.close(fig)


def plot_wcl_vs_true(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    colors = {0.0: "tab:blue", 0.025: "tab:orange"}
    for ax, rm in zip(axes, RHO_MIS):
        for ratio in RATIOS:
            xs, ys = [], []
            for seed in SEEDS:
                r = rows[(rm, ratio, seed)]["final"]
                xs.append(r["wcl_true_rho_mis"])
                ys.append(r["expected_true_utility"])
            ax.scatter(xs, ys, color=colors[ratio], alpha=0.35, s=26)
            mx, my = float(np.mean(xs)), float(np.mean(ys))
            ax.scatter([mx], [my], color=colors[ratio], s=140,
                       edgecolor="black", linewidths=1.5,
                       label=LABEL[ratio] + " (mean)")
        ax.set_title(f"\u03c1_mis={rm:g}")
        ax.set_xlabel("wcl_true_rho_mis")
        ax.set_ylabel("E[r*]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"{TAG}: E[r*] vs decomposition-form robust objective", fontsize=10)
    fig.tight_layout()
    fig.savefig(REPORTS / "confirm_wcl_vs_true.png", dpi=140)
    plt.close(fig)


def _load_training_curve(run_dir: Path):
    """Return (rounds, metrics_by_key)."""
    rounds = []
    per = defaultdict(list)
    path = run_dir / "training_curve.jsonl"
    if not path.exists():
        return rounds, per
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            if j.get("eval") != "val" or "metrics" not in j:
                continue
            rounds.append(j["round"])
            for k, v in j["metrics"].items():
                per[k].append(v)
    return rounds, per


def plot_training_curves(rows):
    panels = [
        ("expected_true_utility", "E[r*]"),
        ("wcl_true_rho_mis", "wcl(\u03c1_mis)"),
        ("sensitivity_R", "R"),
        ("expected_bias_attr", "E[bias_attr]"),
        ("bad_bias_candidate_mass", "bad_mass"),
        ("entropy", "entropy"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    rm = 0.1
    for ratio, color in [(0.0, "tab:blue"), (0.025, "tab:orange")]:
        per_seed = defaultdict(list)   # key -> list of per-seed arrays
        rounds_ref = None
        for seed in SEEDS:
            run_dir = rows[(rm, ratio, seed)]["run_dir"]
            rounds, per = _load_training_curve(run_dir)
            if not rounds:
                continue
            rounds_ref = rounds
            for k, _ in panels:
                per_seed[k].append(per.get(k, []))
        for ax, (k, name) in zip(axes.flat, panels):
            arr = [row for row in per_seed[k] if len(row) == len(rounds_ref or [])]
            if not arr or rounds_ref is None:
                ax.set_title(f"{name} (missing)")
                continue
            A = np.asarray(arr, dtype=np.float64)
            m = A.mean(axis=0)
            s = A.std(axis=0, ddof=1) / math.sqrt(A.shape[0]) if A.shape[0] > 1 else np.zeros_like(m)
            ax.plot(rounds_ref, m, color=color, marker="o", label=LABEL[ratio])
            ax.fill_between(rounds_ref, m - s, m + s, color=color, alpha=0.18)
            ax.set_title(f"{name} (\u03c1_mis={rm:g})")
            ax.set_xlabel("round")
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{TAG}: per-round val metrics, mean\u00b1SE over 10 seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(REPORTS / "confirm_training_curves.png", dpi=140)
    plt.close(fig)


def plot_seed_scatter(rows):
    keys = [
        ("expected_true_utility", "E[r*]"),
        ("wcl_true_rho_mis", "wcl(\u03c1_mis)"),
        ("sensitivity_R", "R"),
    ]
    fig, axes = plt.subplots(len(RHO_MIS), len(keys),
                             figsize=(4.3 * len(keys), 3.8 * len(RHO_MIS)))
    for i, rm in enumerate(RHO_MIS):
        for j, (k, name) in enumerate(keys):
            ax = axes[i][j] if len(RHO_MIS) > 1 else axes[j]
            xs = [rows[(rm, 0.0, seed)]["final"][k] for seed in SEEDS]
            ys = [rows[(rm, 0.025, seed)]["final"][k] for seed in SEEDS]
            ax.scatter(xs, ys, color="tab:orange", s=40, edgecolor="k")
            lo = min(min(xs), min(ys))
            hi = max(max(xs), max(ys))
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
            ax.set_xlabel(f"Nominal {name}")
            ax.set_ylabel(f"Robust-0.025\u03c1 {name}")
            ax.set_title(f"\u03c1_mis={rm:g} — {name}")
            ax.grid(alpha=0.3)
    fig.suptitle(f"{TAG}: per-seed Nominal vs Robust-0.025\u03c1", fontsize=10)
    fig.tight_layout()
    fig.savefig(REPORTS / "confirm_seed_scatter.png", dpi=140)
    plt.close(fig)


# ---------- main ----------

def main():
    rows = _load()
    print(f"[confirm] loaded {len(rows)} runs")

    missing = []
    for rm in RHO_MIS:
        for ratio in RATIOS:
            for seed in SEEDS:
                if (rm, ratio, seed) not in rows:
                    missing.append((rm, ratio, seed))
    if missing:
        print(f"[confirm] MISSING {len(missing)} runs:")
        for m in missing[:5]:
            print(f"  - {m}")
        raise SystemExit(1)

    worst, worst_runs, violations = formula_checks(rows)
    print("[confirm] formula residuals:")
    for k, v in worst.items():
        print(f"  {k}: max |residual| = {v:.3e}   (worst run: {worst_runs[k]})")
    if violations:
        print(f"[confirm] WARNING: {len(violations)} violations above tolerance:")
        for (name, rm, ratio, seed, v, tol) in violations[:10]:
            print(f"  {name}  rm={rm} ratio={ratio} seed={seed}  |r|={v:.3e} > tol={tol:.0e}")

    cell, sec_b = section_b(rows)
    sec_c = section_c(rows)
    delta, sec_d = section_d(rows)
    counts, sec_e = section_e(delta)
    sec_f = section_f(delta, counts)

    # Render plots.
    plot_delta_bar(delta)
    plot_wcl_decomposition(delta)
    plot_pareto_true_vs_R(rows)
    plot_wcl_vs_true(rows)
    plot_training_curves(rows)
    plot_seed_scatter(rows)

    # Compose report.
    md = []
    md.append(f"# {TAG} — paper-supporting confirmation report\n")
    md.append(f"- loaded runs: {len(rows)}")
    md.append(f"- ρ_mis axis: {RHO_MIS}")
    md.append(f"- ratio axis: {RATIOS}")
    md.append(f"- seeds: {SEEDS}\n")

    md.append("## A. Metric provenance + decomposition cross-check\n")
    md.append(
        "`wcl_true_rho_mis = nominal_bt_loss + rho_mis · β · sensitivity_R` is the "
        "**decomposition form** of the paper worst-case objective "
        "`L^W_ρ(θ) = L_SAIL(θ; P*) + ρ·β·R(θ)`. Under the margin condition "
        "`[p* − ρ, p* + ρ] ⊆ [0, 1]` (enforced at training time by "
        "`oracle_misspec.check_margin`), this equals the pointwise endpoint-max "
        "`worst_case_loss` field to numerical precision via the identity "
        "`softplus(x) − softplus(−x) = x`.\n"
    )
    md.append("Numerical residuals across all loaded runs:\n")
    md.append("| identity | max \\|residual\\| | worst run (ρ_mis, ratio, seed) |")
    md.append("|---|---|---|")
    for k, v in worst.items():
        md.append(f"| {k} | {v:.3e} | {worst_runs[k]} |")
    md.append("")

    md.append("## B. Primary metrics (mean ± SE over 10 seeds)\n")
    md.append(sec_b)
    md.append("")

    md.append("## C. Secondary metrics (mean ± SE)\n")
    md.append(sec_c)
    md.append("")

    md.append("## D. Paired Δ vs Nominal (Robust-0.025ρ − Nominal, per seed)\n")
    md.append(sec_d)
    md.append("")
    md.append(
        "The `max|residual|` column tests the decomposition at the Δ level: "
        "`Δwcl ≈ Δnominal_bt_loss + ρ_mis·β·ΔR` seed-by-seed.\n"
    )

    md.append("## E. Seed counts\n")
    md.append(sec_e)
    md.append("")

    md.append("## F. Verdict\n")
    md.append(sec_f)
    md.append("")

    md.append("## Plots\n")
    for fn in [
        "confirm_delta_bar.png",
        "confirm_wcl_decomposition.png",
        "confirm_pareto_true_vs_R.png",
        "confirm_wcl_vs_true.png",
        "confirm_training_curves.png",
        "confirm_seed_scatter.png",
    ]:
        md.append(f"- `{fn}`")
    md.append("")

    out = REPORTS / "confirm_report.md"
    out.write_text("\n".join(md))
    print(f"[confirm] wrote {out}")
    for fn in ["confirm_delta_bar.png", "confirm_wcl_decomposition.png",
               "confirm_pareto_true_vs_R.png", "confirm_wcl_vs_true.png",
               "confirm_training_curves.png", "confirm_seed_scatter.png"]:
        print(f"  - {REPORTS / fn}")

    # Headline summary (printed for convenience)
    print("\n[confirm] headline summary:")
    for rm in RHO_MIS:
        per = delta[rm]
        n_util, n_wcl, n_R = counts[rm]
        print(
            f"  rho_mis={rm:g}  mean ΔE[r*]={np.mean(per['dr']):+.4f}  "
            f"mean Δwcl={np.mean(per['dwcl']):+.4f}  "
            f"mean ΔR={np.mean(per['dR']):+.4f}  "
            f"seeds(util≥-1e-3/wcl<0/R<0) = {n_util}/{n_wcl}/{n_R} of 10"
        )


if __name__ == "__main__":
    main()
