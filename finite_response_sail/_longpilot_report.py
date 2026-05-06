"""Build the 4 longpilot plots and print the 3-part numeric summary.

Reads results/v3_pairbias_longpilot/*/metrics.json and training_curve.jsonl.
"""
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = "v3_pairbias_longpilot"
RESULTS = Path(__file__).resolve().parent / "results" / TAG
REPORTS = RESULTS / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

RATIOS = [0.0, 0.025, 0.05, 0.1, 0.25]
RHO_MIS = [0.05, 0.075, 0.1]
SEEDS = [0, 1, 2]

KEYS = [
    "expected_true_utility",
    "sensitivity_R",
    "expected_bias_attr",
    "entropy",
    "top1_prob",
    "theta_norm",
    "bad_bias_candidate_mass",
    "good_candidate_mass",
]


def _load():
    # rows[(rho_mis, ratio, seed)] = metrics dict (test split)
    rows = {}
    for p in sorted(RESULTS.glob("mode-*/metrics.json")):
        d = json.loads(p.read_text())
        c = d["config"]
        key = (float(c["rho_mis"]), float(c["rho_alg_ratio"]), int(c["seed"]))
        rows[key] = d["final"]["test"]
    return rows


def _se(vals):
    v = np.asarray(vals, dtype=np.float64)
    if len(v) <= 1:
        return 0.0
    return float(v.std(ddof=1) / math.sqrt(len(v)))


def _fmt(m, s, digits=4):
    return f"{m:.{digits}f}±{s:.{digits}f}"


def summarize(rows):
    # cell[(rho_mis, ratio)][key] -> list over seeds
    cell = defaultdict(lambda: defaultdict(list))
    for (rm, ratio, seed), m in rows.items():
        for k in KEYS:
            cell[(rm, ratio)][k].append(m[k])

    # --- §1: per rho_mis, per ratio, mean±SE (paired seeds)
    print("\n=== §1: mean ± stderr over seeds (n=3) for each (rho_mis, ratio) ===")
    hdr = ["rho_mis", "ratio", "E[r*]", "R", "E[bias_attr]", "entropy", "top1", "theta_norm", "bad_mass", "good_mass"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for rm in RHO_MIS:
        for ratio in RATIOS:
            c = cell[(rm, ratio)]
            vals = lambda k: (np.mean(c[k]), _se(c[k]))
            m_r, s_r = vals("expected_true_utility")
            m_R, s_R = vals("sensitivity_R")
            m_b, s_b = vals("expected_bias_attr")
            m_e, s_e = vals("entropy")
            m_t, s_t = vals("top1_prob")
            m_n, s_n = vals("theta_norm")
            m_bd, s_bd = vals("bad_bias_candidate_mass")
            m_gd, s_gd = vals("good_candidate_mass")
            print(f"| {rm:g} | {ratio:g} | {_fmt(m_r, s_r)} | {_fmt(m_R, s_R)} | "
                  f"{_fmt(m_b, s_b)} | {_fmt(m_e, s_e, 3)} | {_fmt(m_t, s_t)} | "
                  f"{_fmt(m_n, s_n, 3)} | {_fmt(m_bd, s_bd)} | {_fmt(m_gd, s_gd)} |")

    # --- §2: Delta vs Nominal per (rho_mis, ratio). Paired seed deltas.
    print("\n=== §2: paired \u0394 vs Nominal (per seed, then mean±SE over seeds, n=3) ===")
    hdr2 = ["rho_mis", "ratio", "\u0394E[r*]", "\u0394R", "\u0394E[bias_attr]", "\u0394bad_mass", "\u0394good_mass"]
    print("| " + " | ".join(hdr2) + " |")
    print("|" + "|".join(["---"] * len(hdr2)) + "|")
    delta_store = defaultdict(dict)  # (rm, ratio) -> dict of arrays
    for rm in RHO_MIS:
        base = []
        for seed in SEEDS:
            base.append(rows[(rm, 0.0, seed)])
        for ratio in RATIOS:
            if ratio == 0.0:
                continue
            dr, dR, db, dbd, dgd = [], [], [], [], []
            for seed in SEEDS:
                m = rows[(rm, ratio, seed)]
                mb = rows[(rm, 0.0, seed)]
                dr.append(m["expected_true_utility"] - mb["expected_true_utility"])
                dR.append(m["sensitivity_R"] - mb["sensitivity_R"])
                db.append(m["expected_bias_attr"] - mb["expected_bias_attr"])
                dbd.append(m["bad_bias_candidate_mass"] - mb["bad_bias_candidate_mass"])
                dgd.append(m["good_candidate_mass"] - mb["good_candidate_mass"])
            delta_store[(rm, ratio)]["dr"] = dr
            delta_store[(rm, ratio)]["dR"] = dR
            delta_store[(rm, ratio)]["db"] = db
            print(f"| {rm:g} | {ratio:g} | "
                  f"{_fmt(np.mean(dr), _se(dr))} | "
                  f"{_fmt(np.mean(dR), _se(dR))} | "
                  f"{_fmt(np.mean(db), _se(db))} | "
                  f"{_fmt(np.mean(dbd), _se(dbd))} | "
                  f"{_fmt(np.mean(dgd), _se(dgd))} |")

    # --- §3: Pareto wins for ratios in {0.025, 0.05, 0.1}: beats/matches Nominal
    # on E[r*] while reducing R. Use paired seeds: count prompts where robust >= nominal.
    print("\n=== §3: does any ratio \u2208 {0.025, 0.05, 0.1} match/beat Nominal on E[r*] AND reduce R? ===")
    hdr3 = ["rho_mis", "ratio", "mean \u0394E[r*]", "SE(\u0394E[r*])", "mean \u0394R", "SE(\u0394R)", "seeds_matched (\u0394r \u2265 -0.001)", "verdict"]
    print("| " + " | ".join(hdr3) + " |")
    print("|" + "|".join(["---"] * len(hdr3)) + "|")
    MATCH_EPS = 1e-3  # 0.001 tolerance for "matches" on utility
    for rm in RHO_MIS:
        for ratio in [0.025, 0.05, 0.1]:
            dr = delta_store[(rm, ratio)]["dr"]
            dR = delta_store[(rm, ratio)]["dR"]
            matched = sum(1 for x in dr if x >= -MATCH_EPS)
            reduces_R_mean = np.mean(dR) < 0
            matches_or_beats_mean = np.mean(dr) >= -MATCH_EPS
            pareto = "YES" if (matches_or_beats_mean and reduces_R_mean) else ("close" if matches_or_beats_mean else "no")
            print(f"| {rm:g} | {ratio:g} | {np.mean(dr):+.4f} | {_se(dr):.4f} | "
                  f"{np.mean(dR):+.4f} | {_se(dR):.4f} | {matched}/3 | {pareto} |")
    return cell, delta_store


def plot_delta_vs_ratio(delta_store):
    # delta_true_vs_ratio.png
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {0.05: "tab:blue", 0.075: "tab:orange", 0.1: "tab:green"}
    nonzero_ratios = [r for r in RATIOS if r != 0.0]
    for rm in RHO_MIS:
        xs = nonzero_ratios
        ys = [np.mean(delta_store[(rm, r)]["dr"]) for r in xs]
        es = [_se(delta_store[(rm, r)]["dr"]) for r in xs]
        ax.errorbar(xs, ys, yerr=es, marker="o", label=f"\u03c1_mis={rm:g}",
                    color=colors[rm], capsize=3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("\u03c1_alg_ratio")
    ax.set_ylabel("\u0394 E[r*]  (Robust - Nominal)")
    ax.set_title(f"{TAG}: paired \u0394 E[r*] vs Nominal (n=3 seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS / "delta_true_vs_ratio.png", dpi=140)
    plt.close(fig)

    # delta_R_vs_ratio.png
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rm in RHO_MIS:
        xs = nonzero_ratios
        ys = [np.mean(delta_store[(rm, r)]["dR"]) for r in xs]
        es = [_se(delta_store[(rm, r)]["dR"]) for r in xs]
        ax.errorbar(xs, ys, yerr=es, marker="o", label=f"\u03c1_mis={rm:g}",
                    color=colors[rm], capsize=3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("\u03c1_alg_ratio")
    ax.set_ylabel("\u0394 sensitivity_R  (Robust - Nominal)")
    ax.set_title(f"{TAG}: paired \u0394 R vs Nominal (n=3 seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS / "delta_R_vs_ratio.png", dpi=140)
    plt.close(fig)


def plot_pareto(cell):
    # true_vs_R_pareto_longpilot.png — per rho_mis, plot mean across seeds
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=False)
    cmap = plt.get_cmap("viridis")
    for ax, rm in zip(axes, RHO_MIS):
        for ratio in RATIOS:
            c = cell[(rm, ratio)]
            m_r = np.mean(c["expected_true_utility"])
            s_r = _se(c["expected_true_utility"])
            m_R = np.mean(c["sensitivity_R"])
            s_R = _se(c["sensitivity_R"])
            color = cmap(ratio / max(RATIOS))
            ax.errorbar(m_R, m_r, xerr=s_R, yerr=s_r,
                        fmt="o", color=color, markersize=8,
                        markeredgecolor="black", capsize=3,
                        label=f"r={ratio:g}")
        ax.set_title(f"\u03c1_mis={rm:g}")
        ax.set_xlabel("sensitivity_R")
        ax.set_ylabel("E[r*]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"{TAG}: true utility vs sensitivity_R (mean \u00b1 SE over 3 seeds)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(REPORTS / "true_vs_R_pareto_longpilot.png", dpi=140)
    plt.close(fig)


def plot_training_curves():
    # training_curves_longpilot.png — curves for rho_mis=0.1 across all 5 ratios,
    # seed 0 (one line per ratio) in 5 metrics panels.
    metrics = [
        ("expected_true_utility", "E[r*]"),
        ("sensitivity_R", "sensitivity_R"),
        ("expected_bias_attr", "E[bias_attr]"),
        ("bad_bias_candidate_mass", "bad_mass"),
        ("good_candidate_mass", "good_mass"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.0))
    rm = 0.1
    cmap = plt.get_cmap("viridis")
    for ratio in RATIOS:
        color = cmap(ratio / max(RATIOS))
        # find all seeds, average across seeds per round for clarity
        series = defaultdict(list)  # key -> list over seeds, each is list over rounds
        rounds_ref = None
        for seed in SEEDS:
            # locate run dir
            run_dir = None
            for d in RESULTS.glob("mode-*"):
                cfg = json.loads((d / "config.json").read_text())
                if (abs(float(cfg["rho_mis"]) - rm) < 1e-9
                        and abs(float(cfg["rho_alg_ratio"]) - ratio) < 1e-9
                        and int(cfg["seed"]) == seed):
                    run_dir = d
                    break
            if run_dir is None:
                continue
            rounds = []
            cur = defaultdict(list)
            with open(run_dir / "training_curve.jsonl") as f:
                for line in f:
                    j = json.loads(line)
                    if j.get("eval") != "val" or "metrics" not in j:
                        continue
                    rounds.append(j["round"])
                    for k, _ in metrics:
                        cur[k].append(j["metrics"].get(k))
            rounds_ref = rounds
            for k, _ in metrics:
                series[k].append(cur[k])
        for ax, (k, name) in zip(axes, metrics):
            arr = np.asarray(series[k], dtype=np.float64)  # [n_seeds, n_rounds]
            if arr.size == 0:
                continue
            m = arr.mean(axis=0)
            ax.plot(rounds_ref, m, marker="o", color=color,
                    label=f"ratio={ratio:g}")
            ax.set_title(f"{name}  (\u03c1_mis={rm:g})", fontsize=9)
            ax.set_xlabel("round")
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(REPORTS / "training_curves_longpilot.png", dpi=140)
    plt.close(fig)


def main():
    rows = _load()
    print(f"[longpilot] loaded {len(rows)} runs")
    cell, delta_store = summarize(rows)
    plot_delta_vs_ratio(delta_store)
    plot_pareto(cell)
    plot_training_curves()
    print(f"\n[longpilot] plots written to {REPORTS}")
    for name in ["delta_true_vs_ratio.png", "delta_R_vs_ratio.png",
                 "true_vs_R_pareto_longpilot.png", "training_curves_longpilot.png"]:
        print(f"  - {REPORTS / name}")


if __name__ == "__main__":
    main()
