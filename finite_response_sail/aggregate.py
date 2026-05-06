"""Aggregate per-run metrics.json into main_table.{csv,md} and v3 plots.

Scans results/<tag>/*/metrics.json and groups by
    (tag, mode, featureset, bias_attr, ref, objective, tau, rho_mis, rho_alg_ratio, label).

Labels are derived from rho_alg_ratio:
    Nominal         -- ratio == 0
    Robust-0.1rho   -- ratio == 0.1
    Robust-0.25rho  -- ratio == 0.25
    Robust-0.5rho   -- ratio == 0.5
    Robust-0.75rho  -- ratio == 0.75
    Robust-matched  -- ratio == 1.0

Robust-best-val is derived afterwards by picking, for each
(mode, featureset, bias_attr, ref, objective, rho_mis), the robust ratio with the highest
seed-0 val expected_true_utility, then pooling all seeds at that choice.

Plots written to results/<tag>/reports/:
    rho_alg_fine_sweep_true_utility.png
    rho_alg_fine_sweep_sensitivity.png
    pareto_true_vs_sensitivity.png
    rho_mis_stress.png
    offdiag_ablation.png
    ref_comparison.png
    bias_diagnostics.png
    training_curves_v3.png
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


ROBUST_RATIOS: Tuple[float, ...] = (0.0, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
ROBUST_LABELS: Tuple[str, ...] = (
    "Nominal",
    "Robust-0.025rho",
    "Robust-0.05rho",
    "Robust-0.1rho",
    "Robust-0.25rho",
    "Robust-0.5rho",
    "Robust-0.75rho",
    "Robust-matched",
)


def _label_for_ratio(ratio: float, eps: float = 1e-6) -> Optional[str]:
    for r, lbl in zip(ROBUST_RATIOS, ROBUST_LABELS):
        if abs(ratio - r) < eps:
            return lbl
    return None


def _method_name(objective: str, ratio: float, eps: float = 1e-6) -> str:
    robust = abs(ratio) > eps
    if objective == "sampled":
        return "Robust Online-SAIL" if robust else "Online-SAIL"
    if objective == "exact":
        return "Exact-Robust" if robust else "Exact-Nominal"
    if objective == "offdiag_exact":
        return "OffdiagExact-Robust" if robust else "OffdiagExact-Nominal"
    return f"{objective}-{'Robust' if robust else 'Nominal'}"


def _fmt_nan(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and math.isnan(x):
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _load_runs(results_dir: Path, tag: str) -> List[Dict[str, Any]]:
    tag_dir = results_dir / tag
    out: List[Dict[str, Any]] = []
    if not tag_dir.exists():
        return out
    for p in sorted(tag_dir.glob("*/metrics.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"[warn] could not parse {p}: {e}")
    return out


_PRIMARY_KEYS = [
    "expected_true_utility",
    "expected_bias_attr",
    "expected_true_win_vs_ref",
    "greedy_true_win_vs_ref",
    "greedy_true_utility",
    "greedy_bias_attr",
    "sensitivity_R",
    "worst_case_loss",
    "entropy",
    "top1_prob",
    "diagonal_mass",
    "theta_norm",
    "candidate_base_true_corr",
    "nominal_bt_loss",
    "biased_bt_loss",
    "delta_empirical",
    "mean_abs_b",
    "max_abs_b",
    "bias_active_rate",
    "bias_conflict_rate",
    "bias_same_direction_zero_rate",
    "bias_attr_true_corr",
    "bias_attr_true_corr_median",
    "pair_bias_mass_under_pi",
    "signed_pair_bias_under_pi",
    "offdiag_denominator_min",
    "offdiag_denominator_mean",
    "bad_bias_candidate_mass",
    "good_candidate_mass",
    "wcl_true_rho_mis",
    "wcl_true_rho_alg",
    "biased_feedback_loss",
]


@dataclass
class _Group:
    mode: str
    featureset: str
    bias_attr: str
    ref: str
    objective: str
    tau: float
    rho_mis: float
    rho_alg_ratio: float
    label: str

    def key(self):
        return (
            self.mode, self.featureset, self.bias_attr, self.ref, self.objective,
            self.tau, self.rho_mis, self.rho_alg_ratio, self.label,
        )


def _resolve_rho_alg_ratio(cfg: Dict[str, Any]) -> float:
    if cfg.get("rho_alg_ratio") is not None:
        return float(cfg["rho_alg_ratio"])
    rho_mis = float(cfg.get("rho_mis", 0.0))
    rho_alg = float(cfg.get("rho_alg", 0.0))
    if rho_mis <= 0:
        return 0.0
    return rho_alg / rho_mis


def _aggregate_metrics(records: List[Dict[str, Any]], split: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n_seeds": len(records),
        "seeds": sorted({r["config"]["seed"] for r in records}),
    }
    for k in _PRIMARY_KEYS:
        vals = [r["final"][split].get(k) for r in records if r["final"][split].get(k) is not None]
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        if not vals:
            out[f"{k}_mean"] = float("nan")
            out[f"{k}_std"] = float("nan")
        else:
            arr = np.asarray(vals, dtype=np.float64)
            out[f"{k}_mean"] = float(arr.mean())
            out[f"{k}_std"] = float(arr.std(ddof=0))
    return out


def _best_val_ratio(records_by_label: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    """Pick the robust label with the best seed-0 val expected_true_utility."""
    best_label, best_val = None, -math.inf
    for lbl in ROBUST_LABELS:
        if lbl == "Nominal":
            continue
        recs = records_by_label.get(lbl, [])
        s0 = [r for r in recs if r["config"]["seed"] == 0]
        if not s0:
            continue
        v = s0[0]["final"]["val"].get("expected_true_utility")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if v > best_val:
            best_val = v
            best_label = lbl
    return best_label


def build_main_table(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    by_cond: Dict[tuple, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for r in runs:
        cfg = r["config"]
        rho_mis = float(cfg.get("rho_mis", 0.0))
        ratio = _resolve_rho_alg_ratio(cfg)
        label = _label_for_ratio(ratio)
        if label is None:
            continue
        objective = cfg.get("objective", "exact")
        if objective == "offdiag":
            objective = "offdiag_exact"
        grp = _Group(
            mode=cfg["mode"], featureset=cfg["featureset"],
            bias_attr=cfg.get("bias_attr", "redundancy"),
            ref=cfg["ref"], objective=objective,
            tau=float(cfg.get("tau", 2.0)),
            rho_mis=rho_mis, rho_alg_ratio=ratio, label=label,
        )
        groups[grp.key()].append(r)
        cond = (
            cfg["mode"], cfg["featureset"], cfg.get("bias_attr", "redundancy"),
            cfg["ref"], objective, float(cfg.get("tau", 2.0)), rho_mis,
        )
        by_cond[cond][label].append(r)

    rows: List[Dict[str, Any]] = []
    for key, recs in sorted(groups.items()):
        (mode, fs, ba, ref, objective, tau, rho_mis, ratio, label) = key
        agg = _aggregate_metrics(recs, split="test")
        row = {
            "method": _method_name(objective, ratio),
            "mode": mode, "featureset": fs, "bias_attr": ba, "ref": ref,
            "objective": objective, "tau": tau,
            "rho_mis": rho_mis, "rho_alg_ratio": ratio, "rho_alg_label": label,
            **agg,
        }
        rows.append(row)

    # Robust-best-val diagnostic rows (one per condition).
    for cond, by_label in sorted(by_cond.items()):
        (mode, fs, ba, ref, objective, tau, rho_mis) = cond
        if rho_mis == 0.0:
            continue
        best = _best_val_ratio(by_label)
        if best is None:
            continue
        recs = by_label[best]
        agg = _aggregate_metrics(recs, split="test")
        cfg0 = recs[0]["config"]
        ratio0 = _resolve_rho_alg_ratio(cfg0)
        rows.append({
            "method": _method_name(objective, ratio0),
            "mode": mode, "featureset": fs, "bias_attr": ba, "ref": ref,
            "objective": objective, "tau": tau,
            "rho_mis": rho_mis, "rho_alg_ratio": ratio0,
            "rho_alg_label": f"Robust-best-val({best})",
            "tuning_diagnostic": True,
            **agg,
        })
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    cols = [
        ("method", "method"),
        ("mode", "mode"),
        ("featureset", "features"),
        ("bias_attr", "bias_attr"),
        ("ref", "ref"),
        ("objective", "obj"),
        ("rho_mis", "\u03c1_mis"),
        ("rho_alg_label", "\u03c1_alg"),
        ("n_seeds", "n"),
        ("expected_true_utility_mean", "E[r*]"),
        ("expected_bias_attr_mean", "E[bias_attr]"),
        ("expected_true_win_vs_ref_mean", "win_vs_ref"),
        ("greedy_true_utility_mean", "greedy_r*"),
        ("sensitivity_R_mean", "R"),
        ("worst_case_loss_mean", "wcl"),
        ("wcl_true_rho_mis_mean", "wcl(\u03c1_mis)"),
        ("nominal_bt_loss_mean", "L_SAIL(P*)"),
        ("delta_empirical_mean", "\u03b4_emp"),
        ("mean_abs_b_mean", "mean|b|"),
        ("bias_active_rate_mean", "bias_active"),
        ("bias_attr_true_corr_mean", "corr(s,r)"),
        ("pair_bias_mass_under_pi_mean", "pair_bias_mass"),
        ("bad_bias_candidate_mass_mean", "bad_mass"),
        ("good_candidate_mass_mean", "good_mass"),
        ("theta_norm_mean", "||\u03b8||"),
    ]
    lines = []
    lines.append("| " + " | ".join(h for _, h in cols) + " | note |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for r in sorted(
        rows,
        key=lambda x: (
            x["mode"], x["featureset"], x["bias_attr"], x["ref"],
            x["objective"], x["rho_mis"], x["rho_alg_label"],
        ),
    ):
        vals = []
        for k, _ in cols:
            v = r.get(k)
            if isinstance(v, str):
                vals.append(v)
            elif k == "n_seeds":
                vals.append(str(v))
            elif k == "rho_mis":
                vals.append(f"{v:g}")
            else:
                vals.append(_fmt_nan(v))
        note_bits = []
        if r.get("tuning_diagnostic"):
            note_bits.append("tuning diagnostic")
        note = "; ".join(note_bits) if note_bits else ""
        lines.append("| " + " | ".join(vals) + f" | {note} |")
    path.write_text("\n".join(lines) + "\n")


def _panel_key(r: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        r["mode"], r["featureset"], r["bias_attr"], r["ref"], r["objective"],
    )


def plot_rho_alg_fine_sweep(
    rows: List[Dict[str, Any]], metric_key: str, ylabel: str, out: Path,
) -> None:
    if not _HAVE_MPL:
        return
    headline = [r for r in rows if not r.get("tuning_diagnostic")]
    panels = sorted({(r["ref"], r["objective"]) for r in headline})
    if not panels:
        return
    fig, axes = plt.subplots(
        nrows=len(panels), ncols=1,
        figsize=(6, 3.1 * max(1, len(panels))), squeeze=False,
    )
    for idx, (ref, obj) in enumerate(panels):
        ax = axes[idx, 0]
        subset = [r for r in headline if r["ref"] == ref and r["objective"] == obj]
        rho_miss = sorted({r["rho_mis"] for r in subset if r["rho_mis"] > 0})
        for rm in rho_miss:
            xs, ys = [], []
            for lbl, ratio in zip(ROBUST_LABELS, ROBUST_RATIOS):
                cand = [r for r in subset if r["rho_mis"] == rm and r["rho_alg_label"] == lbl]
                if not cand:
                    continue
                xs.append(ratio)
                ys.append(cand[0].get(f"{metric_key}_mean"))
            if xs:
                order = np.argsort(xs)
                xs = [xs[i] for i in order]
                ys = [ys[i] for i in order]
                ax.plot(xs, ys, marker="o", label=f"\u03c1_mis={rm:g}")
        ax.set_title(f"ref={ref} / obj={obj}")
        ax.set_xlabel("\u03c1_alg_ratio")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_pareto(rows: List[Dict[str, Any]], out: Path) -> None:
    if not _HAVE_MPL:
        return
    headline = [r for r in rows if not r.get("tuning_diagnostic")]
    refs = sorted({r["ref"] for r in headline})
    if not refs:
        return
    fig, axes = plt.subplots(
        nrows=len(refs), ncols=1, figsize=(6, 4.0 * len(refs)), squeeze=False,
    )
    cmap = plt.get_cmap("viridis")
    obj_marker = {"exact": "o", "offdiag_exact": "s", "sampled": "^"}
    for idx, ref in enumerate(refs):
        ax = axes[idx, 0]
        subset = [r for r in headline if r["ref"] == ref]
        for r in subset:
            x = r.get("sensitivity_R_mean")
            y = r.get("expected_true_utility_mean")
            ratio = r.get("rho_alg_ratio", 0.0)
            if x is None or y is None:
                continue
            marker = obj_marker.get(r["objective"], "x")
            color = cmap(min(1.0, max(0.0, ratio)))
            ax.scatter(x, y, c=[color], marker=marker, edgecolor="black",
                       linewidth=0.3, alpha=0.8)
        ax.set_title(f"ref={ref}")
        ax.set_xlabel("sensitivity_R")
        ax.set_ylabel("E[r*]")
        ax.grid(alpha=0.3)
    # Legend for markers and colorbar.
    handles = [
        plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
                   markeredgecolor="black", label=k)
        for k, m in obj_marker.items()
    ]
    axes[0, 0].legend(handles=handles, title="objective", fontsize=7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:, 0].tolist(), label="\u03c1_alg_ratio", fraction=0.03)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_rho_mis_stress(rows: List[Dict[str, Any]], out: Path) -> None:
    if not _HAVE_MPL:
        return
    headline = [r for r in rows if not r.get("tuning_diagnostic")]
    best_rows = [r for r in rows if r.get("tuning_diagnostic")]
    panels = sorted({(r["ref"], r["objective"]) for r in headline})
    if not panels:
        return
    metrics = [
        ("expected_true_utility_mean", "E[r*]"),
        ("expected_bias_attr_mean", "E[bias_attr]"),
        ("sensitivity_R_mean", "sensitivity_R"),
    ]
    fig, axes = plt.subplots(
        nrows=len(panels), ncols=len(metrics),
        figsize=(4.0 * len(metrics), 3.0 * max(1, len(panels))), squeeze=False,
    )
    for i, (ref, obj) in enumerate(panels):
        subset = [r for r in headline if r["ref"] == ref and r["objective"] == obj]
        best_subset = [r for r in best_rows if r["ref"] == ref and r["objective"] == obj]
        for j, (key, ylabel) in enumerate(metrics):
            ax = axes[i, j]
            for lbl in ("Nominal", "Robust-0.25rho"):
                xs, ys = [], []
                for r in sorted(subset, key=lambda x: x["rho_mis"]):
                    if r["rho_alg_label"] != lbl:
                        continue
                    xs.append(r["rho_mis"])
                    ys.append(r.get(key))
                if xs:
                    ax.plot(xs, ys, marker="o", label=lbl)
            xs, ys = [], []
            for r in sorted(best_subset, key=lambda x: x["rho_mis"]):
                xs.append(r["rho_mis"])
                ys.append(r.get(key))
            if xs:
                ax.plot(xs, ys, marker="x", linestyle="--", label="Robust-best-val")
            ax.set_title(f"{ylabel}  (ref={ref}/{obj})", fontsize=8)
            ax.set_xlabel("\u03c1_mis")
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_offdiag_ablation(rows: List[Dict[str, Any]], out: Path) -> None:
    if not _HAVE_MPL:
        return
    headline = [r for r in rows if not r.get("tuning_diagnostic")]
    # match by (mode, featureset, bias_attr, ref, rho_mis, rho_alg_ratio)
    pair_key = lambda r: (
        r["mode"], r["featureset"], r["bias_attr"], r["ref"],
        r["rho_mis"], r["rho_alg_ratio"],
    )
    pairs: Dict[tuple, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in headline:
        if r["objective"] in ("exact", "offdiag_exact"):
            pairs[pair_key(r)][r["objective"]] = r
    matched = [v for v in pairs.values() if "exact" in v and "offdiag_exact" in v]
    if not matched:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, ylabel in zip(
        axes,
        ("expected_true_utility_mean", "sensitivity_R_mean"),
        ("E[r*]", "sensitivity_R"),
    ):
        xs = list(range(len(matched)))
        ex = [m["exact"].get(key) for m in matched]
        od = [m["offdiag_exact"].get(key) for m in matched]
        width = 0.4
        ax.bar([x - width / 2 for x in xs], ex, width=width, label="exact")
        ax.bar([x + width / 2 for x in xs], od, width=width, label="offdiag_exact")
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [
                f"{m['exact']['rho_mis']:g}/{m['exact']['rho_alg_ratio']:g}"
                for m in matched
            ],
            rotation=60, fontsize=6,
        )
        ax.set_xlabel("\u03c1_mis / \u03c1_alg_ratio")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_ref_comparison(rows: List[Dict[str, Any]], out: Path) -> None:
    if not _HAVE_MPL:
        return
    headline = [r for r in rows if not r.get("tuning_diagnostic")]
    metrics = [
        ("expected_true_utility_mean", "E[r*]"),
        ("expected_true_win_vs_ref_mean", "win vs ref"),
        ("sensitivity_R_mean", "sensitivity_R"),
        ("entropy_mean", "H(\u03c0)"),
        ("top1_prob_mean", "top1"),
    ]
    # Group by (objective, rho_mis, rho_alg_label) so each bar pair is matched.
    keyer = lambda r: (r["objective"], r["rho_mis"], r["rho_alg_label"])
    buckets: Dict[tuple, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in headline:
        buckets[keyer(r)][r["ref"]] = r
    matched = [v for v in buckets.values() if "uniform" in v and "base_lm_lognorm" in v]
    if not matched:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.0))
    for ax, (key, ylabel) in zip(axes, metrics):
        xs = list(range(len(matched)))
        u = [m["uniform"].get(key) for m in matched]
        b = [m["base_lm_lognorm"].get(key) for m in matched]
        width = 0.4
        ax.bar([x - width / 2 for x in xs], u, width=width, label="uniform")
        ax.bar([x + width / 2 for x in xs], b, width=width, label="base_lm_lognorm")
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_xlabel("matched conditions")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _find_representative_run(runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    preferred = {
        "mode": "biased_feedback",
        "featureset": "controlled",
        "bias_attr": "redundancy",
        "objective": "offdiag_exact",
        "ref": "base_lm_lognorm",
        "rho_mis": 0.1,
        "rho_alg_ratio": 0.0,
        "seed": 0,
    }

    def score(r: Dict[str, Any]) -> int:
        cfg = r["config"]
        ratio = _resolve_rho_alg_ratio(cfg)
        s = 0
        s += int(cfg.get("mode") == preferred["mode"])
        s += int(cfg.get("featureset") == preferred["featureset"])
        s += int(cfg.get("bias_attr") == preferred["bias_attr"])
        s += int(cfg.get("ref") == preferred["ref"])
        s += int(cfg.get("objective") in ("offdiag_exact", "offdiag"))
        s += int(abs(float(cfg.get("rho_mis", 0.0)) - preferred["rho_mis"]) < 1e-6)
        s += int(abs(ratio - preferred["rho_alg_ratio"]) < 1e-6)
        s += int(cfg.get("seed") == preferred["seed"])
        return s

    ranked = sorted(runs, key=score, reverse=True)
    return ranked[0] if ranked else None


def plot_bias_diagnostics(
    runs: List[Dict[str, Any]], results_dir: Path, tag: str, out: Path,
) -> None:
    if not _HAVE_MPL:
        return
    rep = _find_representative_run(runs)
    if rep is None:
        return
    tag_dir = results_dir / tag
    run_dir = tag_dir / rep["run_id"]
    npz_path = run_dir / "oracle_tensors_test.npz"
    if not npz_path.exists():
        return
    arr = np.load(npz_path)
    b = arr["b"]
    bias_attr = arr["bias_attr"]
    true_reward = arr["true_reward"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (1) histogram of b_ij (off-diagonal).
    K = b.shape[-1]
    off = 1.0 - np.eye(K)
    b_off = b[:, off.astype(bool)].reshape(-1)
    axes[0].hist(b_off, bins=50)
    axes[0].set_xlabel("b_ij")
    axes[0].set_ylabel("count")
    axes[0].set_title("off-diagonal pairwise bias b_ij")
    axes[0].grid(alpha=0.3)

    # (2) scatter bias_attr vs true_reward for all prompt-candidate pairs (subsample).
    s = bias_attr.reshape(-1)
    r = true_reward.reshape(-1)
    if s.size > 20000:
        idx = np.random.default_rng(0).choice(s.size, 20000, replace=False)
        s = s[idx]; r = r[idx]
    axes[1].scatter(s, r, s=2, alpha=0.3)
    axes[1].set_xlabel("bias_attr (normalized)")
    axes[1].set_ylabel("true_reward")
    axes[1].set_title("bias_attr vs true_reward")
    axes[1].grid(alpha=0.3)

    # (3) pair_bias_mass_under_pi over training rounds (reads training_curve.jsonl).
    curve_path = run_dir / "training_curve.jsonl"
    rounds = []
    masses = []
    if curve_path.exists():
        with open(curve_path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("eval") == "val" and "metrics" in d:
                    rounds.append(d.get("round", len(rounds)))
                    masses.append(d["metrics"].get("pair_bias_mass_under_pi"))
    if rounds:
        axes[2].plot(rounds, masses, marker="o")
        axes[2].set_xlabel("round")
        axes[2].set_ylabel("pair_bias_mass_under_pi (val)")
        axes[2].set_title("pair_bias_mass over training")
    else:
        axes[2].set_title("(no per-round val log)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"representative run: {rep.get('run_id', '?')}", fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_training_curves_v3(
    runs: List[Dict[str, Any]], results_dir: Path, tag: str, out: Path,
) -> None:
    if not _HAVE_MPL:
        return
    preferred = {
        "mode": "biased_feedback",
        "featureset": "controlled",
        "bias_attr": "redundancy",
        "objective": "offdiag_exact",
        "ref": "base_lm_lognorm",
        "rho_mis": 0.1,
    }

    def match(cfg: Dict[str, Any]) -> bool:
        if cfg.get("mode") != preferred["mode"]: return False
        if cfg.get("featureset") != preferred["featureset"]: return False
        if cfg.get("bias_attr") != preferred["bias_attr"]: return False
        if cfg.get("ref") != preferred["ref"]: return False
        if cfg.get("objective") not in ("offdiag_exact", "offdiag"): return False
        if abs(float(cfg.get("rho_mis", 0.0)) - preferred["rho_mis"]) > 1e-6: return False
        return True

    selected = [r for r in runs if match(r["config"])]
    if not selected:
        return

    metrics = [
        ("expected_true_utility", "E[r*]"),
        ("expected_bias_attr", "E[bias_attr]"),
        ("sensitivity_R", "sensitivity_R"),
        ("entropy", "H(\u03c0)"),
        ("theta_norm", "||\u03b8||"),
    ]

    tag_dir = results_dir / tag
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.0))
    for r in selected:
        cfg = r["config"]
        ratio = _resolve_rho_alg_ratio(cfg)
        run_dir = tag_dir / r["run_id"]
        curve_path = run_dir / "training_curve.jsonl"
        if not curve_path.exists():
            continue
        rounds = []
        series: Dict[str, List[float]] = {k: [] for k, _ in metrics}
        with open(curve_path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("eval") != "val" or "metrics" not in d:
                    continue
                m = d["metrics"]
                rounds.append(d.get("round", len(rounds)))
                for k, _ in metrics:
                    series[k].append(m.get(k))
        if not rounds:
            continue
        lbl = f"ratio={ratio:g} s{cfg.get('seed')}"
        for ax, (k, name) in zip(axes, metrics):
            ax.plot(rounds, series[k], marker="o", alpha=0.6, label=lbl)
            ax.set_title(name, fontsize=8)
            ax.set_xlabel("round")
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--results-dir", default="finite_response_sail/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    runs = _load_runs(results_dir, args.tag)
    print(f"[agg] loaded {len(runs)} runs under tag={args.tag}")
    if not runs:
        return

    rows = build_main_table(runs)
    out_dir = results_dir / args.tag / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "main_table.csv")
    write_markdown(rows, out_dir / "main_table.md")
    plot_rho_alg_fine_sweep(
        rows, "expected_true_utility", "E[r*]",
        out_dir / "rho_alg_fine_sweep_true_utility.png",
    )
    plot_rho_alg_fine_sweep(
        rows, "sensitivity_R", "sensitivity_R",
        out_dir / "rho_alg_fine_sweep_sensitivity.png",
    )
    plot_pareto(rows, out_dir / "pareto_true_vs_sensitivity.png")
    plot_rho_mis_stress(rows, out_dir / "rho_mis_stress.png")
    plot_offdiag_ablation(rows, out_dir / "offdiag_ablation.png")
    plot_ref_comparison(rows, out_dir / "ref_comparison.png")
    plot_bias_diagnostics(runs, results_dir, args.tag, out_dir / "bias_diagnostics.png")
    plot_training_curves_v3(runs, results_dir, args.tag, out_dir / "training_curves_v3.png")
    print(f"[write] {out_dir}")


if __name__ == "__main__":
    main()
