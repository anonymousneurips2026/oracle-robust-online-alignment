"""Hard pool-acceptance gate for the finite response set (v3_pairbias).

Exits 0 if all criteria pass, 1 otherwise. Runs on response_set_train.pt by
default; any split can be selected.

Criteria:
    (A) mean unique candidates per prompt              >= 10
    (B) mean std(bias_attr) per prompt                 > 0.08  (variance floor)
    (C) mean std(true_reward) per prompt               > 0.02
    (D) fraction of prompts with high-bias low-true     >= 0.25
    (E) fraction of prompts with low-bias high-true     >= 0.25
    (F) delta_empirical(p_star) > max_rho_mis           (margin condition)
    (G) bias_active_rate > 0 and mean |b| > 0           (oracle nonvacuous)

No probability clipping is used; instead (F) directly asserts the margin.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

from .oracle_misspec import (
    compute_p_obs,
    compute_p_star,
    compute_pairwise_bias,
    empirical_margin_delta,
    normalize_per_prompt,
    oracle_diagnostics,
)


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _percentile(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q))


def _bias_attr_array(cands, bias_attr: str) -> np.ndarray:
    red = np.asarray([[c["redundancy"] for c in cs] for cs in cands], dtype=np.float64)
    ln = np.asarray([[c["length_norm"] for c in cs] for cs in cands], dtype=np.float64)
    if bias_attr == "redundancy":
        raw = red
    elif bias_attr == "length_norm":
        raw = ln
    elif bias_attr == "mixed":
        raw = 0.5 * normalize_per_prompt(red) + 0.5 * normalize_per_prompt(ln)
    else:
        raise ValueError(bias_attr)
    return normalize_per_prompt(raw)


def check(payload, bias_attr: str, tau: float, max_rho_mis: float, verbose: bool = True):
    cands = payload["candidates"]
    N = len(cands)
    if N == 0:
        return False, {"error": "empty split"}

    uniq = []
    bias_std = []
    rew_std = []
    high_bias_low_true = 0
    low_bias_high_true = 0

    true_r = np.asarray(
        [[c["true_reward"] for c in cs] for cs in cands], dtype=np.float64,
    )
    ba = _bias_attr_array(cands, bias_attr)

    for n, cs in enumerate(cands):
        texts = [_norm_text(c["text"]) for c in cs]
        uniq.append(len(set(texts)))
        s = ba[n]
        r = true_r[n]
        bias_std.append(s.std())
        rew_std.append(r.std())
        if s.size < 2:
            continue
        s75 = _percentile(s, 75)
        s25 = _percentile(s, 25)
        r75 = _percentile(r, 75)
        r25 = _percentile(r, 25)
        if ((s >= s75) & (r <= r25)).any():
            high_bias_low_true += 1
        if ((s <= s25) & (r >= r75)).any():
            low_bias_high_true += 1

    p_star = compute_p_star(true_r, tau=tau)
    b = compute_pairwise_bias(true_r, ba)
    # Pick the largest rho_mis in the planned grid for the margin check.
    p_obs = compute_p_obs(p_star, b, rho_mis=max_rho_mis)
    delta = empirical_margin_delta(p_star)
    odiag = oracle_diagnostics(p_star, p_obs, b, ba, true_r)

    report = {
        "N": N,
        "K": len(cands[0]),
        "bias_attr": bias_attr,
        "tau": tau,
        "max_rho_mis": max_rho_mis,
        "mean_unique_per_prompt": float(np.mean(uniq)),
        "mean_bias_std_per_prompt": float(np.mean(bias_std)),
        "mean_reward_std_per_prompt": float(np.mean(rew_std)),
        "high_bias_low_true_frac": high_bias_low_true / N,
        "low_bias_high_true_frac": low_bias_high_true / N,
        "delta_empirical": odiag["delta_empirical"],
        "bias_active_rate": odiag["bias_active_rate"],
        "mean_abs_b": odiag["mean_abs_b"],
    }
    checks = {
        "A_unique_ge_10": report["mean_unique_per_prompt"] >= 10.0,
        "B_bias_std_gt_0_08": report["mean_bias_std_per_prompt"] > 0.08,
        "C_reward_std_gt_0_02": report["mean_reward_std_per_prompt"] > 0.02,
        "D_high_bias_low_true_ge_0_25": report["high_bias_low_true_frac"] >= 0.25,
        "E_low_bias_high_true_ge_0_25": report["low_bias_high_true_frac"] >= 0.25,
        "F_delta_gt_max_rho_mis": report["delta_empirical"] > max_rho_mis,
        "G_bias_active": (report["bias_active_rate"] > 0.0) and (report["mean_abs_b"] > 0.0),
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())

    if verbose:
        def _tag(ok): return "PASS" if ok else "FAIL"
        print(f"[pool] N={report['N']}  K={report['K']}  bias_attr={bias_attr}  tau={tau}")
        print(f"  (A) mean_unique_per_prompt     = {report['mean_unique_per_prompt']:.2f}   (ge 10 -> {_tag(checks['A_unique_ge_10'])})")
        print(f"  (B) mean_bias_std_per_prompt   = {report['mean_bias_std_per_prompt']:.4f}  (> 0.08 -> {_tag(checks['B_bias_std_gt_0_08'])})")
        print(f"  (C) mean_reward_std_per_prompt = {report['mean_reward_std_per_prompt']:.4f}  (> 0.02 -> {_tag(checks['C_reward_std_gt_0_02'])})")
        print(f"  (D) high_bias_low_true_frac    = {report['high_bias_low_true_frac']:.3f}   (ge 0.25 -> {_tag(checks['D_high_bias_low_true_ge_0_25'])})")
        print(f"  (E) low_bias_high_true_frac    = {report['low_bias_high_true_frac']:.3f}   (ge 0.25 -> {_tag(checks['E_low_bias_high_true_ge_0_25'])})")
        print(f"  (F) delta_empirical            = {report['delta_empirical']:.4f}  (> {max_rho_mis:g} -> {_tag(checks['F_delta_gt_max_rho_mis'])})")
        print(f"  (G) bias_active_rate           = {report['bias_active_rate']:.3f}   mean|b| = {report['mean_abs_b']:.4f}  (> 0 -> {_tag(checks['G_bias_active'])})")
        print(f"[pool] overall: {_tag(report['pass'])}")

    return report["pass"], report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="finite_response_sail/cache")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--max-rho-mis", type=float, default=0.1,
                        help="max rho_mis in the planned grid; delta_empirical must exceed it")
    parser.add_argument("--bias_attr", choices=["redundancy", "length_norm", "mixed"],
                        default="redundancy")
    parser.add_argument("--K", type=int, default=12,
                        help="response-set K to validate (must match an existing cache)")
    args = parser.parse_args()
    from .response_set import response_set_path
    path = response_set_path(Path(args.cache_dir), args.K, args.split)
    if not path.exists():
        print(f"[error] {path} not found", file=sys.stderr)
        sys.exit(2)
    payload = torch.load(path, weights_only=False)
    meta_K = payload.get("meta", {}).get("K")
    if meta_K is not None and int(meta_K) != int(args.K):
        print(f"[error] cache meta K={meta_K} != --K={args.K}", file=sys.stderr)
        sys.exit(2)
    ok, _ = check(
        payload,
        bias_attr=args.bias_attr,
        tau=args.tau,
        max_rho_mis=args.max_rho_mis,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
