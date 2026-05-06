"""Qualitative side-by-side examples: Nominal vs Robust Online-SAIL.

For a given tag, auto-pair two runs that differ only in rho_alg (one with
rho_alg == 0 = Nominal, one with rho_alg > 0 = Robust). For each sampled
prompt, render the reference summary, the top-1 candidate under each policy,
and the top-5 expected-probability lists. Highlights where robust training
shifts mass away from spurious-favorable candidates.

Output: results/<tag>/reports/examples.md.

Usage:
    python -m finite_response_sail.export_examples \\
        --tag main_v1 --split test --n-examples 10
"""
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .features import psi_path
from .policy import policy_probs
from .response_set import response_set_path


# ---------- helpers ----------


def _load_runs_metadata(results_dir: Path, tag: str) -> List[Dict[str, Any]]:
    out = []
    tag_dir = results_dir / tag
    if not tag_dir.exists():
        return out
    for p in sorted(tag_dir.glob("*/metrics.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        d["_run_dir"] = p.parent
        out.append(d)
    return out


def _pick_nominal_robust_pair(
    runs: List[Dict[str, Any]],
    featureset: Optional[str],
    mode: Optional[str],
    objective: Optional[str],
    seed: Optional[int],
    prefer_rho_mis: Optional[float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Find a (nominal, robust) run pair matching filters, preferring smaller seed."""

    def keep(r):
        cfg = r["config"]
        if featureset is not None and cfg["featureset"] != featureset:
            return False
        if mode is not None and cfg["mode"] != mode:
            return False
        if objective is not None and cfg.get("objective", "exact") != objective:
            return False
        if seed is not None and cfg["seed"] != seed:
            return False
        return True

    filtered = [r for r in runs if keep(r)]
    if not filtered:
        raise RuntimeError(
            "No runs match filters; check --featureset/--mode/--objective/--seed"
        )

    # Group by (featureset, mode, objective, ref, rho_mis, seed)
    key = lambda r: (
        r["config"]["featureset"], r["config"]["mode"],
        r["config"].get("objective", "exact"), r["config"]["ref"],
        float(r["config"]["rho_mis"]), int(r["config"]["seed"]),
    )
    by_key: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in filtered:
        by_key.setdefault(key(r), []).append(r)

    # Find a group with both rho_alg==0 and rho_alg>0, prefer prefer_rho_mis if set.
    candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for k, rs in by_key.items():
        nom = [r for r in rs if float(r["config"]["rho_alg"]) == 0.0]
        rob = [r for r in rs if float(r["config"]["rho_alg"]) > 0.0]
        if nom and rob:
            rob_sorted = sorted(rob, key=lambda r: float(r["config"]["rho_alg"]))
            candidates.append((nom[0], rob_sorted[0]))
    if not candidates:
        raise RuntimeError(
            "No (nominal, robust) pair found: every group is missing one side."
        )
    if prefer_rho_mis is not None:
        candidates.sort(
            key=lambda pr: abs(float(pr[0]["config"]["rho_mis"]) - prefer_rho_mis)
        )
    else:
        candidates.sort(key=lambda pr: int(pr[0]["config"]["seed"]))
    return candidates[0]


def _build_ref_logits(ref_mode: str, base_logprob_per_token: torch.Tensor) -> torch.Tensor:
    if ref_mode == "uniform":
        return torch.zeros_like(base_logprob_per_token)
    if ref_mode == "base_lm_lognorm":
        x = base_logprob_per_token
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mu) / sd
    raise ValueError(ref_mode)


def _trunc(s: str, n: int = 320) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _topk(probs: torch.Tensor, k: int) -> List[int]:
    k = min(k, probs.numel())
    return torch.topk(probs, k).indices.tolist()


def _candidate_row(rank: int, cand: Dict[str, Any], prob: float) -> str:
    return (
        f"  {rank}. π={prob:.3f}  r*={cand['true_reward']:.3f}  "
        f"len={cand['length_norm']:.3f}  red={cand['redundancy']:.3f}  "
        f"`{_trunc(cand['text'], 200)}`"
    )


# ---------- main ----------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--results-dir", default="finite_response_sail/results")
    p.add_argument("--cache-dir", default="finite_response_sail/cache")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--featureset", default=None, help="Filter; default: any")
    p.add_argument("--mode", default=None, help="Filter; default: any")
    p.add_argument("--objective", default=None, help="Filter; default: any")
    p.add_argument("--seed", type=int, default=None, help="Filter; default: any")
    p.add_argument("--prefer-rho-mis", type=float, default=None,
                   help="When multiple (nominal, robust) pairs exist, prefer the "
                        "one whose rho_mis is closest to this value.")
    p.add_argument("--n-examples", type=int, default=10)
    p.add_argument("--sample-seed", type=int, default=0,
                   help="RNG seed for picking which prompts to display.")
    p.add_argument("--out", default=None, help="Override output path.")
    args = p.parse_args()

    results_dir = Path(args.results_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()

    runs = _load_runs_metadata(results_dir, args.tag)
    if not runs:
        raise RuntimeError(f"No metrics.json under {results_dir / args.tag}")

    nominal, robust = _pick_nominal_robust_pair(
        runs,
        featureset=args.featureset, mode=args.mode, objective=args.objective,
        seed=args.seed, prefer_rho_mis=args.prefer_rho_mis,
    )
    cfg_n, cfg_r = nominal["config"], robust["config"]
    K = int(cfg_n["K"])
    featureset = cfg_n["featureset"]
    ref_mode = cfg_n["ref"]

    # Load pool and features for the requested split.
    rs_p = response_set_path(cache_dir, K, args.split)
    ps_p = psi_path(cache_dir, K, featureset, args.split)
    if not rs_p.exists() or not ps_p.exists():
        raise FileNotFoundError(
            f"Missing cache: need {rs_p} and {ps_p}. Build via `run.sh cache`."
        )
    rs = torch.load(rs_p, weights_only=False)
    psi_payload = torch.load(ps_p, weights_only=False)
    psi = psi_payload["psi"].float()           # [N, K, d]
    cands = rs["candidates"]                   # list[list[dict]]
    dialogues = rs["dialogues"]
    refs = rs["refs"]
    N = len(cands)

    base_lp = torch.tensor(
        [[c["base_logprob_per_token"] for c in cs] for cs in cands],
        dtype=torch.float32,
    )
    ref_logits = _build_ref_logits(ref_mode, base_lp)

    # Load thetas.
    theta_n_p = nominal["_run_dir"] / "theta.pt"
    theta_r_p = robust["_run_dir"] / "theta.pt"
    if not theta_n_p.exists() or not theta_r_p.exists():
        raise FileNotFoundError(
            f"theta.pt missing under {nominal['_run_dir']} and/or {robust['_run_dir']}. "
            f"Re-run training after the checkpoint-saving patch."
        )
    theta_n = torch.load(theta_n_p, map_location="cpu").float()
    theta_r = torch.load(theta_r_p, map_location="cpu").float()

    with torch.no_grad():
        probs_n = policy_probs(theta_n, psi, ref_logits)   # [N, K]
        probs_r = policy_probs(theta_r, psi, ref_logits)

    rng = random.Random(args.sample_seed)
    idxs = list(range(N))
    rng.shuffle(idxs)
    idxs = idxs[: args.n_examples]

    out_lines: List[str] = []
    out_lines.append(f"# Examples — Nominal vs Robust ({args.tag})\n")
    out_lines.append(
        f"**Nominal run:** `{nominal['run_id']}`  "
        f"(rho_alg={cfg_n['rho_alg']}, rho_mis={cfg_n['rho_mis']})\n"
    )
    out_lines.append(
        f"**Robust run:**  `{robust['run_id']}`  "
        f"(rho_alg={cfg_r['rho_alg']}, rho_mis={cfg_r['rho_mis']})\n"
    )
    out_lines.append(f"**Split:** {args.split}  |  K={K}  |  featureset={featureset}  |  ref={ref_mode}\n")
    out_lines.append("")

    # Aggregate delta summaries across the chosen examples.
    total = {"true_n": 0.0, "true_r": 0.0, "spur_n": 0.0, "spur_r": 0.0}
    for n, i in enumerate(idxs, start=1):
        pn, pr = probs_n[i], probs_r[i]
        cand_i = cands[i]
        top1_n = int(torch.argmax(pn).item())
        top1_r = int(torch.argmax(pr).item())
        exp_true_n = float((pn * torch.tensor([c["true_reward"] for c in cand_i])).sum())
        exp_true_r = float((pr * torch.tensor([c["true_reward"] for c in cand_i])).sum())
        exp_spur_n = float((pn * torch.tensor([c["length_norm"] for c in cand_i])).sum())
        exp_spur_r = float((pr * torch.tensor([c["length_norm"] for c in cand_i])).sum())
        total["true_n"] += exp_true_n
        total["true_r"] += exp_true_r
        total["spur_n"] += exp_spur_n
        total["spur_r"] += exp_spur_r

        out_lines.append(f"## Example {n} (idx={i})\n")
        out_lines.append(f"**Dialogue:** {_trunc(dialogues[i], 600)}\n")
        out_lines.append(f"**Reference summary:** {_trunc(refs[i], 300)}\n")

        out_lines.append(
            f"**Nominal top-1** (π={float(pn[top1_n]):.3f}, "
            f"r*={cand_i[top1_n]['true_reward']:.3f}, "
            f"len={cand_i[top1_n]['length_norm']:.3f}, "
            f"red={cand_i[top1_n]['redundancy']:.3f}):  "
            f"`{_trunc(cand_i[top1_n]['text'], 280)}`\n"
        )
        out_lines.append(
            f"**Robust  top-1** (π={float(pr[top1_r]):.3f}, "
            f"r*={cand_i[top1_r]['true_reward']:.3f}, "
            f"len={cand_i[top1_r]['length_norm']:.3f}, "
            f"red={cand_i[top1_r]['redundancy']:.3f}):  "
            f"`{_trunc(cand_i[top1_r]['text'], 280)}`\n"
        )

        out_lines.append(
            f"**E_π[r*]:** nominal={exp_true_n:.3f}  robust={exp_true_r:.3f}  "
            f"Δ={exp_true_r - exp_true_n:+.3f}"
        )
        out_lines.append(
            f"**E_π[spurious]:** nominal={exp_spur_n:.3f}  robust={exp_spur_r:.3f}  "
            f"Δ={exp_spur_r - exp_spur_n:+.3f}\n"
        )

        out_lines.append("**Nominal top-5 (by π):**")
        for rank, k in enumerate(_topk(pn, 5), start=1):
            out_lines.append(_candidate_row(rank, cand_i[k], float(pn[k])))
        out_lines.append("")
        out_lines.append("**Robust top-5 (by π):**")
        for rank, k in enumerate(_topk(pr, 5), start=1):
            out_lines.append(_candidate_row(rank, cand_i[k], float(pr[k])))
        out_lines.append("\n---\n")

    n_ex = max(1, len(idxs))
    out_lines.append("## Aggregate over shown examples\n")
    out_lines.append(
        f"- Mean E_π[r*]:        nominal={total['true_n']/n_ex:.3f}  "
        f"robust={total['true_r']/n_ex:.3f}  "
        f"Δ={(total['true_r']-total['true_n'])/n_ex:+.3f}"
    )
    out_lines.append(
        f"- Mean E_π[spurious]:  nominal={total['spur_n']/n_ex:.3f}  "
        f"robust={total['spur_r']/n_ex:.3f}  "
        f"Δ={(total['spur_r']-total['spur_n'])/n_ex:+.3f}"
    )

    out_path = (
        Path(args.out) if args.out
        else results_dir / args.tag / "reports" / "examples.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
