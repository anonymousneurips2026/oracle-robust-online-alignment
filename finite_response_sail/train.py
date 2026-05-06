"""Online training loop for the finite-response oracle-robust SAIL policy (v3_pairbias).

Runs entirely over cached features + cached true/bias rewards. No LLM forward
pass happens here. Writes:

    results/<tag>/<run_id>/metrics.json
    results/<tag>/<run_id>/config.json
    results/<tag>/<run_id>/training_curve.jsonl
    results/<tag>/<run_id>/theta.pt
    results/<tag>/<run_id>/oracle_tensors_test.npz

where run_id encodes the config and seed.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .evaluate import evaluate
from .features import psi_path
from .objective import exact_expectation_loss
from .offdiag_objective import offdiag_exact_loss
from .policy import kl_to_ref as _kl_to_ref
from .oracle_misspec import (
    check_margin,
    compute_p_obs,
    compute_p_star,
    compute_pairwise_bias,
    normalize_per_prompt,
)
from .policy import project_l2_
from .response_set import DEFAULT_GEN_MODEL, response_set_path
from .sampled_objective import sampled_pair_loss


def _assert_cache_matches(
    payload_meta: Dict[str, Any],
    *,
    K: int,
    featureset: str,
    split: str,
    expect_model: str,
    source: str,
) -> None:
    """Hard-fail on any mismatch between the cached pool and the current CLI args.

    Catches the silent-reuse hazard where e.g. `grid-k16` picks up a K=12 pool.
    """
    meta_K = payload_meta.get("K")
    if meta_K is None or int(meta_K) != int(K):
        raise RuntimeError(
            f"Cache K mismatch in {source} (split={split}, featureset={featureset}): "
            f"cache has K={meta_K!r}, CLI --K={K}. "
            f"Rebuild the cache for K={K} or pass --K={meta_K}."
        )
    meta_model = payload_meta.get("model")
    if expect_model and meta_model and str(meta_model) != str(expect_model):
        raise RuntimeError(
            f"Cache model mismatch in {source} (split={split}): "
            f"cache built with model={meta_model!r}, --expect-model={expect_model!r}. "
            f"Either rebuild the cache or pass --expect-model={meta_model!r}."
        )


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_ref_logits(
    ref_mode: str, base_logprob_per_token: torch.Tensor,
) -> torch.Tensor:
    """base_logprob_per_token: [N, K]. Returns [N, K]."""
    if ref_mode == "uniform":
        return torch.zeros_like(base_logprob_per_token)
    if ref_mode == "base_lm_lognorm":
        x = base_logprob_per_token
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mu) / sd
    raise ValueError(ref_mode)


def _bias_attr_array(cands, bias_attr: str) -> np.ndarray:
    """Build the per-prompt, per-candidate bias attribute array in [0, 1]."""
    red = np.asarray([[c["redundancy"] for c in cs] for cs in cands], dtype=np.float64)
    ln = np.asarray([[c["length_norm"] for c in cs] for cs in cands], dtype=np.float64)
    if bias_attr == "redundancy":
        raw = red
    elif bias_attr == "length_norm":
        raw = ln
    elif bias_attr == "mixed":
        raw = 0.5 * normalize_per_prompt(red) + 0.5 * normalize_per_prompt(ln)
    else:
        raise ValueError(f"unknown --bias_attr: {bias_attr!r}")
    return normalize_per_prompt(raw)


def _load_split(
    cache_dir: Path, featureset: str, split: str, device: torch.device,
    K: int, expect_model: str, bias_attr: str,
):
    """Returns a dict of tensors already on `device`.

    Fields: psi [N, K, d], true_rewards [N, K], bias_attr [N, K] (normalized to
    [0, 1] per prompt), base_logprob_per_token [N, K]. Oracle tensors are not
    populated here; caller builds them per (tau, rho_mis).

    Raises if the cached pool's metadata does not match the current run config.
    """
    rs_p = response_set_path(cache_dir, K, split)
    psi_p = psi_path(cache_dir, K, featureset, split)
    if not rs_p.exists():
        raise FileNotFoundError(
            f"response-set cache missing: {rs_p}. "
            f"Run `run.sh cache` (or `response_set.py --K {K}`) first."
        )
    if not psi_p.exists():
        raise FileNotFoundError(
            f"psi cache missing: {psi_p}. "
            f"Run `features.py --featureset {featureset} --K {K}` first."
        )
    rs = torch.load(rs_p, weights_only=False)
    psi_payload = torch.load(psi_p, weights_only=False)

    _assert_cache_matches(
        rs["meta"], K=K, featureset=featureset, split=split,
        expect_model=expect_model, source=str(rs_p),
    )
    _assert_cache_matches(
        psi_payload["meta"], K=K, featureset=featureset, split=split,
        expect_model=expect_model, source=str(psi_p),
    )

    psi = psi_payload["psi"].float().to(device)                    # [N, K, d]
    cands = rs["candidates"]
    N, K_cache = len(cands), len(cands[0])
    if K_cache != K:
        raise RuntimeError(
            f"Candidate-count mismatch in {rs_p}: payload has K={K_cache}, CLI --K={K}."
        )
    if psi.shape[1] != K:
        raise RuntimeError(
            f"psi shape mismatch in {psi_p}: psi.shape[1]={psi.shape[1]}, CLI --K={K}."
        )
    tr = np.asarray([[c["true_reward"] for c in cs] for cs in cands], dtype=np.float64)
    ba = _bias_attr_array(cands, bias_attr)
    bl = np.asarray([[c["base_logprob_per_token"] for c in cs] for cs in cands], dtype=np.float64)

    true_r = torch.from_numpy(tr).float().to(device)
    bias_t = torch.from_numpy(ba).float().to(device)
    base_logp = torch.from_numpy(bl).float().to(device)

    return {
        "psi": psi,
        "true_rewards": true_r,
        "bias_attr": bias_t,
        "base_logprob_per_token": base_logp,
        "N": N,
        "K": K,
        "true_r_np": tr,
        "bias_attr_np": ba,
    }


def _prepare_probabilities(
    data: Dict[str, Any], tau: float, rho_mis: float, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    p_star_np = compute_p_star(data["true_r_np"], tau=tau)
    check_margin(p_star_np, rho_mis)
    b_np = compute_pairwise_bias(data["true_r_np"], data["bias_attr_np"])
    p_obs_np = compute_p_obs(p_star_np, b_np, rho_mis)

    def _t(a):
        return torch.from_numpy(a).float().to(device)

    return _t(p_star_np), _t(b_np), _t(p_obs_np), p_star_np, b_np, p_obs_np


def _run_id(args: argparse.Namespace) -> str:
    base = (
        f"mode-{args.mode}_obj-{args.objective}_fs-{args.featureset}_ref-{args.ref}"
        f"_bias-{args.bias_attr}"
        f"_rhoMis-{args.rho_mis:g}_rhoAlg-{args.rho_alg:g}"
        f"_beta-{args.beta:g}_tau-{args.tau:g}_R-{args.R:g}_K-{args.K}_seed-{args.seed}"
    )
    # Only append when nonzero so all pre-existing run-dir names remain valid.
    eta_kl = float(getattr(args, "eta_kl", 0.0) or 0.0)
    if eta_kl > 0.0:
        base = base + f"_etaKL-{eta_kl:g}"
    return base


def _evaluate_split(
    theta: torch.Tensor, data: Dict[str, Any], ref_logits: torch.Tensor,
    p_star: torch.Tensor, b: torch.Tensor, p_obs: torch.Tensor,
    tau: float, rho_mis: float, beta: float, rho_alg: float,
):
    metrics = evaluate(
        theta=theta,
        psi=data["psi"],
        ref_logits=ref_logits,
        true_rewards=data["true_rewards"],
        bias_attr=data["bias_attr"],
        base_logprob_per_token=data["base_logprob_per_token"],
        p_star=p_star,
        b=b,
        p_obs=p_obs,
        tau=tau, rho_mis=rho_mis, beta=beta, rho_alg=rho_alg,
    )
    return asdict(metrics)


def _resolve_rho_alg(args: argparse.Namespace) -> Tuple[float, float]:
    """Returns (rho_alg, rho_alg_ratio). Exactly one of --rho_alg / --rho_alg_ratio
    can be passed; both default to None and we treat either being None as "unset"."""
    alg_set = args.rho_alg is not None
    ratio_set = args.rho_alg_ratio is not None
    if alg_set and ratio_set:
        raise SystemExit(
            "Pass exactly one of --rho_alg / --rho_alg_ratio, not both."
        )
    if ratio_set:
        rho_alg = float(args.rho_alg_ratio) * float(args.rho_mis)
        rho_alg_ratio = float(args.rho_alg_ratio)
    elif alg_set:
        rho_alg = float(args.rho_alg)
        rho_alg_ratio = (
            float(rho_alg / args.rho_mis) if args.rho_mis > 0 else 0.0
        )
    else:
        rho_alg = 0.0
        rho_alg_ratio = 0.0
    return rho_alg, rho_alg_ratio


def train(args: argparse.Namespace):
    device = _choose_device()
    _set_seed(args.seed)

    # Resolve rho_alg / ratio and freeze them onto args so config.json records both.
    rho_alg, rho_alg_ratio = _resolve_rho_alg(args)
    args.rho_alg = rho_alg
    args.rho_alg_ratio = rho_alg_ratio

    # Handle deprecated --objective offdiag alias.
    if args.objective == "offdiag":
        print("[warn] --objective offdiag is deprecated; using offdiag_exact.", file=sys.stderr)
        args.objective = "offdiag_exact"

    cache_dir = Path(args.cache_dir).resolve()
    train_data = _load_split(cache_dir, args.featureset, "train", device, args.K, args.expect_model, args.bias_attr)
    val_data = _load_split(cache_dir, args.featureset, "val", device, args.K, args.expect_model, args.bias_attr)
    test_data = _load_split(cache_dir, args.featureset, "test", device, args.K, args.expect_model, args.bias_attr)

    ref_train = _build_ref_logits(args.ref, train_data["base_logprob_per_token"])
    ref_val = _build_ref_logits(args.ref, val_data["base_logprob_per_token"])
    ref_test = _build_ref_logits(args.ref, test_data["base_logprob_per_token"])

    pst_train, b_train, pob_train, _, _, _ = _prepare_probabilities(train_data, args.tau, args.rho_mis, device)
    pst_val, b_val, pob_val, _, _, _ = _prepare_probabilities(val_data, args.tau, args.rho_mis, device)
    pst_test, b_test, pob_test, p_star_test_np, b_test_np, p_obs_test_np = _prepare_probabilities(
        test_data, args.tau, args.rho_mis, device,
    )

    d = train_data["psi"].shape[-1]
    theta = torch.zeros(d, device=device, requires_grad=True)
    optim = torch.optim.AdamW([theta], lr=args.lr, weight_decay=0.0)

    N = train_data["N"]
    batch_size = min(args.batch_size, N)
    steps_per_round = args.steps_per_round
    rounds = args.rounds
    if args.smoke:
        rounds = 1
        steps_per_round = min(steps_per_round, 20)

    run_id = _run_id(args)
    out_dir = Path(args.results_dir).resolve() / args.tag / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / "training_curve.jsonl"
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    # Pick center probabilities for training loss.
    if args.mode == "theory_centered":
        p_center_train = pst_train
    elif args.mode == "biased_feedback":
        p_center_train = pob_train
    else:
        raise ValueError(args.mode)

    sampled_baseline = torch.zeros((), device=device)
    baseline_ema = 0.9

    start = time.time()
    step_counter = 0

    curve_file = open(curve_path, "w")

    def _log_curve(record: Dict[str, Any]):
        curve_file.write(json.dumps(record) + "\n")
        curve_file.flush()

    for rd in range(rounds):
        perm = torch.randperm(N, device=device)
        cursor = 0
        for st in range(steps_per_round):
            if cursor + batch_size > N:
                perm = torch.randperm(N, device=device)
                cursor = 0
            idx = perm[cursor:cursor + batch_size]
            cursor += batch_size

            psi_b = train_data["psi"][idx]
            ref_b = ref_train[idx]
            pc_b = p_center_train[idx]

            optim.zero_grad(set_to_none=True)
            if args.objective == "exact":
                loss = exact_expectation_loss(theta, psi_b, ref_b, pc_b, args.rho_alg, args.beta)
            elif args.objective == "offdiag_exact":
                loss = offdiag_exact_loss(theta, psi_b, ref_b, pc_b, args.rho_alg, args.beta)
            elif args.objective == "sampled":
                loss, sl = sampled_pair_loss(
                    theta, psi_b, ref_b, pc_b, args.rho_alg, args.beta,
                    baseline=sampled_baseline,
                )
                sampled_baseline = (baseline_ema * sampled_baseline
                                    + (1 - baseline_ema) * sl.detach())
            else:
                raise ValueError(args.objective)

            # Additive KL-to-reference smoothing (orthogonal to objective family).
            # Computed from the same minibatch psi/ref so no extra RNG draws.
            if args.eta_kl and args.eta_kl > 0.0:
                kl_term = _kl_to_ref(theta, psi_b, ref_b).mean()
                loss = loss + args.eta_kl * kl_term

            loss.backward()
            optim.step()
            project_l2_(theta, args.R)
            step_counter += 1

            if step_counter % max(1, args.log_every) == 0:
                _log_curve({
                    "round": rd, "step": st, "global_step": step_counter,
                    "loss": float(loss.item()),
                    "theta_norm": float(theta.norm(p=2).item()),
                })

        val_metrics = _evaluate_split(
            theta.detach(), val_data, ref_val, pst_val, b_val, pob_val,
            args.tau, args.rho_mis, args.beta, args.rho_alg,
        )
        _log_curve({
            "round": rd, "step": steps_per_round - 1,
            "global_step": step_counter,
            "eval": "val", "metrics": val_metrics,
        })
        elapsed = time.time() - start
        print(
            f"[train] rd={rd+1}/{rounds}  step={step_counter}  "
            f"val_true={val_metrics['expected_true_utility']:.4f}  "
            f"val_bias={val_metrics['expected_bias_attr']:.4f}  "
            f"wcl={val_metrics['worst_case_loss']:.4f}  "
            f"||theta||={val_metrics['theta_norm']:.3f}  "
            f"t={elapsed:.1f}s"
        )

    final_val = _evaluate_split(
        theta.detach(), val_data, ref_val, pst_val, b_val, pob_val,
        args.tau, args.rho_mis, args.beta, args.rho_alg,
    )
    final_test = _evaluate_split(
        theta.detach(), test_data, ref_test, pst_test, b_test, pob_test,
        args.tau, args.rho_mis, args.beta, args.rho_alg,
    )
    final_train = _evaluate_split(
        theta.detach(), train_data, ref_train, pst_train, b_train, pob_train,
        args.tau, args.rho_mis, args.beta, args.rho_alg,
    )
    curve_file.close()

    out = {
        "run_id": run_id,
        "tag": args.tag,
        "config": vars(args),
        "final": {
            "train": final_train,
            "val": final_val,
            "test": final_test,
        },
        "wall_clock_seconds": time.time() - start,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    torch.save(theta.detach().cpu(), out_dir / "theta.pt")

    # Persist the test-split oracle tensors so aggregate.py can render
    # bias_diagnostics.png without recomputing them.
    np.savez_compressed(
        out_dir / "oracle_tensors_test.npz",
        p_star=p_star_test_np.astype(np.float32),
        p_obs=p_obs_test_np.astype(np.float32),
        b=b_test_np.astype(np.float32),
        true_reward=test_data["true_r_np"].astype(np.float32),
        bias_attr=test_data["bias_attr_np"].astype(np.float32),
    )

    print(f"[write] {out_dir / 'metrics.json'}")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["theory_centered", "biased_feedback"], required=True)
    p.add_argument(
        "--objective",
        choices=["exact", "offdiag_exact", "offdiag", "sampled"],
        default="exact",
        help="'offdiag' is a deprecated alias for 'offdiag_exact'.",
    )
    p.add_argument("--featureset", choices=["controlled", "no_oracle"], required=True)
    p.add_argument("--ref", choices=["uniform", "base_lm_lognorm"], default="uniform")
    p.add_argument(
        "--bias_attr", choices=["redundancy", "length_norm", "mixed"],
        default="redundancy",
    )
    p.add_argument("--rho_alg", type=float, default=None,
                   help="Absolute rho_alg. Mutually exclusive with --rho_alg_ratio.")
    p.add_argument("--rho_alg_ratio", type=float, default=None,
                   help="rho_alg = rho_alg_ratio * rho_mis. Mutually exclusive with --rho_alg.")
    p.add_argument("--eta_kl", type=float, default=0.0,
                   help="Additive KL-to-reference smoothing weight. When > 0, the "
                        "training loss has +eta_kl * E_x KL(pi_theta || pi_ref) added "
                        "in. Default 0.0 (no smoothing). Used by the KL-smoothing "
                        "baseline experiment.")
    p.add_argument("--rho_mis", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--R", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--steps_per_round", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--K", type=int, default=12)
    p.add_argument("--expect-model", default=DEFAULT_GEN_MODEL,
                   help="assert the cached response-set was built with this model; "
                        "pass '' to skip the check")
    p.add_argument("--tag", type=str, default="dev")
    p.add_argument("--cache-dir", default="finite_response_sail/cache")
    p.add_argument("--results_dir", default="finite_response_sail/results")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--smoke", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
