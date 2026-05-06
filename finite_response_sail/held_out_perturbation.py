"""Held-out perturbation robustness evaluation (no retraining).

Evaluates the four already-trained policies (Nominal, Robust-0.025rho,
KL-matchR, KL-matchEntropy) plus the full KL eta-grid against a fresh family
of bounded oracle perturbations (M random unit-vector directions in feature
space per training seed). The activation rule for b' matches training exactly
(see oracle_misspec.compute_pairwise_bias); only the bias direction is
re-drawn.

This is a pure evaluation experiment. No model is (re)trained.

Run:
    python -m finite_response_sail.held_out_perturbation \
        --rho-mis 0.075 --num-directions 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --include-axis-aligned --include-kl-frontier \
        --out-dir reports/held_out_perturbation
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

# ---- existing modules (read-only reuse) -----------------------------------
from .train import _load_split, _build_ref_logits
from .oracle_misspec import (
    compute_p_star,
    compute_pairwise_bias,
    compute_p_obs,
    check_margin,
    oracle_diagnostics,
)
from .objective import (
    exact_expectation_loss,
    sensitivity_R,
    expected_true_utility,
)
from .offdiag_objective import offdiag_exact_loss
from .policy import policy_probs, kl_to_ref


# ---- constants -------------------------------------------------------------
DEFAULT_RHO_MIS = 0.075
DEFAULT_BETA = 0.1
DEFAULT_TAU = 2.0
DEFAULT_K = 12
DEFAULT_FEATURESET = "controlled"
DEFAULT_BIAS_ATTR = "redundancy"
DEFAULT_REF = "base_lm_lognorm"
DEFAULT_NUM_DIRECTIONS = 16
DEFAULT_NUM_SEEDS = 10
RNG_NAMESPACE = "held_out_v1"
RNG_RETRY_NAMESPACE_PREFIX = "held_out_v1_retry_"
RETRY_MAX = 5
BIAS_ACTIVE_RATE_FLOOR = 0.05

KL_ETA_GRID = [0.0001, 0.0003, 0.0004, 0.0005, 0.0007,
               0.001, 0.003, 0.01, 0.03, 0.1]
KL_MATCH_ETA = 0.0004  # both KL-matchR and KL-matchEntropy at rho_mis=0.075

ROBUST_RHO_ALG_RATIO = 0.025

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "held_out_perturbation"
DEFAULT_CACHE_DIR = REPO_ROOT / "finite_response_sail" / "cache"
DEFAULT_CONFIRM_DIR = REPO_ROOT / "finite_response_sail" / "results" / "v3_pairbias_confirm"
DEFAULT_KL_DIR = REPO_ROOT / "finite_response_sail" / "results" / "v3_pairbias_kl_smoothing"


# ---- hashing & RNG ---------------------------------------------------------
def held_out_seed(namespace: str, s: int, m: int) -> int:
    """Deterministic 64-bit seed from (namespace, training_seed, direction_idx).

    Uses blake2b (deterministic across processes) rather than Python's salted
    builtin ``hash()`` so reproducibility holds across runs.
    """
    payload = f"{namespace}|{s}|{m}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def generate_unit_direction(s: int, m: int, d: int,
                            namespace: str = RNG_NAMESPACE) -> np.ndarray:
    """Sample u in R^d uniformly on the unit sphere with reproducible seeding."""
    rng = np.random.default_rng(held_out_seed(namespace, s, m))
    u = rng.standard_normal(d).astype(np.float64)
    n = float(np.linalg.norm(u))
    if n < 1e-12:
        raise RuntimeError(
            f"degenerate direction (||u||={n:.3e}) at namespace={namespace}, "
            f"s={s}, m={m}"
        )
    return u / n


# ---- perturbation construction --------------------------------------------
@dataclass
class Perturbation:
    seed: int
    m: int
    direction_kind: str  # "random" | "axis"
    u: np.ndarray
    b: np.ndarray
    p_obs: np.ndarray
    bias_active_rate: float
    max_abs_b: float
    namespace_used: str  # which namespace ended up producing the kept direction


def _build_one_perturbation(
    psi_np: np.ndarray,         # [N, K, d]
    true_reward_np: np.ndarray, # [N, K]
    p_star_np: np.ndarray,      # [N, K, K]
    rho_mis: float,
    u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    s_held = np.einsum("nkd,d->nk", psi_np, u)        # [N, K]
    b = compute_pairwise_bias(true_reward_np, s_held)  # [N, K, K]
    p_obs = compute_p_obs(p_star_np, b, rho_mis)
    diag = oracle_diagnostics(
        p_star=p_star_np,
        p_obs=p_obs,
        b=b,
        bias_attr=s_held,
        true_reward=true_reward_np,
    )
    return b, p_obs, diag


def build_held_out_perturbation(
    psi_np: np.ndarray,
    true_reward_np: np.ndarray,
    p_star_np: np.ndarray,
    rho_mis: float,
    s: int, m: int, d: int,
) -> Tuple[Perturbation, List[dict]]:
    """Generate one held-out perturbation with retry-on-low-activation logic."""
    retry_log: List[dict] = []
    last_diag: Optional[Dict[str, float]] = None
    for attempt in range(RETRY_MAX + 1):
        ns = (RNG_NAMESPACE if attempt == 0
              else f"{RNG_RETRY_NAMESPACE_PREFIX}{attempt}")
        u = generate_unit_direction(s, m, d, namespace=ns)
        b, p_obs, diag = _build_one_perturbation(
            psi_np, true_reward_np, p_star_np, rho_mis, u
        )
        last_diag = diag
        ok = diag["bias_active_rate"] >= BIAS_ACTIVE_RATE_FLOOR
        retry_log.append({
            "seed": s, "m": m, "attempt": attempt, "namespace": ns,
            "bias_active_rate": diag["bias_active_rate"],
            "max_abs_b": diag["max_abs_b"],
            "ok": ok,
        })
        # |b| <= 1 sanity (per spec validity check 1).
        max_abs = float(np.abs(b).max())
        if max_abs > 1.0 + 1e-10:
            raise RuntimeError(
                f"validity check 1 failed: |b|.max()={max_abs} > 1 at "
                f"(s={s}, m={m}, attempt={attempt})"
            )
        if ok:
            pert = Perturbation(
                seed=s, m=m, direction_kind="random",
                u=u, b=b, p_obs=p_obs,
                bias_active_rate=diag["bias_active_rate"],
                max_abs_b=diag["max_abs_b"],
                namespace_used=ns,
            )
            return pert, retry_log
    raise RuntimeError(
        f"held-out direction (s={s}, m={m}) failed bias_active_rate floor "
        f"({BIAS_ACTIVE_RATE_FLOOR}) after {RETRY_MAX} retries; "
        f"last_active_rate={last_diag['bias_active_rate']:.4f}"
    )


def build_axis_aligned_perturbations(
    psi_np: np.ndarray,
    true_reward_np: np.ndarray,
    p_star_np: np.ndarray,
    rho_mis: float,
    s: int,
    d: int,
) -> List[Perturbation]:
    """Build d deterministic e_j perturbations (m = -1..-d). No retry needed."""
    out: List[Perturbation] = []
    for j in range(d):
        u = np.zeros(d, dtype=np.float64)
        u[j] = 1.0
        b, p_obs, diag = _build_one_perturbation(
            psi_np, true_reward_np, p_star_np, rho_mis, u
        )
        out.append(Perturbation(
            seed=s, m=-(j + 1), direction_kind="axis",
            u=u, b=b, p_obs=p_obs,
            bias_active_rate=diag["bias_active_rate"],
            max_abs_b=diag["max_abs_b"],
            namespace_used="axis_aligned",
        ))
    return out


# ---- run resolution --------------------------------------------------------
@dataclass
class RunSpec:
    method: str  # "Nominal" | "Robust-0.025rho" | "KL-matchR" | "KL-matchEntropy"
    seed: int
    rho_alg: float
    eta_kl: float  # 0.0 means no KL smoothing
    run_dir: Path

    def label(self) -> str:
        return f"{self.method}@seed{self.seed}"


def _run_id(
    rho_mis: float, rho_alg: float, eta_kl: float, seed: int,
    beta: float = DEFAULT_BETA, tau: float = DEFAULT_TAU,
    R: float = 10.0, K: int = DEFAULT_K,
    featureset: str = DEFAULT_FEATURESET,
    bias_attr: str = DEFAULT_BIAS_ATTR,
    ref: str = DEFAULT_REF,
) -> str:
    base = (
        f"mode-biased_feedback_obj-offdiag_exact_fs-{featureset}"
        f"_ref-{ref}_bias-{bias_attr}"
        f"_rhoMis-{rho_mis:g}_rhoAlg-{rho_alg:g}"
        f"_beta-{beta:g}_tau-{tau:g}_R-{R:g}_K-{K}_seed-{seed}"
    )
    if eta_kl > 0:
        base += f"_etaKL-{eta_kl:g}"
    return base


def resolve_runs(
    confirm_dir: Path, kl_dir: Path,
    rho_mis: float, seeds: Iterable[int],
) -> List[RunSpec]:
    """Build (method, seed) -> run_dir for the four canonical methods.

    KL-matchR and KL-matchEntropy point to the same physical run dir at
    eta_kl=KL_MATCH_ETA (per existing kl_selected_comparison.csv).
    """
    rho_alg_robust = rho_mis * ROBUST_RHO_ALG_RATIO
    specs: List[RunSpec] = []
    for s in seeds:
        # Nominal
        rid = _run_id(rho_mis=rho_mis, rho_alg=0.0, eta_kl=0.0, seed=s)
        specs.append(RunSpec(
            method="Nominal", seed=s, rho_alg=0.0, eta_kl=0.0,
            run_dir=confirm_dir / rid,
        ))
        # Robust-0.025rho
        rid = _run_id(rho_mis=rho_mis, rho_alg=rho_alg_robust, eta_kl=0.0, seed=s)
        specs.append(RunSpec(
            method="Robust-0.025rho", seed=s, rho_alg=rho_alg_robust, eta_kl=0.0,
            run_dir=confirm_dir / rid,
        ))
        # KL-matchR + KL-matchEntropy (same dir)
        rid = _run_id(rho_mis=rho_mis, rho_alg=0.0, eta_kl=KL_MATCH_ETA, seed=s)
        kl_dir_full = kl_dir / rid
        specs.append(RunSpec(
            method="KL-matchR", seed=s, rho_alg=0.0, eta_kl=KL_MATCH_ETA,
            run_dir=kl_dir_full,
        ))
        specs.append(RunSpec(
            method="KL-matchEntropy", seed=s, rho_alg=0.0, eta_kl=KL_MATCH_ETA,
            run_dir=kl_dir_full,
        ))
    return specs


def resolve_kl_grid_runs(
    kl_dir: Path, rho_mis: float, seeds: Iterable[int],
    eta_grid: Iterable[float] = KL_ETA_GRID,
) -> List[RunSpec]:
    out: List[RunSpec] = []
    for s in seeds:
        for eta in eta_grid:
            rid = _run_id(rho_mis=rho_mis, rho_alg=0.0, eta_kl=eta, seed=s)
            out.append(RunSpec(
                method=f"KL-grid-eta{eta:g}", seed=s, rho_alg=0.0, eta_kl=eta,
                run_dir=kl_dir / rid,
            ))
    return out


def validate_runs(specs: Iterable[RunSpec], rho_mis: float) -> None:
    """Assert each run_dir has theta.pt + config.json, and the config matches."""
    seen_dirs: set = set()
    for spec in specs:
        if spec.run_dir in seen_dirs:
            continue  # KL-matchR / KL-matchEntropy share a path
        seen_dirs.add(spec.run_dir)
        if not (spec.run_dir / "theta.pt").exists():
            raise FileNotFoundError(f"theta.pt missing in {spec.run_dir}")
        if not (spec.run_dir / "config.json").exists():
            raise FileNotFoundError(f"config.json missing in {spec.run_dir}")
        with open(spec.run_dir / "config.json") as f:
            cfg = json.load(f)
        # consistency checks
        for key, expected in [
            ("rho_mis", rho_mis), ("seed", spec.seed),
            ("featureset", DEFAULT_FEATURESET),
            ("bias_attr", DEFAULT_BIAS_ATTR), ("ref", DEFAULT_REF),
            ("K", DEFAULT_K), ("tau", DEFAULT_TAU), ("beta", DEFAULT_BETA),
        ]:
            actual = cfg.get(key)
            if isinstance(expected, float):
                if not math.isclose(float(actual), expected, abs_tol=1e-9):
                    raise ValueError(
                        f"config mismatch in {spec.run_dir}: {key}={actual} "
                        f"!= expected {expected}"
                    )
            elif actual != expected:
                raise ValueError(
                    f"config mismatch in {spec.run_dir}: {key}={actual!r} "
                    f"!= expected {expected!r}"
                )
        # rho_alg / eta_kl
        if not math.isclose(float(cfg.get("rho_alg", 0.0)),
                            spec.rho_alg, abs_tol=1e-12):
            raise ValueError(
                f"rho_alg mismatch in {spec.run_dir}: "
                f"{cfg.get('rho_alg')} != {spec.rho_alg}"
            )
        if spec.eta_kl > 0:
            if not math.isclose(float(cfg.get("eta_kl", 0.0)),
                                spec.eta_kl, abs_tol=1e-12):
                raise ValueError(
                    f"eta_kl mismatch in {spec.run_dir}: "
                    f"{cfg.get('eta_kl')} != {spec.eta_kl}"
                )


# ---- cache loading ---------------------------------------------------------
def load_eval_cache(
    cache_dir: Path,
    device: torch.device,
    featureset: str = DEFAULT_FEATURESET,
    K: int = DEFAULT_K,
    bias_attr: str = DEFAULT_BIAS_ATTR,
    ref: str = DEFAULT_REF,
    tau: float = DEFAULT_TAU,
    rho_mis: float = DEFAULT_RHO_MIS,
) -> Dict[str, object]:
    data = _load_split(
        cache_dir=Path(cache_dir),
        featureset=featureset, split="test", device=device,
        K=K, expect_model="", bias_attr=bias_attr,
    )
    ref_logits = _build_ref_logits(ref, data["base_logprob_per_token"])
    p_star_np = compute_p_star(data["true_r_np"], tau=tau)  # [N, K, K]
    check_margin(p_star_np, rho_mis)  # must hold for all held-out cells
    psi_np = data["psi"].cpu().numpy().astype(np.float64)
    d = int(psi_np.shape[-1])
    return {
        "data": data,
        "ref_logits": ref_logits,
        "p_star_np": p_star_np,
        "psi_np": psi_np,
        "true_reward_np": data["true_r_np"],
        "d": d,
        "device": device,
        "N": int(data["N"]),
        "K": int(data["K"]),
    }


# ---- per-cell evaluation ---------------------------------------------------
@torch.no_grad()
def evaluate_cell(
    theta: torch.Tensor,
    cache: Dict[str, object],
    p_obs_np: Optional[np.ndarray],
    beta: float,
) -> Dict[str, float]:
    """Compute scalar metrics for a single (theta, perturbation) pair.

    If p_obs_np is None, evaluates at the true oracle p_star.
    """
    psi: torch.Tensor = cache["data"]["psi"]
    ref_logits: torch.Tensor = cache["ref_logits"]
    device: torch.device = cache["device"]
    theta_t = theta.to(device).float()

    p_center_np = cache["p_star_np"] if p_obs_np is None else p_obs_np
    p_center = torch.from_numpy(p_center_np).float().to(device)

    L_offdiag = float(offdiag_exact_loss(theta_t, psi, ref_logits, p_center,
                                         0.0, beta).item())
    L_exact = float(exact_expectation_loss(theta_t, psi, ref_logits, p_center,
                                           0.0, beta).item())
    R = float(sensitivity_R(theta_t, psi, ref_logits))
    E_rstar = float(expected_true_utility(theta_t, psi, ref_logits,
                                          cache["data"]["true_rewards"]))
    KL = float(kl_to_ref(theta_t, psi, ref_logits).mean().item())
    return {
        "L_offdiag": L_offdiag, "L_exact": L_exact,
        "sensitivity_R": R, "E_rstar": E_rstar, "KL_to_ref": KL,
    }


# ---- main runner -----------------------------------------------------------
def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


def _save_perturbations_npz(
    out_path: Path,
    perturbations: Dict[Tuple[int, int], Perturbation],
    seeds: List[int], num_directions: int, d: int, include_axis: bool,
    N: int, K: int,
) -> None:
    """Save (u, b) arrays for the random and axis-aligned directions.

    Random directions: u_random [n_seeds, M, d], b_random [n_seeds, M, N, K, K] (float32).
    Axis directions: u_axis [n_seeds, d, d], b_axis [n_seeds, d, N, K, K] (float32).
    """
    n_seeds = len(seeds)
    u_random = np.zeros((n_seeds, num_directions, d), dtype=np.float64)
    b_random = np.zeros((n_seeds, num_directions, N, K, K), dtype=np.float32)
    bias_active_rates_random = np.zeros((n_seeds, num_directions), dtype=np.float64)
    namespaces_random = np.empty((n_seeds, num_directions), dtype=object)
    for si, s in enumerate(seeds):
        for m in range(1, num_directions + 1):
            p = perturbations[(s, m)]
            u_random[si, m - 1] = p.u
            b_random[si, m - 1] = p.b.astype(np.float32)
            bias_active_rates_random[si, m - 1] = p.bias_active_rate
            namespaces_random[si, m - 1] = p.namespace_used

    arrays = {
        "seeds": np.asarray(seeds, dtype=np.int64),
        "m_random": np.arange(1, num_directions + 1, dtype=np.int64),
        "u_random": u_random,
        "b_random": b_random,
        "bias_active_rate_random": bias_active_rates_random,
        "namespace_random": namespaces_random.astype(str),
    }

    if include_axis:
        u_axis = np.zeros((n_seeds, d, d), dtype=np.float64)
        b_axis = np.zeros((n_seeds, d, N, K, K), dtype=np.float32)
        bias_active_rates_axis = np.zeros((n_seeds, d), dtype=np.float64)
        for si, s in enumerate(seeds):
            for j in range(d):
                p = perturbations[(s, -(j + 1))]
                u_axis[si, j] = p.u
                b_axis[si, j] = p.b.astype(np.float32)
                bias_active_rates_axis[si, j] = p.bias_active_rate
        arrays["m_axis"] = -np.arange(1, d + 1, dtype=np.int64)
        arrays["u_axis"] = u_axis
        arrays["b_axis"] = b_axis
        arrays["bias_active_rate_axis"] = bias_active_rates_axis

    np.savez_compressed(out_path, **arrays)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.seeds = [0, 1]
        args.num_directions = 4
        args.include_kl_frontier = False

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}, out_dir={out_dir}", flush=True)

    cache = load_eval_cache(
        cache_dir=Path(args.cache_dir),
        device=device,
        featureset=args.featureset, K=args.K, bias_attr=args.bias_attr,
        ref=args.ref, tau=args.tau, rho_mis=args.rho_mis,
    )
    print(f"[cache] N={cache['N']}, K={cache['K']}, d={cache['d']}", flush=True)

    # Resolve and validate run dirs
    main_specs = resolve_runs(
        confirm_dir=Path(args.confirm_dir),
        kl_dir=Path(args.kl_dir),
        rho_mis=args.rho_mis, seeds=args.seeds,
    )
    validate_runs(main_specs, rho_mis=args.rho_mis)
    print(f"[runs] resolved {len(main_specs)} main specs", flush=True)

    kl_grid_specs: List[RunSpec] = []
    if args.include_kl_frontier:
        kl_grid_specs = resolve_kl_grid_runs(
            kl_dir=Path(args.kl_dir),
            rho_mis=args.rho_mis, seeds=args.seeds,
        )
        validate_runs(kl_grid_specs, rho_mis=args.rho_mis)
        print(f"[runs] resolved {len(kl_grid_specs)} KL-grid specs", flush=True)

    # ---------------- Generate perturbations ----------------
    perturbations: Dict[Tuple[int, int], Perturbation] = {}
    retry_rows: List[dict] = []
    for s in args.seeds:
        for m in range(1, args.num_directions + 1):
            pert, log = build_held_out_perturbation(
                cache["psi_np"], cache["true_reward_np"], cache["p_star_np"],
                args.rho_mis, s, m, cache["d"],
            )
            perturbations[(s, m)] = pert
            retry_rows.extend(log)
        if args.include_axis_aligned:
            axes = build_axis_aligned_perturbations(
                cache["psi_np"], cache["true_reward_np"], cache["p_star_np"],
                args.rho_mis, s, cache["d"],
            )
            for p in axes:
                perturbations[(p.seed, p.m)] = p
        print(f"[perturb] built {args.num_directions} random + "
              f"{cache['d'] if args.include_axis_aligned else 0} axis "
              f"directions for seed {s}", flush=True)

    # ---------------- Compute nominal cells (m=0) ----------------
    rows: List[dict] = []
    nominal_metrics: Dict[Tuple[str, int], Dict[str, float]] = {}

    def _theta_for(spec: RunSpec) -> torch.Tensor:
        t = torch.load(spec.run_dir / "theta.pt", map_location=device,
                       weights_only=False)
        return t.float()

    def _record_nominal(spec: RunSpec, theta: torch.Tensor) -> Dict[str, float]:
        m_at_pstar = evaluate_cell(theta, cache, p_obs_np=None, beta=args.beta)
        theta_norm = float(torch.norm(theta).item())
        rows.append({
            "seed": spec.seed,
            "method": spec.method,
            "eta_kl": spec.eta_kl,
            "rho_alg": spec.rho_alg,
            "rho_mis": args.rho_mis,
            "m": 0,
            "direction_kind": "nominal",
            "alpha": 0.0,
            "shift_loss": m_at_pstar["L_offdiag"],
            "L_offdiag": m_at_pstar["L_offdiag"],
            "L_exact": m_at_pstar["L_exact"],
            "L_SAIL_true": m_at_pstar["L_offdiag"],
            "delta_shift": 0.0,
            "sensitivity_R": m_at_pstar["sensitivity_R"],
            "R_eval": m_at_pstar["sensitivity_R"],
            "E_rstar": m_at_pstar["E_rstar"],
            "KL_to_ref": m_at_pstar["KL_to_ref"],
            "bias_active_rate": float("nan"),
            "max_abs_b": float("nan"),
            "theta_norm": theta_norm,
        })
        return m_at_pstar

    # main 4 methods
    print("[eval] main methods (nominal + 16 random + axis dirs)...", flush=True)
    for spec in main_specs:
        # KL-matchR and KL-matchEntropy share theta; load only once via caching
        nm_key = (spec.method, spec.seed)
        if nm_key in nominal_metrics:
            continue
        theta = _theta_for(spec)
        nominal_metrics[nm_key] = _record_nominal(spec, theta)

        L_true = nominal_metrics[nm_key]["L_offdiag"]
        # shift cells
        for (s_p, m), pert in perturbations.items():
            if s_p != spec.seed:
                continue
            cell = evaluate_cell(theta, cache, p_obs_np=pert.p_obs, beta=args.beta)
            theta_norm = float(torch.norm(theta).item())
            rows.append({
                "seed": spec.seed,
                "method": spec.method,
                "eta_kl": spec.eta_kl,
                "rho_alg": spec.rho_alg,
                "rho_mis": args.rho_mis,
                "m": m,
                "direction_kind": pert.direction_kind,
                "alpha": 1.0,
                "shift_loss": cell["L_offdiag"],
                "L_offdiag": cell["L_offdiag"],
                "L_exact": cell["L_exact"],
                "L_SAIL_true": L_true,
                "delta_shift": cell["L_offdiag"] - L_true,
                "sensitivity_R": cell["sensitivity_R"],
                "R_eval": nominal_metrics[nm_key]["sensitivity_R"],
                "E_rstar": cell["E_rstar"],
                "KL_to_ref": cell["KL_to_ref"],
                "bias_active_rate": pert.bias_active_rate,
                "max_abs_b": pert.max_abs_b,
                "theta_norm": theta_norm,
            })

    # KL-matchR and KL-matchEntropy point to the same checkpoint (both select
    # eta_kl=0.0004 at rho_mis=0.075 per kl_selected_comparison.csv), so each
    # is enumerated as its own RunSpec and produces an independent (but
    # numerically identical) set of rows. No further duplication needed.

    # KL-grid frontier
    if args.include_kl_frontier:
        print("[eval] KL-grid frontier...", flush=True)
        for spec in kl_grid_specs:
            theta = _theta_for(spec)
            m_at_pstar = evaluate_cell(theta, cache, p_obs_np=None, beta=args.beta)
            theta_norm = float(torch.norm(theta).item())
            rows.append({
                "seed": spec.seed,
                "method": spec.method,
                "eta_kl": spec.eta_kl,
                "rho_alg": spec.rho_alg,
                "rho_mis": args.rho_mis,
                "m": 0,
                "direction_kind": "kl_grid_nominal",
                "alpha": 0.0,
                "shift_loss": m_at_pstar["L_offdiag"],
                "L_offdiag": m_at_pstar["L_offdiag"],
                "L_exact": m_at_pstar["L_exact"],
                "L_SAIL_true": m_at_pstar["L_offdiag"],
                "delta_shift": 0.0,
                "sensitivity_R": m_at_pstar["sensitivity_R"],
                "R_eval": m_at_pstar["sensitivity_R"],
                "E_rstar": m_at_pstar["E_rstar"],
                "KL_to_ref": m_at_pstar["KL_to_ref"],
                "bias_active_rate": float("nan"),
                "max_abs_b": float("nan"),
                "theta_norm": theta_norm,
            })
            for m in range(1, args.num_directions + 1):
                pert = perturbations[(spec.seed, m)]
                cell = evaluate_cell(theta, cache, p_obs_np=pert.p_obs,
                                     beta=args.beta)
                rows.append({
                    "seed": spec.seed,
                    "method": spec.method,
                    "eta_kl": spec.eta_kl,
                    "rho_alg": spec.rho_alg,
                    "rho_mis": args.rho_mis,
                    "m": m,
                    "direction_kind": "kl_grid_random",
                    "alpha": 1.0,
                    "shift_loss": cell["L_offdiag"],
                    "L_offdiag": cell["L_offdiag"],
                    "L_exact": cell["L_exact"],
                    "L_SAIL_true": m_at_pstar["L_offdiag"],
                    "delta_shift": cell["L_offdiag"] - m_at_pstar["L_offdiag"],
                    "sensitivity_R": cell["sensitivity_R"],
                    "R_eval": m_at_pstar["sensitivity_R"],
                    "E_rstar": cell["E_rstar"],
                    "KL_to_ref": cell["KL_to_ref"],
                    "bias_active_rate": pert.bias_active_rate,
                    "max_abs_b": pert.max_abs_b,
                    "theta_norm": theta_norm,
                })

    # ---------------- Persist artifacts ----------------
    print("[write] artifacts...", flush=True)
    write_per_run_csv(out_dir / "per_run_loss.csv", rows)
    write_retry_csv(out_dir / "retry_log.csv", retry_rows)
    _save_perturbations_npz(
        out_dir / "perturbations.npz",
        perturbations, args.seeds, args.num_directions, cache["d"],
        args.include_axis_aligned, cache["N"], cache["K"],
    )
    write_config_yaml(out_dir / "config.yaml", args, _git_commit())

    # ---------------- Aggregation & plots ----------------
    print("[aggregate] summaries + plots...", flush=True)
    summary_per_method = compute_summary_per_method(rows, args.rho_mis)
    write_summary_per_method_csv(out_dir / "summary_per_method.csv",
                                  summary_per_method)
    summary_paired = compute_summary_paired(rows, args.rho_mis)
    write_summary_paired_csv(out_dir / "summary_paired.csv", summary_paired)

    plot_pareto(rows, out_dir / "plots", args.rho_mis,
                include_kl_frontier=args.include_kl_frontier)
    plot_paired_delta_histogram(rows, out_dir / "plots", args.rho_mis)
    plot_per_seed_traces(rows, out_dir / "plots", args.rho_mis)

    write_readme(out_dir / "README.md", summary_per_method, summary_paired,
                 retry_rows, args)

    # ---------------- Acceptance checks ----------------
    print("[acceptance] running checks...", flush=True)
    run_acceptance_checks(rows, perturbations, cache, main_specs, args)
    print("[acceptance] PASS", flush=True)
    return 0


# ---- CSV / config writers --------------------------------------------------
PER_RUN_FIELDS = [
    "seed", "method", "eta_kl", "rho_alg", "rho_mis", "m", "direction_kind",
    "alpha", "shift_loss", "L_offdiag", "L_exact", "L_SAIL_true",
    "delta_shift", "sensitivity_R", "R_eval", "E_rstar", "KL_to_ref",
    "bias_active_rate", "max_abs_b", "theta_norm",
]


def write_per_run_csv(path: Path, rows: List[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_RUN_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in PER_RUN_FIELDS})


RETRY_FIELDS = ["seed", "m", "attempt", "namespace",
                "bias_active_rate", "max_abs_b", "ok"]


def write_retry_csv(path: Path, rows: List[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RETRY_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in RETRY_FIELDS})


def write_config_yaml(path: Path, args: argparse.Namespace,
                       git_commit: str) -> None:
    cfg = {
        "namespace": RNG_NAMESPACE,
        "M": args.num_directions,
        "seeds": list(args.seeds),
        "rho_mis": [args.rho_mis],
        "methods": ["Nominal", "Robust-0.025rho", "KL-matchR", "KL-matchEntropy"],
        "construction": (
            "random_unit_direction_in_feature_space; "
            "s_held(x,y) = u^T psi(x,y); "
            "b' = compute_pairwise_bias(true_reward, s_held) "
            "(activation rule and per-prompt-normalized magnitude match training)"
        ),
        "activation_density_threshold": BIAS_ACTIVE_RATE_FLOOR,
        "max_resamples_per_direction": RETRY_MAX,
        "alpha_evaluated": [1.0],
        "include_axis_aligned": bool(args.include_axis_aligned),
        "include_kl_frontier": bool(args.include_kl_frontier),
        "kl_eta_grid": KL_ETA_GRID,
        "kl_match_eta": KL_MATCH_ETA,
        "robust_rho_alg_ratio": ROBUST_RHO_ALG_RATIO,
        "beta": args.beta,
        "tau": args.tau,
        "K": args.K,
        "featureset": args.featureset,
        "bias_attr": args.bias_attr,
        "ref": args.ref,
        "git_commit": git_commit,
    }
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ---- Aggregation -----------------------------------------------------------
import statistics  # noqa: E402


def _se(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values) / math.sqrt(len(values)))


def compute_summary_per_method(rows: List[dict],
                               rho_mis: float) -> List[dict]:
    """Per-method aggregates across seeds.

    Per spec §5.1: WorstShift_per_seed = max_m shift_loss(s, method, m) over
    the M=16 random directions only. AvgShift_per_seed = mean over the same
    16. Axis-aligned directions (spec §3.4 sanity set) are reported in
    parallel `*_axis` columns; they are not pooled into the primary metric.
    """
    methods = ["Nominal", "Robust-0.025rho", "KL-matchR", "KL-matchEntropy"]
    out: List[dict] = []
    for method in methods:
        # Nominal-row-only metrics (m=0, direction_kind="nominal")
        nominal = [r for r in rows
                   if r["method"] == method and r["direction_kind"] == "nominal"]
        if not nominal:
            continue
        seeds = sorted({r["seed"] for r in nominal})
        E_rstar_per_seed = [next(r["E_rstar"] for r in nominal
                                 if r["seed"] == s) for s in seeds]
        R_per_seed = [next(r["R_eval"] for r in nominal
                           if r["seed"] == s) for s in seeds]
        KL_per_seed = [next(r["KL_to_ref"] for r in nominal
                            if r["seed"] == s) for s in seeds]
        L_true_per_seed = [next(r["L_SAIL_true"] for r in nominal
                                if r["seed"] == s) for s in seeds]

        def _worst_avg_per_seed(direction_kind: str) -> Tuple[List[float], List[float]]:
            shift = [r for r in rows
                     if r["method"] == method
                     and r["direction_kind"] == direction_kind]
            worst, avg = [], []
            for s in seeds:
                sm = [r["shift_loss"] for r in shift if r["seed"] == s]
                if sm:
                    worst.append(max(sm))
                    avg.append(sum(sm) / len(sm))
            return worst, avg

        worst_rand, avg_rand = _worst_avg_per_seed("random")
        worst_axis, avg_axis = _worst_avg_per_seed("axis")

        out.append({
            "method": method,
            "rho_mis": rho_mis,
            "n_seeds": len(seeds),
            "mean_E_rstar": statistics.mean(E_rstar_per_seed),
            "SE_E_rstar": _se(E_rstar_per_seed),
            "mean_R_eval": statistics.mean(R_per_seed),
            "SE_R_eval": _se(R_per_seed),
            "mean_KL_to_ref": statistics.mean(KL_per_seed),
            "SE_KL_to_ref": _se(KL_per_seed),
            "mean_L_SAIL_true": statistics.mean(L_true_per_seed),
            "SE_L_SAIL_true": _se(L_true_per_seed),
            # Primary: random-only (spec §5.1)
            "mean_WorstShift": (statistics.mean(worst_rand)
                                 if worst_rand else float("nan")),
            "SE_WorstShift": _se(worst_rand),
            "mean_AvgShift": (statistics.mean(avg_rand)
                               if avg_rand else float("nan")),
            "SE_AvgShift": _se(avg_rand),
            # Secondary: axis-aligned sanity (spec §3.4)
            "mean_WorstShift_axis": (statistics.mean(worst_axis)
                                      if worst_axis else float("nan")),
            "SE_WorstShift_axis": _se(worst_axis),
            "mean_AvgShift_axis": (statistics.mean(avg_axis)
                                    if avg_axis else float("nan")),
            "SE_AvgShift_axis": _se(avg_axis),
        })
    return out


def write_summary_per_method_csv(path: Path, summary: List[dict]) -> None:
    if not summary:
        path.write_text("")
        return
    fields = list(summary[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary:
            writer.writerow(r)


def compute_summary_paired(rows: List[dict], rho_mis: float) -> List[dict]:
    """Paired-seed comparisons: Robust-0.025rho vs {Nominal, KL-matchR, KL-matchEntropy}."""
    try:
        from scipy import stats
    except ImportError:
        stats = None  # type: ignore

    methods_b = ["Nominal", "KL-matchR", "KL-matchEntropy"]
    method_a = "Robust-0.025rho"
    out: List[dict] = []

    # Per spec §5.3: paired comparisons use random directions only. Build
    # WorstShift / AvgShift over random-only and emit axis-aligned variants
    # as parallel rows tagged by `direction_set`.
    def _per_seed_metric(method: str, direction_kind: str
                         ) -> Dict[str, Dict[int, float]]:
        nominal = [r for r in rows
                   if r["method"] == method and r["direction_kind"] == "nominal"]
        shift = [r for r in rows
                 if r["method"] == method
                 and r["direction_kind"] == direction_kind]
        seeds = sorted({r["seed"] for r in nominal})
        worst = {}
        avg = {}
        R = {}
        for s in seeds:
            sm = [r["shift_loss"] for r in shift if r["seed"] == s]
            if sm:
                worst[s] = max(sm)
                avg[s] = sum(sm) / len(sm)
            for r in nominal:
                if r["seed"] == s:
                    R[s] = r["R_eval"]
                    break
        return {"WorstShift": worst, "AvgShift": avg, "R_eval": R}

    for direction_set in ("random", "axis"):
        a = _per_seed_metric(method_a, direction_set)
        for mb in methods_b:
            b = _per_seed_metric(mb, direction_set)
            for metric in ["WorstShift", "AvgShift", "R_eval"]:
                common = sorted(set(a[metric].keys()) & set(b[metric].keys()))
                deltas = [a[metric][s] - b[metric][s] for s in common]
                n = len(deltas)
                if n == 0:
                    continue
                mean_d = statistics.mean(deltas)
                se_d = _se(deltas)
                n_better = sum(1 for d in deltas if d < 0)  # smaller-is-better
                pt_p = float("nan")
                wlcx_p = float("nan")
                if stats is not None and n >= 2:
                    try:
                        pt_p = float(stats.ttest_rel(
                            [a[metric][s] for s in common],
                            [b[metric][s] for s in common],
                        ).pvalue)
                    except Exception:
                        pass
                    try:
                        if any(abs(d) > 0 for d in deltas):
                            wlcx_p = float(stats.wilcoxon(deltas).pvalue)
                    except Exception:
                        pass
                out.append({
                    "method_A": method_a,
                    "method_B": mb,
                    "rho_mis": rho_mis,
                    "direction_set": direction_set,
                    "metric": metric,
                    "n_seeds": n,
                    "mean_delta": mean_d,
                    "SE_delta": se_d,
                    "n_seeds_A_better": n_better,
                    "paired_t_pvalue": pt_p,
                    "wilcoxon_pvalue": wlcx_p,
                })
    return out


def write_summary_paired_csv(path: Path, summary: List[dict]) -> None:
    if not summary:
        path.write_text("")
        return
    fields = list(summary[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary:
            writer.writerow(r)


# ---- Plots -----------------------------------------------------------------
def _safe_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


METHOD_COLORS = {
    "Nominal": "#4C72B0",
    "Robust-0.025rho": "#C44E52",
    "KL-matchR": "#55A868",
    "KL-matchEntropy": "#8172B2",
    "KL-grid": "#999999",
}


def plot_pareto(rows: List[dict], plot_dir: Path, rho_mis: float,
                include_kl_frontier: bool) -> None:
    plt = _safe_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

    # Compute method markers (mean ± SE across seeds). Random-only WorstShift
    # per spec §5.4 (axis-aligned is reported separately, not pooled here).
    for method in ["Nominal", "Robust-0.025rho", "KL-matchR", "KL-matchEntropy"]:
        method_rows = [r for r in rows if r["method"] == method]
        nominal = [r for r in method_rows if r["direction_kind"] == "nominal"]
        seeds = sorted({r["seed"] for r in nominal})
        if not seeds:
            continue
        R_per_seed = [next(r["R_eval"] for r in nominal if r["seed"] == s)
                      for s in seeds]
        worst_per_seed = []
        for s in seeds:
            sm = [r["shift_loss"] for r in method_rows
                  if r["seed"] == s
                  and r["direction_kind"] == "random"]
            if sm:
                worst_per_seed.append(max(sm))
        if not worst_per_seed:
            continue
        x = statistics.mean(R_per_seed)
        y = statistics.mean(worst_per_seed)
        xerr = _se(R_per_seed)
        yerr = _se(worst_per_seed)
        ax.errorbar([x], [y], xerr=[xerr], yerr=[yerr],
                    marker="o", linestyle="", color=METHOD_COLORS[method],
                    label=method, capsize=3, markersize=7)

    # KL-grid frontier
    if include_kl_frontier:
        kl_grid = [r for r in rows if r["method"].startswith("KL-grid-eta")]
        etas = sorted({r["eta_kl"] for r in kl_grid})
        xs, ys = [], []
        for eta in etas:
            method_label = f"KL-grid-eta{eta:g}"
            method_rows = [r for r in kl_grid if r["method"] == method_label]
            nominal = [r for r in method_rows
                       if r["direction_kind"] == "kl_grid_nominal"]
            seeds = sorted({r["seed"] for r in nominal})
            R_per_seed = [next(r["R_eval"] for r in nominal if r["seed"] == s)
                          for s in seeds]
            worst_per_seed = []
            for s in seeds:
                sm = [r["shift_loss"] for r in method_rows
                      if r["seed"] == s
                      and r["direction_kind"] == "kl_grid_random"]
                if sm:
                    worst_per_seed.append(max(sm))
            if R_per_seed and worst_per_seed:
                xs.append(statistics.mean(R_per_seed))
                ys.append(statistics.mean(worst_per_seed))
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        ax.plot(xs, ys, "-", color=METHOD_COLORS["KL-grid"],
                label="KL grid (varying eta)", alpha=0.8, linewidth=1.5)

    ax.set_xlabel("R_eval (sensitivity at p*)")
    ax.set_ylabel("WorstShift_held_out (mean over seeds)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"pareto_held_out.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def plot_paired_delta_histogram(rows: List[dict], plot_dir: Path,
                                  rho_mis: float) -> None:
    plt = _safe_matplotlib()
    # Per spec §5.5: histogram of the 10 seeds × M random directions only
    # (axis-aligned directions are reported separately, not pooled here).
    def _by_method(method: str) -> Dict[Tuple[int, int], float]:
        return {(r["seed"], r["m"]): r["shift_loss"]
                for r in rows
                if r["method"] == method
                and r["direction_kind"] == "random"}

    a = _by_method("Robust-0.025rho")
    b = _by_method("KL-matchR")
    keys = sorted(set(a.keys()) & set(b.keys()))
    deltas = [a[k] - b[k] for k in keys]
    if not deltas:
        return

    frac_neg = sum(1 for d in deltas if d < 0) / len(deltas)
    try:
        from scipy import stats
        wp = float(stats.wilcoxon(deltas).pvalue) if any(abs(d) > 0 for d in deltas) else float("nan")
    except Exception:
        wp = float("nan")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.hist(deltas, bins=30, color="#888888", edgecolor="black", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\Delta = $ shift_loss(Robust-0.025$\rho$) - shift_loss(KL-matchR)")
    ax.set_ylabel("count")
    ax.text(0.02, 0.95,
            f"n={len(deltas)}\nfraction<0: {frac_neg:.2f}\nWilcoxon p: {wp:.3g}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"paired_delta_histogram.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def plot_per_seed_traces(rows: List[dict], plot_dir: Path,
                          rho_mis: float) -> None:
    plt = _safe_matplotlib()
    methods = ["Nominal", "Robust-0.025rho", "KL-matchR", "KL-matchEntropy"]
    seeds = sorted({r["seed"] for r in rows
                    if r["method"] == "Nominal"
                    and r["direction_kind"] == "nominal"})
    n_seeds = len(seeds)
    if n_seeds == 0:
        return

    def _render(direction_kind: str, file_stem: str, xlabel_suffix: str) -> None:
        fig, axes = plt.subplots(n_seeds, 1, figsize=(7, 1.8 * n_seeds),
                                 dpi=300, sharex=True)
        if n_seeds == 1:
            axes = [axes]
        any_data = False
        for ax, s in zip(axes, seeds):
            nominal_dirs = [r for r in rows
                            if r["seed"] == s
                            and r["method"] == "Nominal"
                            and r["direction_kind"] == direction_kind]
            nominal_dirs_sorted = sorted(nominal_dirs,
                                         key=lambda r: r["shift_loss"])
            m_order = [r["m"] for r in nominal_dirs_sorted]
            x = list(range(len(m_order)))
            if x:
                any_data = True
            for method in methods:
                method_dirs = {r["m"]: r["shift_loss"]
                               for r in rows
                               if r["seed"] == s and r["method"] == method
                               and r["direction_kind"] == direction_kind}
                y = [method_dirs.get(m, float("nan")) for m in m_order]
                ax.plot(x, y, marker=".", color=METHOD_COLORS[method],
                        label=method, linewidth=1, markersize=4)
            ax.set_ylabel(f"seed {s}\nshift_loss", fontsize=8)
            ax.grid(True, alpha=0.3)
        if not any_data:
            plt.close(fig)
            return
        axes[0].legend(loc="best", fontsize=7, ncol=2)
        axes[-1].set_xlabel(
            f"perturbation index ({xlabel_suffix}, sorted by Nominal loss)"
        )
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(plot_dir / f"{file_stem}.{ext}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)

    # Primary: 16 random directions (spec §5.4)
    _render("random", "per_seed_traces", "16 random directions")
    # Secondary: 7 axis-aligned directions (spec §3.4)
    _render("axis", "per_seed_traces_axis", "axis-aligned e_1..e_d")


# ---- README ----------------------------------------------------------------
def _fmt(v, p=4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{p}f}"
    return str(v)


def write_readme(path: Path, summary_per_method: List[dict],
                  summary_paired: List[dict], retry_rows: List[dict],
                  args: argparse.Namespace) -> None:
    # Outcome classification: based on (Robust-0.025rho - KL-matchR) WorstShift
    # paired-t over the random directions only (primary metric per spec §5.1).
    outcome_row = next(
        (r for r in summary_paired
         if r["method_A"] == "Robust-0.025rho"
         and r["method_B"] == "KL-matchR"
         and r["metric"] == "WorstShift"
         and r.get("direction_set") == "random"),
        None,
    )
    if outcome_row is None:
        outcome_sentence = "(no Robust vs KL-matchR comparison available.)"
    else:
        mean_d = outcome_row["mean_delta"]
        p = outcome_row["paired_t_pvalue"]
        n_better = outcome_row["n_seeds_A_better"]
        if (isinstance(p, float) and math.isfinite(p) and p < 0.05
                and mean_d < 0):
            outcome_sentence = (
                f"**Outcome B**: Robust-0.025ρ reduces worst-case "
                f"held-out shift loss by Δ = {mean_d:.5f} "
                f"(paired-t p = {p:.3g}, n_seeds_A_better = {n_better}/"
                f"{outcome_row['n_seeds']}) over tuned KL-matchR. "
                f"The empirical distinction predicted by the theory is "
                f"observable on held-out perturbation directions."
            )
        else:
            outcome_sentence = (
                f"**Outcome A**: Robust-0.025ρ and KL-matchR remain "
                f"statistically indistinguishable on the held-out family "
                f"(Δ_WorstShift = {mean_d:.5f}, paired-t p = "
                f"{_fmt(p, 3)}, n_seeds_A_better = {n_better}/"
                f"{outcome_row['n_seeds']}). The empirical equivalence "
                f"reported in the main text extends to held-out directions."
            )

    n_retries = sum(1 for r in retry_rows if not r["ok"])
    retry_summary = "No retries needed." if n_retries == 0 else (
        f"{n_retries} direction-attempts failed the activation-density floor "
        f"of {BIAS_ACTIVE_RATE_FLOOR} and were resampled (see retry_log.csv)."
    )

    n_random = args.num_directions
    has_axis = bool(args.include_axis_aligned)

    lines: List[str] = []
    lines.append("# Held-Out Perturbation Robustness — Results\n")
    lines.append("")
    lines.append("## 1. What this experiment did\n")
    lines.append(
        f"For each training seed s in {list(args.seeds)} and each "
        f"direction index m = 1..{n_random}, drew a unit-norm u in R^d "
        f"(d = feature dimension; controlled featureset, d=7) from a "
        f"deterministic blake2b-seeded RNG (namespace `{RNG_NAMESPACE}`). "
        f"Built a held-out perturbation b' identically to the training-time "
        f"construction (`finite_response_sail/oracle_misspec.py:"
        f"compute_pairwise_bias`), substituting `s_held(x,y) = u^T psi(x,y)` "
        f"for the training redundancy attribute. Evaluated all four "
        f"already-trained policies (Nominal, Robust-0.025ρ, KL-matchR, "
        f"KL-matchEntropy) on these held-out perturbations at α = 1 "
        f"(full magnitude). No retraining.\n"
    )
    lines.append("## 2. Construction of b'_{m, ij}(x)\n")
    lines.append(
        "```\n"
        "ds_ij = s_held(x, y_i) - s_held(x, y_j)\n"
        "dr_ij = r*(x, y_i) - r*(x, y_j)\n"
        "M_x   = max_{u,v} |s_held(x, y_u) - s_held(x, y_v)| + eps\n"
        "b'_ij = sign(ds_ij) * |ds_ij|/M_x * 1[ds_ij*dr_ij <= 0]\n"
        "```\n"
        "Activation rule and per-prompt-normalized magnitude are exactly "
        "as in the training b (see "
        "finite_response_sail/oracle_misspec.py lines 61–78). Only the "
        "bias direction differs: u replaces the redundancy attribute.\n"
    )
    # Convention block: how each statistic / plot uses random vs axis dirs.
    lines.append("## 3. Direction sets and which goes where\n")
    lines.append(
        f"Two perturbation families were generated per training seed:\n\n"
        f"- **Random** (primary, spec §3.1): {n_random} unit-norm directions "
        f"u ∈ R^d drawn uniformly on the sphere via `blake2b('held_out_v1', "
        f"s, m)`. **All headline statistics — `mean_WorstShift`, "
        f"`mean_AvgShift`, the paired-t/Wilcoxon p-values in §5, the Pareto "
        f"plot, the paired-Δ histogram, and `per_seed_traces.png` — are "
        f"computed from these {n_random} directions only.**\n"
    )
    if has_axis:
        lines.append(
            f"- **Axis-aligned** (secondary sanity, spec §3.4): the d=7 "
            f"standard-basis directions e_1..e_7 per seed. "
            f"Reported in parallel `*_axis` columns of "
            f"`summary_per_method.csv`, in `summary_paired.csv` rows tagged "
            f"`direction_set=axis`, and in `per_seed_traces_axis.png`. "
            f"**Not pooled into the primary metric.**\n"
        )
    else:
        lines.append(
            "- **Axis-aligned** (secondary): not enabled in this run "
            "(--no-axis-aligned).\n"
        )
    lines.append("")
    lines.append("## 4. Per-method summary on the {} random directions (mean ± SE over seeds)\n".format(n_random))
    lines.append(
        "| method | E[r*] | R_eval | L_SAIL_true | mean WorstShift (random) | mean AvgShift (random) |\n"
        "|---|---|---|---|---|---|\n"
    )
    for r in summary_per_method:
        lines.append(
            f"| {r['method']} "
            f"| {_fmt(r['mean_E_rstar'])} ± {_fmt(r['SE_E_rstar'])} "
            f"| {_fmt(r['mean_R_eval'])} ± {_fmt(r['SE_R_eval'])} "
            f"| {_fmt(r['mean_L_SAIL_true'])} ± {_fmt(r['SE_L_SAIL_true'])} "
            f"| {_fmt(r['mean_WorstShift'])} ± {_fmt(r['SE_WorstShift'])} "
            f"| {_fmt(r['mean_AvgShift'])} ± {_fmt(r['SE_AvgShift'])} |\n"
        )
    if has_axis:
        lines.append("\n### 4a. Per-method summary on the 7 axis-aligned directions (sanity)\n")
        lines.append(
            "| method | mean WorstShift (axis) | mean AvgShift (axis) |\n"
            "|---|---|---|\n"
        )
        for r in summary_per_method:
            lines.append(
                f"| {r['method']} "
                f"| {_fmt(r.get('mean_WorstShift_axis'))} "
                f"± {_fmt(r.get('SE_WorstShift_axis'))} "
                f"| {_fmt(r.get('mean_AvgShift_axis'))} "
                f"± {_fmt(r.get('SE_AvgShift_axis'))} |\n"
            )
    lines.append("")
    lines.append(
        "## 5. Paired Robust-0.025ρ vs (Nominal, KL-matchR, KL-matchEntropy) "
        "— random directions (primary)\n"
    )
    lines.append(
        "| metric | vs | mean Δ | SE Δ | n_A_better | paired-t p | Wilcoxon p |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    for r in summary_paired:
        if r.get("direction_set") != "random":
            continue
        lines.append(
            f"| {r['metric']} | {r['method_B']} "
            f"| {_fmt(r['mean_delta'])} | {_fmt(r['SE_delta'])} "
            f"| {r['n_seeds_A_better']}/{r['n_seeds']} "
            f"| {_fmt(r['paired_t_pvalue'], 3)} "
            f"| {_fmt(r['wilcoxon_pvalue'], 3)} |\n"
        )
    if has_axis:
        lines.append("\n### 5a. Same paired comparisons on axis-aligned directions (sanity)\n")
        lines.append(
            "| metric | vs | mean Δ | SE Δ | n_A_better | paired-t p | Wilcoxon p |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        for r in summary_paired:
            if r.get("direction_set") != "axis":
                continue
            lines.append(
                f"| {r['metric']} | {r['method_B']} "
                f"| {_fmt(r['mean_delta'])} | {_fmt(r['SE_delta'])} "
                f"| {r['n_seeds_A_better']}/{r['n_seeds']} "
                f"| {_fmt(r['paired_t_pvalue'], 3)} "
                f"| {_fmt(r['wilcoxon_pvalue'], 3)} |\n"
            )
    lines.append("")
    lines.append("## 6. Outcome (random-direction, primary)\n")
    lines.append(outcome_sentence + "\n")
    lines.append("\n## 7. Anomalies\n")
    lines.append(retry_summary + "\n")
    lines.append("\n## 8. Reproduction\n")
    lines.append(
        "```\n"
        "python -m finite_response_sail.held_out_perturbation \\\n"
        f"    --rho-mis {args.rho_mis} --num-directions {args.num_directions} \\\n"
        f"    --seeds {' '.join(str(s) for s in args.seeds)} \\\n"
        f"    {'--include-axis-aligned ' if args.include_axis_aligned else ''}"
        f"{'--include-kl-frontier ' if args.include_kl_frontier else ''}\\\n"
        f"    --out-dir {args.out_dir}\n"
        "```\n"
    )

    path.write_text("".join(lines))


# ---- Acceptance checks -----------------------------------------------------
def run_acceptance_checks(
    rows: List[dict],
    perturbations: Dict[Tuple[int, int], Perturbation],
    cache: Dict[str, object],
    main_specs: List[RunSpec],
    args: argparse.Namespace,
) -> None:
    # 1. No NaN in shift_loss / R_eval / E_rstar
    for r in rows:
        for col in ("shift_loss", "R_eval", "E_rstar", "L_SAIL_true"):
            v = r.get(col)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                raise AssertionError(
                    f"acceptance: NaN/None in {col} at row "
                    f"(method={r['method']}, seed={r['seed']}, m={r['m']})"
                )

    # 2. |b| <= 1 for every perturbation
    for (s, m), p in perturbations.items():
        max_abs = float(np.abs(p.b).max())
        if max_abs > 1.0 + 1e-10:
            raise AssertionError(
                f"acceptance: |b|.max()={max_abs} > 1 at (s={s}, m={m})"
            )

    # 3. Antisymmetry on one perturbation
    p_sample = next(iter(perturbations.values()))
    asym = float(np.abs(p_sample.b + p_sample.b.transpose(0, 2, 1)).max())
    if asym > 1e-10:
        raise AssertionError(
            f"acceptance: bias antisymmetry violated, max|b+b^T|={asym}"
        )

    # 4. Bias-active-rate floor (post-hoc)
    for (s, m), p in perturbations.items():
        if p.direction_kind == "random" and p.bias_active_rate < BIAS_ACTIVE_RATE_FLOOR:
            raise AssertionError(
                f"acceptance: random direction (s={s}, m={m}) has "
                f"bias_active_rate {p.bias_active_rate:.4f} < floor"
            )

    # 5. Alpha-affineness on one cell: pick (Nominal, seed=args.seeds[0], m=1).
    s0 = args.seeds[0]
    nominal_spec = next(spec for spec in main_specs
                        if spec.method == "Nominal" and spec.seed == s0)
    theta = torch.load(nominal_spec.run_dir / "theta.pt",
                       map_location=cache["device"], weights_only=False).float()
    pert = perturbations[(s0, 1)]
    L0 = evaluate_cell(theta, cache, p_obs_np=None, beta=args.beta)["L_offdiag"]
    L1 = evaluate_cell(theta, cache, p_obs_np=pert.p_obs,
                       beta=args.beta)["L_offdiag"]
    p_half = cache["p_star_np"] + 0.5 * args.rho_mis * pert.b / args.rho_mis
    # the above is just (p_star + 0.5 * b * rho_mis); rewrite cleanly:
    p_half = cache["p_star_np"] + 0.5 * (pert.p_obs - cache["p_star_np"])
    L_half = evaluate_cell(theta, cache, p_obs_np=p_half,
                           beta=args.beta)["L_offdiag"]
    predicted = 0.5 * (L0 + L1)
    if abs(L_half - predicted) > 1e-5:
        raise AssertionError(
            f"acceptance: alpha-affineness violated: L(0.5)={L_half}, "
            f"predicted=(L0+L1)/2={predicted}, |diff|={abs(L_half-predicted)}"
        )

    # 6. R_eval(Robust-0.025rho) ~ R_eval(KL-matchR) within 5%
    nominal_rows = [r for r in rows
                    if r["direction_kind"] == "nominal"
                    and r["method"] in ("Robust-0.025rho", "KL-matchR")]
    seeds = sorted({r["seed"] for r in nominal_rows})
    for s in seeds:
        rA = next(r for r in nominal_rows
                  if r["method"] == "Robust-0.025rho" and r["seed"] == s)
        rB = next(r for r in nominal_rows
                  if r["method"] == "KL-matchR" and r["seed"] == s)
        if abs(rA["R_eval"] - rB["R_eval"]) / max(rA["R_eval"], 1e-9) > 0.05:
            print(
                f"[acceptance] WARN: seed {s}: R(Robust)={rA['R_eval']:.4f} "
                f"vs R(KL-matchR)={rB['R_eval']:.4f} differ by >5%",
                file=sys.stderr,
            )


# ---- CLI -------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--rho-mis", type=float, default=DEFAULT_RHO_MIS)
    p.add_argument("--num-directions", type=int, default=DEFAULT_NUM_DIRECTIONS)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=list(range(DEFAULT_NUM_SEEDS)))
    p.add_argument("--include-axis-aligned", dest="include_axis_aligned",
                   action="store_true", default=True)
    p.add_argument("--no-axis-aligned", dest="include_axis_aligned",
                   action="store_false")
    p.add_argument("--include-kl-frontier", dest="include_kl_frontier",
                   action="store_true", default=True)
    p.add_argument("--no-kl-frontier", dest="include_kl_frontier",
                   action="store_false")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--confirm-dir", default=str(DEFAULT_CONFIRM_DIR))
    p.add_argument("--kl-dir", default=str(DEFAULT_KL_DIR))
    p.add_argument("--featureset", default=DEFAULT_FEATURESET)
    p.add_argument("--bias-attr", default=DEFAULT_BIAS_ATTR)
    p.add_argument("--ref", default=DEFAULT_REF)
    p.add_argument("--beta", type=float, default=DEFAULT_BETA)
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--K", type=int, default=DEFAULT_K)
    p.add_argument("--smoke", action="store_true",
                   help="seeds=[0,1], num-directions=4, no kl-frontier")
    return p


if __name__ == "__main__":
    sys.exit(main())
