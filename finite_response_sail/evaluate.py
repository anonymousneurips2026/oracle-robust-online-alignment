"""Evaluation on val/test for the finite-response online SAIL policy (v3_pairbias).

Reports all metrics required by the v3 plan. Uses cached psi/ref_logits/true_reward.
The oracle passed in is (p_star, b, p_obs); no clipping, no p_star_tilde.
"""
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .objective import (
    exact_expectation_loss,
    expected_bias_attr,
    expected_true_utility,
    expected_true_win_vs_ref,
    label_entropy as _label_entropy,
    mean_abs_h_sampled as _mean_abs_h_sampled,
    observed_win_vs_ref as _observed_win_vs_ref,
    pair_bias_mass_under_pi as _pair_bias_mass,
    sensitivity_R,
    signed_pair_bias_under_pi as _signed_pair_bias,
    worst_case_loss,
)
from .offdiag_objective import offdiag_denominator_stats
from .oracle_misspec import oracle_diagnostics
from .policy import (
    diagonal_mass,
    kl_to_ref as _kl_to_ref,
    policy_entropy,
    policy_probs,
    top1_prob,
)


@dataclass
class EvalMetrics:
    # policy-level metrics
    sensitivity_R: float
    worst_case_loss: float
    expected_true_utility: float
    expected_bias_attr: float
    entropy: float
    top1_prob: float
    diagonal_mass: float
    theta_norm: float
    # correlations
    candidate_base_true_corr: float
    bias_attr_true_corr: float
    bias_attr_true_corr_median: float
    # oracle diagnostics
    delta_empirical: float
    min_p_star: float
    max_p_star: float
    max_abs_p_obs_minus_p_star: float
    mean_abs_p_obs_minus_p_star: float
    max_abs_b: float
    mean_abs_b: float
    bias_active_rate: float
    bias_conflict_rate: float
    bias_same_direction_zero_rate: float
    pair_bias_mass_under_pi: float
    signed_pair_bias_under_pi: float
    # offdiag denominator diagnostics
    offdiag_denominator_min: float
    offdiag_denominator_mean: float
    # directional policy-weighted diagnostics (not used by any objective)
    bad_bias_candidate_mass: float
    good_candidate_mass: float
    # wins and labels
    expected_true_win_vs_ref: float
    observed_win_vs_ref: float
    label_entropy: float
    mean_abs_h_sampled: float
    greedy_true_win_vs_ref: float
    greedy_true_utility: float
    greedy_bias_attr: float
    # training-objective reference values (no penalty)
    nominal_bt_loss: float   # BT NLL at p_center = p_star (raw, no clipping)
    biased_bt_loss: float    # BT NLL at p_center = p_obs
    # Paper decomposition-form robust objectives (derived). Under the margin
    # condition [p* - rho, p* + rho] in [0, 1], these equal the endpoint-max
    # worst_case_loss to numerical tolerance; see objective.worst_case_loss.
    #   L^W_rho(theta) = L_SAIL(theta; P*) + rho * beta * R(theta)
    wcl_true_rho_mis: float       # nominal_bt_loss + rho_mis * beta * sensitivity_R
    wcl_true_rho_alg: float       # nominal_bt_loss + rho_alg * beta * sensitivity_R
    biased_feedback_loss: float   # alias of biased_bt_loss; BT loss at p_obs (no penalty)
    # KL(pi_theta || pi_ref) averaged over prompts (load-bearing for the
    # KL-smoothing baseline; recorded for every run so all methods are comparable).
    KL_to_ref: float


def _corr_per_prompt(a: np.ndarray, b: np.ndarray) -> float:
    out = []
    for n in range(a.shape[0]):
        x, y = a[n], b[n]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue
        out.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(out)) if out else 0.0


def _greedy_pick(probs: torch.Tensor) -> torch.Tensor:
    return probs.argmax(dim=-1)


def _bad_good_masses(
    probs: torch.Tensor, bias_attr: torch.Tensor, true_rewards: torch.Tensor,
) -> Tuple[float, float]:
    """Policy-weighted masses on {high-bias, low-true} and {low-bias, high-true}
    candidates, averaged over prompts. Medians are computed per-prompt over the K
    candidates; `>=`/`<=` gives an inclusive split that is robust to ties.

    Returns (bad_bias_candidate_mass, good_candidate_mass).
    """
    ba = bias_attr.double()
    tr = true_rewards.double()
    med_s = ba.median(dim=-1, keepdim=True).values
    med_r = tr.median(dim=-1, keepdim=True).values
    bad = ((ba >= med_s) & (tr <= med_r)).to(probs.dtype)
    good = ((ba <= med_s) & (tr >= med_r)).to(probs.dtype)
    bad_mass = (probs * bad).sum(dim=-1).mean().item()
    good_mass = (probs * good).sum(dim=-1).mean().item()
    return float(bad_mass), float(good_mass)


@torch.no_grad()
def evaluate(
    theta: torch.Tensor,
    psi: torch.Tensor,
    ref_logits: torch.Tensor,
    true_rewards: torch.Tensor,
    bias_attr: torch.Tensor,
    base_logprob_per_token: torch.Tensor,
    p_star: torch.Tensor,
    b: torch.Tensor,
    p_obs: torch.Tensor,
    tau: float,
    rho_mis: float,
    beta: float,
    rho_alg: float = 0.0,
) -> EvalMetrics:
    probs = policy_probs(theta, psi, ref_logits)                 # [B, K]
    pi_ref = torch.softmax(ref_logits, dim=-1)

    entropy = policy_entropy(probs).mean().item()
    top1 = top1_prob(probs).mean().item()
    dm = diagonal_mass(probs).mean().item()
    theta_norm = float(theta.norm(p=2).item())

    s_R = sensitivity_R(theta, psi, ref_logits)
    wcl = worst_case_loss(theta, psi, ref_logits, p_star, rho_mis, beta)

    exp_true = expected_true_utility(theta, psi, ref_logits, true_rewards)
    exp_bias = expected_bias_attr(theta, psi, ref_logits, bias_attr)

    pair_mass = _pair_bias_mass(theta, psi, ref_logits, b)
    signed_mass = _signed_pair_bias(theta, psi, ref_logits, b)

    # Oracle diagnostics (per cache -- independent of theta, except pi which is
    # passed for the policy-weighted aggregates).
    tr_np = true_rewards.cpu().numpy()
    ba_np = bias_attr.cpu().numpy()
    bl_np = base_logprob_per_token.cpu().numpy()
    p_star_np = p_star.cpu().numpy()
    p_obs_np = p_obs.cpu().numpy()
    b_np = b.cpu().numpy()
    odiag = oracle_diagnostics(
        p_star_np, p_obs_np, b_np, ba_np, tr_np, pi=probs.cpu().numpy(),
    )

    corr_bt = _corr_per_prompt(bl_np, tr_np)

    # Offdiag denominator diagnostics (per-prompt 1 - sum_k pi_k^2).
    off_stats = offdiag_denominator_stats(probs)

    # Directional policy-weighted diagnostics (median split per prompt).
    bad_mass, good_mass = _bad_good_masses(probs, bias_attr, true_rewards)

    # Wins vs reference.
    exp_win = expected_true_win_vs_ref(theta, psi, ref_logits, pi_ref, true_rewards)
    obs_win = _observed_win_vs_ref(theta, psi, ref_logits, pi_ref, p_obs)
    lab_H = _label_entropy(theta, psi, ref_logits, p_obs)
    h_sampled = _mean_abs_h_sampled(theta, psi, ref_logits, n_pairs=16)
    pol_arg = _greedy_pick(probs)
    ref_arg = _greedy_pick(pi_ref)
    bidx = torch.arange(probs.shape[0], device=probs.device)
    r_pol = true_rewards[bidx, pol_arg]
    r_ref = true_rewards[bidx, ref_arg]
    greedy_wins = (r_pol > r_ref).float() + 0.5 * (r_pol == r_ref).float()
    greedy_win_rate = greedy_wins.mean().item()
    greedy_util = r_pol.mean().item()
    greedy_bias = bias_attr[bidx, pol_arg].mean().item()

    # Training-objective reference values (no penalty) for the two modes.
    nom = exact_expectation_loss(theta, psi, ref_logits, p_star, 0.0, beta).item()
    bi = exact_expectation_loss(theta, psi, ref_logits, p_obs, 0.0, beta).item()

    # Paper decomposition-form robust objectives (derived scalars, no extra
    # forward passes). Under the margin condition, wcl_true_rho_mis equals
    # `worst_case_loss` to numerical precision.
    wcl_rho_mis = float(nom + rho_mis * beta * s_R)
    wcl_rho_alg = float(nom + rho_alg * beta * s_R)
    biased_fb_loss = float(bi)

    kl_ref = float(_kl_to_ref(theta, psi, ref_logits).mean().item())

    return EvalMetrics(
        sensitivity_R=s_R,
        worst_case_loss=wcl,
        expected_true_utility=exp_true,
        expected_bias_attr=exp_bias,
        entropy=entropy,
        top1_prob=top1,
        diagonal_mass=dm,
        theta_norm=theta_norm,
        candidate_base_true_corr=corr_bt,
        bias_attr_true_corr=odiag["bias_attr_true_corr"],
        bias_attr_true_corr_median=odiag["bias_attr_true_corr_median"],
        delta_empirical=odiag["delta_empirical"],
        min_p_star=odiag["min_p_star"],
        max_p_star=odiag["max_p_star"],
        max_abs_p_obs_minus_p_star=odiag["max_abs_p_obs_minus_p_star"],
        mean_abs_p_obs_minus_p_star=odiag["mean_abs_p_obs_minus_p_star"],
        max_abs_b=odiag["max_abs_b"],
        mean_abs_b=odiag["mean_abs_b"],
        bias_active_rate=odiag["bias_active_rate"],
        bias_conflict_rate=odiag["bias_conflict_rate"],
        bias_same_direction_zero_rate=odiag["bias_same_direction_zero_rate"],
        pair_bias_mass_under_pi=pair_mass,
        signed_pair_bias_under_pi=signed_mass,
        offdiag_denominator_min=off_stats["offdiag_denominator_min"],
        offdiag_denominator_mean=off_stats["offdiag_denominator_mean"],
        bad_bias_candidate_mass=bad_mass,
        good_candidate_mass=good_mass,
        expected_true_win_vs_ref=exp_win,
        observed_win_vs_ref=obs_win,
        label_entropy=lab_H,
        mean_abs_h_sampled=h_sampled,
        greedy_true_win_vs_ref=greedy_win_rate,
        greedy_true_utility=greedy_util,
        greedy_bias_attr=greedy_bias,
        nominal_bt_loss=nom,
        biased_bt_loss=bi,
        wcl_true_rho_mis=wcl_rho_mis,
        wcl_true_rho_alg=wcl_rho_alg,
        biased_feedback_loss=biased_fb_loss,
        KL_to_ref=kl_ref,
    )
