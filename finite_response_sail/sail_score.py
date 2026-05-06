"""Pairwise SAIL score h_ij in two forms (used by the cancellation test)."""
import torch
import torch.nn.functional as F


def h_matrix_from_theta(theta, psi):
    """h[b, i, j] = theta^T (psi_i - psi_j). Returns [B, K, K]."""
    score = torch.einsum("bkd,d->bk", psi, theta)
    return score.unsqueeze(-1) - score.unsqueeze(-2)


def h_matrix_from_log_ratio(theta, psi, ref_logits):
    """Compute h_ij via the log-ratio form:

        h_ij = (log pi_i - log pi_ref_i) - (log pi_j - log pi_ref_j)

    where pi = softmax(ref_logits + theta^T psi) and
    pi_ref = softmax(ref_logits).
    """
    score = torch.einsum("bkd,d->bk", psi, theta)
    log_pi = F.log_softmax(ref_logits + score, dim=-1)
    log_pi_ref = F.log_softmax(ref_logits, dim=-1)
    diff = log_pi - log_pi_ref
    return diff.unsqueeze(-1) - diff.unsqueeze(-2)
