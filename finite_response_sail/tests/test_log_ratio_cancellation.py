"""h_ij via log-ratio form equals theta^T (psi_i - psi_j) under the log-linear policy.

Under pi_theta(y_i|x) proportional to pi_ref(y_i|x) * exp(theta^T psi_i) we have
    log pi_theta_i - log pi_ref_i = theta^T psi_i - log Z
so log-partitions cancel in the pairwise difference.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from finite_response_sail.sail_score import h_matrix_from_log_ratio, h_matrix_from_theta


@pytest.mark.parametrize("ref_mode", ["uniform", "base_lm_lognorm"])
def test_log_ratio_equals_theta_difference(ref_mode):
    torch.manual_seed(0)
    B, K, d = 5, 8, 16
    theta = torch.randn(d)
    psi = torch.randn(B, K, d) * 0.3
    if ref_mode == "uniform":
        ref_logits = torch.zeros(B, K)
    else:
        ref_logits = torch.randn(B, K) * 0.7

    h_theta = h_matrix_from_theta(theta, psi)
    h_ratio = h_matrix_from_log_ratio(theta, psi, ref_logits)
    max_abs_err = (h_theta - h_ratio).abs().max().item()
    assert max_abs_err < 1e-5, f"{ref_mode}: max|diff|={max_abs_err}"


def test_cancellation_holds_for_degenerate_zero_theta():
    theta = torch.zeros(10)
    psi = torch.randn(3, 4, 10)
    ref = torch.randn(3, 4) * 2.0
    h_theta = h_matrix_from_theta(theta, psi)
    h_ratio = h_matrix_from_log_ratio(theta, psi, ref)
    assert h_theta.abs().max().item() == 0.0
    assert h_ratio.abs().max().item() < 1e-6
