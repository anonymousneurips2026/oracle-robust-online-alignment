"""Properties of the directional pairwise bias b_ij from oracle_misspec.

Definition:
    ds_ij = s_i - s_j
    dr_ij = r_i - r_j
    M_x   = max_{u,v} |s_u - s_v| + eps
    b_ij  = sign(ds_ij) * 1[ds_ij * dr_ij <= 0] * |ds_ij| / M_x

Required properties (verified here):
    (i)   b_ii = 0 ; b_ji = -b_ij ; |b_ij| <= 1
    (ii)  b_ij = 0 whenever (s_i - s_j)(r_i - r_j) > 0 (same direction)
    (iii) neutral cases:
          ds_ij = 0        -> b_ij = 0  (because sign(0) = 0)
          dr_ij = 0, ds !=0 -> b_ij = ds_ij / M_x  (activates; neutral is <= 0)
    (iv)  nonzero b_ij only for
          (ds*dr < 0)  OR  (ds*dr == 0  AND  ds != 0)
"""
import numpy as np

from finite_response_sail.oracle_misspec import compute_pairwise_bias


def _sym(a):
    return a + np.swapaxes(a, -1, -2)


def test_antisymmetry_and_bounds():
    rng = np.random.default_rng(0)
    r = rng.uniform(0.0, 1.0, size=(4, 12))
    s = rng.uniform(0.0, 1.0, size=(4, 12))
    b = compute_pairwise_bias(r, s)

    diag = np.diagonal(b, axis1=-2, axis2=-1)
    assert np.allclose(diag, 0.0, atol=1e-12)
    assert np.allclose(_sym(b), 0.0, atol=1e-10)
    assert np.abs(b).max() <= 1.0 + 1e-9


def test_zero_on_same_direction_pairs():
    rng = np.random.default_rng(1)
    r = rng.uniform(0.0, 1.0, size=(6, 10))
    s = rng.uniform(0.0, 1.0, size=(6, 10))
    b = compute_pairwise_bias(r, s)

    ds = s[..., :, None] - s[..., None, :]
    dr = r[..., :, None] - r[..., None, :]
    same_dir = (ds * dr) > 0.0

    assert np.all(b[same_dir] == 0.0), (
        "b_ij must be zero whenever (s_i - s_j) and (r_i - r_j) have the same sign"
    )


def test_neutral_cases_ds_zero_and_dr_zero():
    # Construct two candidates with identical s (ds=0, dr!=0) and two with
    # identical r (dr=0, ds!=0). Both must meet the spec.
    # Prompt 0: s = [0.5, 0.5, 0.1, 0.9], r = [0.1, 0.9, 0.2, 0.3]
    #   -> pair (0,1): ds=0, dr=-0.8 -> b = 0 (sign(0)=0)
    # Prompt 1: s = [0.2, 0.8, 0.5, 0.1], r = [0.5, 0.5, 0.1, 0.9]
    #   -> pair (0,1): dr=0, ds=-0.6 -> b = ds / M_x   (activates; nonzero)
    s = np.array([
        [0.5, 0.5, 0.1, 0.9],
        [0.2, 0.8, 0.5, 0.1],
    ], dtype=np.float64)
    r = np.array([
        [0.1, 0.9, 0.2, 0.3],
        [0.5, 0.5, 0.1, 0.9],
    ], dtype=np.float64)
    b = compute_pairwise_bias(r, s)

    # Prompt 0, pair (0, 1): ds=0, must zero out.
    assert b[0, 0, 1] == 0.0
    assert b[0, 1, 0] == 0.0

    # Prompt 1, pair (0, 1): dr=0, ds=-0.6 < 0 -> b = -0.6 / M_x (nonzero).
    ds01_p1 = s[1, 0] - s[1, 1]  # -0.6
    abs_ds_p1 = np.abs(s[1, :, None] - s[1, None, :])
    M_x_p1 = abs_ds_p1.max() + 1e-8
    expected_b01_p1 = np.sign(ds01_p1) * abs(ds01_p1) / M_x_p1  # negative, nonzero
    assert np.isclose(b[1, 0, 1], expected_b01_p1, atol=1e-9)
    assert b[1, 0, 1] != 0.0


def test_nonzero_only_when_conflict_or_dr_neutral():
    rng = np.random.default_rng(2)
    r = rng.uniform(0.0, 1.0, size=(8, 12))
    s = rng.uniform(0.0, 1.0, size=(8, 12))
    b = compute_pairwise_bias(r, s)

    ds = s[..., :, None] - s[..., None, :]
    dr = r[..., :, None] - r[..., None, :]
    nonzero = np.abs(b) > 0
    conflict = (ds * dr) < 0.0
    dr_zero_ds_nonzero = (dr == 0.0) & (ds != 0.0)

    allowed = conflict | dr_zero_ds_nonzero
    assert np.all(~nonzero | allowed), (
        "b_ij is nonzero only when ds*dr < 0 (conflict) or dr==0 with ds!=0 (neutral activation)"
    )
