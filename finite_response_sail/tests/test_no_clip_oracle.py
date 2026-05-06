"""Guardrail: oracle_misspec.compute_p_star must not clip p_star.

We parse the module AST and walk compute_p_star specifically; the oracle as a
whole can still use clamp-like helpers elsewhere (e.g. numeric stability in
_stable_sigmoid). Test only rejects clip/clamp calls *inside* compute_p_star
and rejects deprecated identifiers (p_star_tilde, CAL_LO, CAL_HI, RHO_MAX) at
any scope.

We avoid string-grepping for the bare word "clip" -- legitimate references
appear in docstrings ("no probability clipping is used").
"""
import ast
import math
from pathlib import Path

import numpy as np

import finite_response_sail.oracle_misspec as om


FORBIDDEN_CLIP_CALLS = {"clip", "clamp", "clamp_"}
FORBIDDEN_IDENTIFIERS = {"p_star_tilde", "CAL_LO", "CAL_HI", "RHO_MAX"}


def _load_module_ast() -> ast.Module:
    path = Path(om.__file__)
    return ast.parse(path.read_text())


def _find_function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in {om.__file__}")


def test_compute_p_star_has_no_clip_call():
    tree = _load_module_ast()
    fn = _find_function_def(tree, "compute_p_star")
    offenders = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = node.func
            fn_name = None
            if isinstance(func, ast.Attribute):
                fn_name = func.attr
            elif isinstance(func, ast.Name):
                fn_name = func.id
            if fn_name in FORBIDDEN_CLIP_CALLS:
                offenders.append(fn_name)
    assert not offenders, f"compute_p_star calls forbidden clip-like ops: {offenders}"


def test_compute_p_star_has_no_deprecated_name_refs():
    tree = _load_module_ast()
    fn = _find_function_def(tree, "compute_p_star")
    offenders = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_IDENTIFIERS:
            offenders.append(node.id)
    assert not offenders, f"compute_p_star references deprecated names: {offenders}"


def test_module_has_no_deprecated_names_at_top_level():
    for name in list(FORBIDDEN_IDENTIFIERS) + ["p_star_tilde_matrix", "p_obs_matrix",
                                               "clip_stats", "effective_rho_coeff"]:
        assert not hasattr(om, name), (
            f"finite_response_sail.oracle_misspec exposes deprecated name {name!r}"
        )


def test_compute_p_star_matches_sigmoid_formula():
    rng = np.random.default_rng(42)
    r = rng.uniform(0.0, 1.0, size=(5, 10))
    tau = 2.0
    got = om.compute_p_star(r, tau=tau)
    diff = r[..., :, None] - r[..., None, :]
    expected = 1.0 / (1.0 + np.exp(-tau * diff))
    assert np.allclose(got, expected, atol=1e-12), (
        f"compute_p_star deviates from sigmoid(tau*(r_i - r_j)); "
        f"max abs diff = {np.abs(got - expected).max()}"
    )
