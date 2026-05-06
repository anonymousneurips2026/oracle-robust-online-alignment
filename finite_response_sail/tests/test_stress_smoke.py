"""Smoke test: invoke the evaluator CLI on one prior run, two alphas.

Asserts exit 0, that stress_alpha_metrics.csv exists, and that one PNG was emitted.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_smoke_invocation(tmp_path: Path):
    candidate_dirs = [
        REPO_ROOT / "finite_response_sail" / "results" / "v3_pairbias_confirm",
        REPO_ROOT / "finite_response_sail" / "results" / "v3_pairbias_rhoalg_sweep",
    ]
    if not any(c.exists() for c in candidate_dirs):
        pytest.skip("no prior run results available")

    out_dir = tmp_path / "ttw_smoke"
    cmd = [
        sys.executable, "-m", "finite_response_sail._true_to_worst_stress_eval",
        "--out-dir", str(out_dir),
        "--alpha-grid", "0,1",
        "--rho-mis", "0.075",
        "--smoke",
    ]
    for c in candidate_dirs:
        if c.exists():
            cmd += ["--include-results-dir", str(c)]

    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    assert (out_dir / "stress_alpha_metrics.csv").exists()
    pngs = list(out_dir.glob("*.png"))
    assert pngs, "no PNG emitted"
