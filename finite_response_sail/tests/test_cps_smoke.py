"""End-to-end smoke test for the CPS evaluator CLI."""
import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest


CONFIRM_DIR = Path("finite_response_sail/results/v3_pairbias_confirm")


@pytest.mark.skipif(not CONFIRM_DIR.exists(),
                    reason="v3_pairbias_confirm results not present")
def test_cps_smoke(tmp_path: Path):
    out = tmp_path / "cps_smoke"
    rc = subprocess.call([
        sys.executable, "-m", "finite_response_sail._cps_shift_eval",
        "--tag", "smoke",
        "--results-dir", str(CONFIRM_DIR),
        "--out-dir", str(out),
        "--smoke",
    ])
    assert rc == 0
    csv_path = out / "cps_alpha_metrics.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) >= 1
    assert "L_alpha_exact" in rows[0] or "L_alpha_offdiag" in rows[0]
    # Smoke mode emits the loss curves PNG too.
    assert (out / "cps_loss_curves_by_rho.png").exists()
