"""Skip-if-exists predicate test for the rhoalg/kl wrapper scripts.

Builds two synthetic run directories — one complete, one half-finished —
then invokes the shell helper `is_run_complete` and asserts the right
decisions.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest


HELPER = Path(__file__).resolve().parent.parent.parent / "scripts" / "_run_helpers.sh"


@pytest.mark.skipif(not HELPER.exists(), reason="helper script not present")
def test_skip_if_exists(tmp_path: Path):
    complete = tmp_path / "complete"
    incomplete = tmp_path / "incomplete"
    complete.mkdir()
    incomplete.mkdir()

    (complete / "metrics.json").write_text(json.dumps({
        "final": {"test": {"expected_true_utility": 0.123}},
    }))
    (complete / "theta.pt").write_bytes(b"\x00" * 32)
    (complete / "oracle_tensors_test.npz").write_bytes(b"\x00" * 32)

    # incomplete is missing theta.pt
    (incomplete / "metrics.json").write_text(json.dumps({
        "final": {"test": {"expected_true_utility": 0.456}},
    }))
    (incomplete / "oracle_tensors_test.npz").write_bytes(b"\x00")

    rc_complete = subprocess.run(
        ["bash", "-c",
         f"source {HELPER}; if is_run_complete '{complete}'; then exit 0; else exit 1; fi"],
        capture_output=True,
    ).returncode
    rc_incomplete = subprocess.run(
        ["bash", "-c",
         f"source {HELPER}; if is_run_complete '{incomplete}'; then exit 0; else exit 1; fi"],
        capture_output=True,
    ).returncode
    assert rc_complete == 0, "complete run should be flagged as complete"
    assert rc_incomplete == 1, "incomplete run should NOT be flagged as complete"
