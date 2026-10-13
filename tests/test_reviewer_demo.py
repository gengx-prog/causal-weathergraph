"""Behavioral checks for the small reviewer entry point."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_demo import run_demo
from revision.inference import fit_edge


def test_demo_fits_saved_data_deterministically_and_writes_verified_receipt(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    a, b = run_demo(first), run_demo(second)
    assert a["status"] == b["status"] == "passed"
    assert a["metrics"] == b["metrics"]
    assert all(item["passed"] for item in a["checks"].values())
    assert a["checks"]["independent_ols_coefficient"]["absolute_difference"] < 1e-10
    assert a["checks"]["independent_calendar_hac"]["absolute_difference"] < 1e-10
    for name in ("synthetic_data.csv", "truth.csv", "fitted_tests.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert len(a["source_sha256"]) == 3
    assert a["artifact_sha256"] == b["artifact_sha256"]
    assert json.loads((first / "summary.json").read_text())["status"] == "passed"


def test_existing_output_is_refused_without_modification(tmp_path):
    output = tmp_path / "evidence"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text('{"preserve": true}', encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_demo(output)
    assert sentinel.read_text(encoding="utf-8") == '{"preserve": true}'
    assert list(output.iterdir()) == [sentinel]


def test_cli_works_outside_repository_and_refuses_existing_output(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/run_demo.py"), "--output", "demo"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    receipt = tmp_path / "demo/summary.json"
    original = receipt.read_bytes()
    repeated = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr
    assert receipt.read_bytes() == original


def test_conditioning_distinguishes_direct_from_mediated_prediction():
    # A fixed, transparent chain X(t-2) -> M(t-1) -> Y(t). The X-Y
    # predictive relationship disappears after controlling its mediator.
    rng = np.random.default_rng(99)
    data = rng.normal(size=(3000, 3))
    data[1:, 1] += 0.85 * data[:-1, 0]
    data[1:, 2] += 0.85 * data[:-1, 1]
    y = data[2:, 2]
    target_history = data[1:-1, 2, None]
    own = fit_edge(y, target_history, data[:-2, 0], bandwidths=(64,))
    conditioned = fit_edge(y, np.column_stack([target_history, data[1:-1, 1]]),
                           data[:-2, 0], bandwidths=(64,))
    direct = fit_edge(y, np.column_stack([target_history, data[:-2, 0]]),
                      data[1:-1, 1], bandwidths=(64,))
    assert own["p_hac64"] < 1e-10
    assert conditioned["p_hac64"] > 0.05
    assert direct["p_hac64"] < 1e-10
    assert abs(conditioned["effect"]) < 0.1
    assert direct["effect"] > 0.7


def test_scientific_failure_is_recorded_and_not_reported_as_success(tmp_path, monkeypatch):
    import revision.simulations

    def fail(*args, **kwargs):
        raise RuntimeError("injected fitting failure")

    monkeypatch.setattr(revision.simulations, "regression_pvalues", fail)
    with pytest.raises(RuntimeError, match="injected fitting failure"):
        run_demo(tmp_path / "failed")
    report = json.loads((tmp_path / "failed/summary.json").read_text())
    assert report["status"] == "failed"
    assert "injected fitting failure" in report["error"]
