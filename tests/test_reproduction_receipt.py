import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reproduce import compare_csv, compare_whec_metadata


def record():
    return {"design_sha256": "frozen", "periods": {"discovery": 58437},
            "decision": {"cloud_cover": {"role": "primary", "verdict": "not_supported",
                                        "E1": {"holds": False, "estimate": 0.125}}}}


def compare(tmp_path, changed):
    actual, reference = tmp_path / "actual.json", tmp_path / "reference.json"
    actual.write_text(json.dumps(changed), encoding="utf-8")
    reference.write_text(json.dumps(record()), encoding="utf-8")
    return compare_whec_metadata(actual, reference)


@pytest.mark.parametrize("change", ["verdict", "holds", "design", "period", "missing", "numeric"])
def test_whec_scientific_decisions_cannot_change(tmp_path, change):
    value = record()
    if change == "verdict":
        value["decision"]["cloud_cover"]["verdict"] = "supported"
    elif change == "holds":
        value["decision"]["cloud_cover"]["E1"]["holds"] = 0
    elif change == "design":
        value["design_sha256"] = "different"
    elif change == "period":
        value["periods"]["discovery"] += 1
    elif change == "missing":
        del value["decision"]["cloud_cover"]["E1"]["holds"]
    else:
        value["decision"]["cloud_cover"]["E1"]["estimate"] = 0.15
    with pytest.raises(AssertionError):
        compare(tmp_path, value)


def test_whec_accepts_small_blas_roundoff_only(tmp_path):
    value = record()
    value["decision"]["cloud_cover"]["E1"]["estimate"] += 1e-12
    result = compare(tmp_path, value)
    assert result["status"] == "passed"
    assert result["verdicts"] == {"cloud_cover": "not_supported"}


def test_counts_are_exact_even_when_relative_tolerance_would_allow_change(tmp_path):
    a, b = tmp_path / "actual.csv", tmp_path / "reference.csv"
    a.write_text("count,effect\n1000000001,0.5\n", encoding="utf-8")
    b.write_text("count,effect\n1000000000,0.5\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        compare_csv(a, b)


def test_matching_counts_allow_small_floating_roundoff(tmp_path):
    a, b = tmp_path / "actual.csv", tmp_path / "reference.csv"
    a.write_text("count,effect,confirmed\n1000000000,0.500000000001,True\n", encoding="utf-8")
    b.write_text("count,effect,confirmed\n1000000000,0.5,True\n", encoding="utf-8")
    assert compare_csv(a, b)["status"] == "passed"
