"""Compare a corrected WHEC rerun with independent experimental reference data."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy import stats

TABLES = {
    "whec/whec_tests_per_lag.csv": "whec_tests_per_lag.csv",
    "whec/whec_block_tests.csv": "whec_block_tests.csv",
    "whec/whec_band_endpoints.csv": "whec_band_endpoints.csv",
    "whec/whec_ceres_check.csv": "whec_ceres_check.csv",
}
RTOL, ATOL = 1e-7, 1e-10


def compare(actual, reference):
    """Compare every cell; categorical, integer and Boolean values are exact."""
    assert_frame_equal(actual, reference, check_dtype=False, check_exact=False,
                       rtol=RTOL, atol=ATOL)
    largest = 0.0
    for col in reference:
        if pd.api.types.is_float_dtype(reference[col]):
            delta = np.abs(actual[col].to_numpy() - reference[col].to_numpy())
            if np.isfinite(delta).any():
                largest = max(largest, float(np.nanmax(delta)))
        else:
            assert_frame_equal(actual[[col]], reference[[col]],
                               check_dtype=False, check_exact=True)
    return {"rows": len(actual), "columns": len(actual.columns),
            "max_absolute_numeric_difference": largest, "passed": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actual", type=Path, required=True)
    ap.add_argument("--reference-archive", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path,
                    help="Optional complete corrected output directory for six CSVs and endpoint series.")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.actual / "manifest.json").read_text(encoding="utf-8"))
    report = {"scope": "Real-input corrected test rerun; index construction is not rerun.",
              "rtol": RTOL, "atol": ATOL, "reference_tables": {}}
    with zipfile.ZipFile(args.reference_archive) as archive:
        for entry, filename in TABLES.items():
            report["reference_tables"][entry] = compare(
                pd.read_csv(args.actual / filename), pd.read_csv(io.BytesIO(archive.read(entry))))
        historical = json.loads(archive.read("whec/correction_verification.json"))
        assert manifest["design_sha256"] == historical["design_sha256"]
        assert manifest["placebo_discovery_heldout_source_count"] == historical["heldout_sources_in_discovery"] == 0
        assert {k: v["verdict"] for k, v in manifest["decision"].items()} == historical["decisions"]
        assert manifest["fwl_vs_direct_max_abs_difference"] < 1e-8
        endpoint = pd.read_csv(args.actual / "whec_band_endpoints.csv")
        for row in historical["independent_endpoint_checks"]["selected_C_midlatitude"]:
            matched = endpoint[(endpoint.family == row["family"]) &
                               (endpoint.target_var == row["target"]) &
                               (endpoint.period == row["period"]) &
                               (endpoint.statistic == row["statistic"])].iloc[0]
            np.testing.assert_allclose([matched.estimate, matched.p_one_sided],
                                       [row["estimate"], row["p_one_sided"]], rtol=RTOL, atol=ATOL)
        blocks = pd.read_csv(args.actual / "whec_block_tests.csv")
        direct = historical["independent_selected_direct_fit"]
        for row in direct["frozen_gain_checks"]:
            matched = blocks[(blocks.family == direct["family"]) &
                             (blocks.target_var == direct["target_var"]) &
                             (blocks.target_region == direct["target_region"]) &
                             (blocks.period == "frozen_" + row["period"])].iloc[0]
            np.testing.assert_allclose([matched.relative_block_gain, matched.block_gain_p1],
                                       [row["relative_block_gain"], row["p_one_sided"]], rtol=RTOL, atol=ATOL)
        report["reference_decisions_isolation_and_selected_checks_passed"] = True

    # Independently reconstruct all band means and explicit Bartlett-HAC sums.
    with np.load(args.actual / "whec_band_endpoint_series.npz") as series:
        assert len(series.files) == len(endpoint) == 108
        for row in endpoint.itertuples():
            values = series[f"{row.family}__{row.target_var}__{row.period}__{row.statistic}"]
            mean = float(values.mean())
            centered = values - mean
            variance = centered @ centered
            for lag in range(1, 65):
                variance += 2 * (1 - lag / 65) * (centered[lag:] @ centered[:-lag])
            se = np.sqrt(variance * len(values) / (len(values) - 1)) / len(values)
            np.testing.assert_allclose([mean, se, stats.norm.sf(mean / se)],
                                       [row.estimate, row.se_hac64, row.p_one_sided],
                                       rtol=RTOL, atol=ATOL)
        report["independent_endpoint_recalculations"] = len(endpoint)
        if args.reference_dir:
            report["full_reference_tables"] = {}
            for path in sorted(args.reference_dir.glob("*.csv")):
                report["full_reference_tables"][path.name] = compare(
                    pd.read_csv(args.actual / path.name), pd.read_csv(path))
            with np.load(args.reference_dir / "whec_band_endpoint_series.npz") as expected:
                assert set(series.files) == set(expected.files)
                for key in series.files:
                    np.testing.assert_allclose(series[key], expected[key], rtol=RTOL, atol=ATOL)
            report["full_endpoint_series_equal"] = True
    report["run"] = {k: manifest[k] for k in ["code_sha256", "inputs", "corrected_whec_index_sha256",
                    "correction_amendment_sha256", "periods", "fwl_vs_direct_max_abs_difference",
                    "elapsed_seconds", "placebo_discovery_heldout_source_count"]}
    report["reference_archive_sha256"] = hashlib.sha256(args.reference_archive.read_bytes()).hexdigest()
    report["all_pass"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": True, "tables": len(TABLES), "endpoints": 108,
                      "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
