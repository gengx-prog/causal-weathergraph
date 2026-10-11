# WHEC temporal-control correction

The October 9 correction changes the unavailable first year of the 365-day
humidity-shift control. Humidity for 1979 now wraps within discovery to 2018,
instead of wrapping across the full record to 2025. From 1980 onward the control
uses the actual value 365 days earlier. Its climatology and scale remain fitted
on discovery data. This is a retrospective correction with a nonchronological
initialization, not an operational forecast input or prospective validation.

The original design and its SHA-256
`0507157c5a8ec0b248a36bca29818af6623ba86b1e19576f57fd743d92beee92`
remain unchanged. The
[`experiment-results.zip` archive](https://github.com/gengx-prog/causal-weathergraph/releases/download/revision-20261010/experiment-results.zip)
records the amendment and corrected
results under `whec/`, alongside `correction_verification.json` and the
correction source. The portable entry points
below adapt that correction while retaining the repository's configurable
input paths and cloud-input fingerprints.

## Separate entry points and inputs

`revision/run_whec_test.py`, `revision/make_whec_index.py` and the October 8
`scripts/reproduce.py --experiment whec` workflow retain their historical
semantics. That archived reproduction does **not** verify the amended control.
Use `revision/run_whec_test_20261009.py` for the corrected analysis. It rejects
an index lacking the corrected humidity-source indices, any held-out source
in discovery, and any departure from the specified shift schedule.

Install the normal scientific dependencies from `requirements-test.txt`.
The October 8 processed-input release supplies `inputs/`, the original
`whec_test/design.json`, and the ERA5/CERES cloud arrays described in the
[reproduction guide](REPRODUCING.md). Its historical `whec_index/` must not be
substituted for the corrected index.

For the fast route, download
[`13_WHEC_amendment_inputs.zip`](https://github.com/gengx-prog/causal-weathergraph/releases/download/revision-20261010/13_WHEC_amendment_inputs.zip)
from the separate
[`revision-20261010` release](https://github.com/gengx-prog/causal-weathergraph/releases/tag/revision-20261010)
and extract it into `artifacts/data`. It adds
`whec_index_corrected_20261009/` and leaves the October 8 files unchanged.
The runner needs `whec_regional.npz` and its sibling `correction_amendment.json`.
The index is 210,078,225 bytes with SHA-256
`410024b59dd38f5cf0414e8a0496844e4b52d16c581c6dd7fa944f5cc0de24bb`.
The amendment SHA-256 is
`cb4c0c0ae192ef4861ecd5b0186cb6c283c0d60f6f32fe9fe1c07da908efcd8f`.
The compact `experiment-results.zip` archive contains results and source code,
not this larger index or the upstream grids.

Alternatively, rebuild the corrected index from the base pipeline's native/conservatively
regridded six-hourly `u`, `v` and humidity inputs, run from the repository root:

```sh
python -m revision.make_whec_index_20261009 \
  --data-root /path/to/native/data_root \
  --inputs artifacts/data/inputs \
  --output artifacts/data/whec_index_corrected_20261009
```

The grid folder must follow `revision.prepare_inputs.SEGMENTS` and contain the
three filenames defined in `revision.make_whec_index_20261009.FILES`. The input
geometry must match `trainfit_parameters.npz`. This step requires the large
upstream meteorological inputs; downloading the compact evidence ZIP alone is
insufficient. The index builder writes the correction amendment beside the index.

With that corrected index and the archived processed inputs in place:

```sh
python -m revision.run_whec_test_20261009 \
  --artifact-root artifacts/data \
  --whec-index artifacts/data/whec_index_corrected_20261009 \
  --design artifacts/data/whec_test/design.json \
  --era5-cloud-npz artifacts/data/ceres_cloud_substitution/era5_cloud_6h_64x32_2017_2025.npz \
  --output outputs/my-corrected-whec
```

Choose a new output directory. Alternatively, `--data-root` and
`--extension-dir` select the original NetCDF cloud inputs instead of the
portable cloud NPZ. The new runner records all input hashes, the correction
hash and the design hash, and saves the 108 endpoint time series for audit.

## Checks and observed validation

```sh
python -m pytest tests/test_whec_placebo.py tests/test_whec_corrected_runner.py -q
python -m revision.verify_whec_20261009 \
  --actual outputs/my-corrected-whec \
  --reference-archive /path/to/experiment-results.zip \
  --output outputs/my-corrected-whec/verification.json
```

The verifier compares every cell of `whec_tests_per_lag.csv`,
`whec_block_tests.csv`, `whec_band_endpoints.csv` and `whec_ceres_check.csv`
under `whec/`, preserving categorical, integer and
Boolean values exactly and using `rtol=1e-7, atol=1e-10` for floating-point
values. It also checks the archived verification decisions, source isolation and selected direct
fit results, and independently recomputes all 108 calendar-HAC endpoints from
the saved time series. Optional `--reference-dir` compares all six CSVs and
every endpoint-series value with a complete corrected output archive.

The [current numerical receipt](../validation/whec-correction-experiments-20261010.json)
records the corrected experiment's inputs, execution and comparison outcomes.
It checks the six result tables and all 108 endpoint series using the October 8
downloadable processed inputs, the portable ERA5 cloud NPZ and the separate
corrected index. Inspect that receipt for measured runtime and configuration;
they are observations, not promised performance on another machine.
Reusing the archived corrected index does not repeat its raw-grid construction.
This comparison also does not rerun the 400 full-dimension calibration
replicates or provide independent future outcomes.
