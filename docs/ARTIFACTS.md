# Research artifacts and their scope

The inventory below describes the October 8 processed-input release.
The later temporal-control correction is documented in the
[WHEC correction guide](WHEC_AMENDMENT_20261009.md).

This page describes the processed-input release. The subsequently published
[raw-data archive](RAW_DATA.md) adds the 1,100 historical ERA5/WB2 and CERES
scientific source/input files and their [official source links](RAW_DATA_SOURCES.md).

The release tag is `reproducibility-20261008.1`. The
[release manifest](../artifacts/release-manifest.json) lists every archive and
every contained file with its byte count and SHA-256. ZIP entry paths are
relative to the research `revision_outputs` directory. Extract the selected
archives into one data directory, for example `artifacts/data/`; the resulting
directories include `inputs/`, `three_stage/`, and `graph_nulls/` directly.
There is no extra enclosing `revision_outputs/` directory inside the ZIPs.

| Archive | Contents | Use |
| --- | --- | --- |
| `core-inputs.zip` | 16 files: the full 1979–2025 regional input, discovery-fitted preprocessing parameters and source manifest, symmetric candidate table, original screening reference, and all three-stage reference results | Refit the screen–confirm–replicate analysis on the archived reanalysis inputs |
| `extended-inputs.zip` | 21 files: vector and physical inputs, full-record exploratory and diurnal regional arrays, WHEC arrays and parameters, prepared CERES cloud, the traceable ERA5 cloud reference, and six-field physical-control inputs | Refit CERES substitution and WHEC using the portable runners; supply inputs for the corresponding sensitivity analyses |
| `experiment-evidence.zip` | Selected machine-readable results, designs, manifests, independent numerical checks, small simulation arrays, and executing source snapshots | Inspect the evidence supporting calibration, null graphs, held-out comparisons, regime/latitude/hemisphere contrasts, paths, source-route checks, RPCMCI, and CaStLe |

Download URLs and exact archive sizes are authoritative in the manifest. The
core, extended, and evidence ZIPs are respectively 144,583,135, 1,578,896,832,
and 73,129,251 bytes. Each archive is below 2,000,000,000 bytes. Numerical data retain their recorded file bytes;
files shared by multiple analyses appear in only one asset. The extended and
evidence assets supplement the core asset. Keep reference results separate from
new experiment output directories.

The archive cleanup updated metadata and portable source defaults; scientific
CSV and array files retain their original bytes. Use the updated release manifest for ZIP and member
SHA-256 values rather than an earlier downloaded manifest.

The `revision-20261010` data release adds separate experimental assets:

| Archive | Scope |
| --- | --- |
| `experiment-results.zip` | Additional recorded experimental results, configurations, analysis source and numerical checks, including corrected WHEC references under `whec/`. |
| `experiment-figures.zip` | Experimental visualization data, renderers and figure checks. |
| `13_WHEC_amendment_inputs.zip` | Corrected regional WHEC index and its amendment provenance; separate from the smaller result archive. |

The [WHEC correction guide](WHEC_AMENDMENT_20261009.md) specifies how to combine
the corrected index with the earlier processed inputs. The old and corrected
WHEC controls are distinct experiments; their inputs must not be interchanged.

## What the data represent

`inputs/region_trainfit.npz` contains a float64 array of shape
`(68668, 66, 4)`, ordered as temperature, humidity, wind speed, and cloud cover.
It preserves every six-hour timestamp from 1979-01-01 through 2025-12-31.
Climatology and standardization were fitted on 1979–2018. The input's SHA-256 is
`d7acba521f315f0ab973e8fcfa71bf8f67a10130b30eb6871666b33704f3b841`, matching
the recorded three-stage and WHEC manifests. The 2023–2025 segment uses the
native-CDS conservative regridding route recorded in the input manifest.

`inputs/region_series_fullrecord.npz` uses full-record preprocessing and is
included to audit the original exploratory screening. It must not replace the
discovery-fitted input in temporal replication. Use the manifest file identifiers
and hashes to select inputs, and supply your own input/output directories.

`ceres_cloud_substitution/ceres_cloud_6h_64x32.npz` is the prepared CERES
reference on the common ERA5 grid. The first timestamp lacks the required
preceding CERES hour box; the substitution runner checks and excludes this
documented missing timestamp.

`ceres_cloud_substitution/era5_cloud_6h_64x32_2017_2025.npz` is a portable
reference derived from the original ERA5 cloud loader. Its `cloud_percent`
array is float64 with shape `(13148, 2048)` and its `timestamps` array is
`datetime64[ns]`. The grid is flattened in ascending latitude, then longitude.
Values are in percentage points, range from 0 to 100, and were saved without
rounding, clipping, interpolation, or new fitting. The saved arrays were checked
with `numpy.array_equal` against the original loader outputs.

The accompanying `era5_cloud_reference_manifest.json` records the three source
NetCDF hashes, byte counts, relative research paths, selected periods, exact
transformations, and resulting NPZ hash. The first two segments come from the
recorded WeatherBench 2 inputs; the last comes from the native-conservative
2023–2025 extension. This reference is a deterministic extraction of existing
data, not newly acquired meteorological observations.

## Reproduction boundaries

The release supplies the processed inputs needed to rerun the core three-stage,
CERES-substitution, and WHEC analyses without downloading the original global
archives. Follow the repository's reproduction guide for the portable commands,
including explicit data and output paths. WHEC requires the unchanged frozen
`design.json` from `whec_test/` to be copied into a new output directory first.

The evidence archive includes the finalized aligned CaStLe comparison rather
than the superseded response-window run. Its executing third-party source and
MIT license are retained together. Post-diagnostic calibration amendments and
graph-mixing limitations remain recorded in the original output metadata;
publishing these files does not reclassify exploratory analyses as preregistered
confirmation. Independent audit scripts and older sensitivity runners may
require explicit input/output path configuration to rerun.

These three processed-input assets exclude raw meteorological fields, download
sessions, provider credentials, trained prototype models, logs, caches, and some
historical analysis directories. Retained original ERA5/CERES acquisitions,
including native-resolution fields, are now supplied in the separate
[raw-data release](RAW_DATA.md). Reproducing conservative remapping, the WHEC
index from raw grids, and physical-control preparation still requires those
source files and the appropriate historical processing steps. Small processed
inputs and output checks do not execute those stages. See
[GitHub execution limits and remedies](GITHUB_EXECUTION.md) before attempting
the complete raw workflow.

The intervention benchmark, weather-token forecasting pilot, transport
consistency prototype, and cloud simulator are separate exploratory projects.
Their output archives are outside this release. Older result-summary builders
that read prototype folders cannot all be rerun from these three assets alone.

## Building the release archives locally

Maintainers can rebuild the same explicit selection with the standard-library
utility below. The source root is the recorded `revision_outputs` directory.
The supplemental root contains only the newly staged ERA5 cloud NPZ and its
provenance manifest, using the same relative paths described above.

```text
python scripts/build_release_assets.py --source-root /path/to/recorded/revision_outputs --supplemental-root /path/to/staged/revision_outputs --output-dir /path/to/new/assets --manifest /path/to/new/release-manifest.json
```

Add `--check-only` to inspect file counts and selected byte totals without
writing. Existing archive and manifest paths are refused. The builder checks
that paths stay inside the declared roots, rejects recognizable embedded
credentials, preserves archived bytes, and then reads every ZIP member back to
verify its CRC and SHA-256. The generated manifest is the input used by the
download and verification tools. This utility performs no upload.

The prepared release contains 459 unique file paths. Archive integrity checks
verify their recorded byte counts and SHA-256 digests.
