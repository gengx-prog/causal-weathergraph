# Running experiments on GitHub

This guide separates retained numerical receipts from a new execution. The
historical checks below were recorded in October 2026. They do not automatically
validate the current checkout or every subsequent experiment. Inspect a new
Actions run before claiming that the current version passed.

The recorded workflows and configurations are reference examples, not mandatory
or exclusive conditions for reproduction. A passing run does not promise that
all devices or environments will execute successfully. See the
[reference and compatibility notice](COMPATIBILITY.md#reference-status-and-compatibility)
for the scope of these materials and possible external causes of failure.

Opening a Python file on GitHub does not run it or host the optional estimator.
Use the [local reproduction commands](REPRODUCING.md) or the configured workflows.

## Retained numerical evidence

| Task | Recorded result | Receipt and scope |
| --- | --- | --- |
| CPU unit/numerical checks | Windows and Ubuntu: 173 passed, 13 model-dependent checks skipped | [Raw-data validation receipt](../validation/raw-data-local.json). The skipped checks require separate trained models. |
| Small synthetic simulation | One repetition per scenario; two null scenarios and five graph scenarios | [Clean installation receipt](../environment/validation-clean-20261008.json). This is a command/output smoke test, not full calibration. |
| Raw-data subset extraction | 85 files verified in the local restore test | [Local raw-data receipt](../validation/raw-data-local.json). It covers the legacy CDS extension, January 2019 CERES pilot and source metadata, not the whole raw archive. |
| Core + two CERES settings + WHEC from processed inputs | All 24 tables and the WHEC design/period/decision checks passed on the recorded Ubuntu run | [Per-table receipt](../validation/github-experiments-all-20261008.json). Four analysis jobs took about 342 seconds, excluding setup/download. |
| Corrected WHEC temporal control | Checks compare six tables and all 108 endpoint series with archived corrected outputs within specified tolerances | [Current correction receipt](../validation/whec-correction-experiments-20261010.json) and [correction guide](WHEC_AMENDMENT_20261009.md). Separate from the historical WHEC workflow. |

The 24-table workflow starts from archived processed arrays. It does not repeat
all raw-data acquisition, remapping, preprocessing, index construction, large
calibration runs or independent model training. Software checks do not establish
atmospheric causal identification.

## Run a supported workflow

1. Open **Actions** in this repository or a current fork with Actions enabled.
2. Select the experiment-reproduction workflow, choose **Run workflow** on
   **main**, and select `core` or `all`. Here `all` means the core graph analysis,
   two CERES climatology settings and historical WHEC analysis from processed
   inputs. It does not mean every experiment or all raw preprocessing.
3. Use the raw-data checksum workflow for supported download checks, or the
   reproducibility workflow for CPU tests and a small synthetic simulation.
4. Inspect every job's status and download its artifacts. Check the numerical
   comparison report, logs, CSVs and recorded runner environment. The report
   must show `status: passed`, with all planned jobs and CSV comparisons complete;
   WHEC also requires its design/period/decision checks to pass.

The analysis and unit workflows install the pinned CPU dependencies. The raw
checksum workflow uses the Python standard library. Archived public downloads
do not require CDS/NASA credentials. Input checksums must match.

Manual workflow dispatch requires repository write access. Without it, use an
existing result, run locally, or fork the repository and enable Actions there.
This permission requirement is separate from whether the experiment code works.

## Resource and compatibility limits

| Situation | Check and remedy |
| --- | --- |
| Complete raw download/extraction on a small runner | The archive is about 108.69 GB. Keeping ZIPs and extracted files needs more than 220 GB plus processing space. Check the actual runner's available storage before starting. Download selected groups or use appropriately sized local/hosted resources. |
| Expecting every upstream stage to run automatically | Download/hash verification does not execute conservative remapping, fitted preprocessing or WHEC construction. Follow [RAW_DATA.md](RAW_DATA.md), supply explicit paths and compare generated inputs with archived references. A complete portable raw-to-results rerun has not been verified. |
| Missing optional cloud estimator | A clean checkout does not include its trained bundle in `../revision_outputs/cloud_simulator_v1/`. This is a CPU scikit-learn prototype; its absence is not a GPU problem. See the recovery route below. |
| Historical path not found | Some helpers expect a previous workspace layout or sibling data directory. Prefer the portable runner's input/output options. Adapt other helpers in a separate copy and verify their outputs; never overwrite archived results. |
| Interrupted transfer or checksum mismatch | Follow the downloader's reported error. Resume an interrupted transfer where supported. A completed file with a wrong hash is not accepted as valid scientific input. |
| Failure on a different machine | Check the command, inputs, dependency versions, memory, disk and error details. These are candidate causes, not proof that the computer is at fault. Reproducible failures may be project defects; report them through the issue tracker. |

Use the [software guide](SOFTWARE.md) and [compatibility guidance](COMPATIBILITY.md)
for installation and diagnosis. The graph-analysis checks run on CPU and do not
require a GPU.

## Optional cloud-estimator recovery

The bundle needs `manifest.json`, `models.joblib` and `metadata.json`; integration
checks also need `split_indices.npz`, `predictions.npz` and `protocol.json`.
The two training/source-fingerprint inputs are in `extended-inputs.zip`.

For a new checkout with an absent or empty sibling
`../revision_outputs/cloud_simulator_v1` directory, the scripts expose this route.
It has not been executed by the supported analysis workflow; the 13 optional
checks remain unverified until the required trained outputs exist and pass.

```bash
python -m pip install -r requirements-test.txt
python scripts/download_artifacts.py --asset extended-inputs.zip --output ../revision_outputs
python scripts/verify_artifacts.py --asset extended-inputs.zip --data-root ../revision_outputs
python revision/cloud_simulator/engine.py --output ../revision_outputs/cloud_simulator_v1 --horizons 0
python -m pytest tests/test_cloud_simulator.py -q -rs
python revision/cloud_simulator/app.py --artifacts ../revision_outputs/cloud_simulator_v1 --port 8765
```

Preserve existing trained artifacts. These tests use the fixed
`cloud_simulator_v1` path and horizon-0 protocol. Other horizons require separate
protocols/checks, and changing the training directory does not change that lookup.
New training is not guaranteed to reproduce historical model bytes or metrics.
The app runs locally; it is not deployed by GitHub Actions.

The [weather-token experiment](../revision/weather_tokens/) needs PyTorch,
its own inputs and checkpoint validation, and supports CPU or CUDA. The
[intervention benchmark](../revision/intervention_benchmark/) also has a separate
design and execution scope. Neither is covered by the core/CERES/WHEC workflow.
