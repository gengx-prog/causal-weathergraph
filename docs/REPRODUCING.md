# Reproduce and test the experiments

Use the [reviewer quickstart](REVIEWER_QUICKSTART.md) for installation and the
small three-node demo. Add `--core` to recompute and compare the 10 core tables:

~~~powershell
py -3.13 scripts/reviewer_quickstart.py --core
~~~

~~~bash
python3.13 scripts/reviewer_quickstart.py --core
~~~

Both commands create a fresh isolated environment, install CPU dependencies,
verify inputs and preserve logs and numerical receipts. Existing extracted
inputs can be selected with `--data-root`.

## Scientific scope

The core/CERES/historical-WHEC runner below uses the October 8 computational
inputs. Its `all` profile compares 24 tables. The later
[WHEC temporal-control correction](WHEC_AMENDMENT_20261009.md) has separate
entry points and reference data. Neither profile reruns all raw preprocessing,
later experiments or optional model training. A new result is verified by its
own receipt; the [historical records](GITHUB_EXECUTION.md) identify earlier runs.

The core release includes 68,668 × 66 × 4 discovery-fitted regional series, the
candidate graph, screening reference and original result tables. The runner
checks all core reference CSVs, including 2,574 candidate-lag tests and the
symmetric family. It checks ordering, labels, Boolean/integer decisions and
numeric values at `rtol=1e-7, atol=1e-10`.

## Direct commands for additional checks

The one-command route is sufficient for the demo and core comparison. For
additional commands, this example explicitly creates or reuses a named
environment with the full CPU profile.

Windows:

~~~powershell
py -3.13 scripts/reviewer_quickstart.py --core --venv .venv-reviewer-full
~~~

Linux:

~~~bash
python3.13 scripts/reviewer_quickstart.py --core --venv .venv-reviewer-full
~~~

The commands below invoke that interpreter directly; no activation is required.
Choose a new output directory for each calculation.

### Unit checks and the broader synthetic smoke run

Windows:

~~~powershell
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:NUMEXPR_NUM_THREADS = '2'
& .\.venv-reviewer-full\Scripts\python.exe scripts/verify_artifacts.py
& .\.venv-reviewer-full\Scripts\python.exe -m pytest tests -q -rs
& .\.venv-reviewer-full\Scripts\python.exe -m revision.run_simulations --output outputs/my-smoke --stage all --null-repetitions 1 --graph-repetitions 1 --n-null 256 --n-graph 1200
~~~

Linux:

~~~bash
.venv-reviewer-full/bin/python scripts/verify_artifacts.py
.venv-reviewer-full/bin/python -m pytest tests -q -rs
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 .venv-reviewer-full/bin/python -m revision.run_simulations --output outputs/my-smoke --stage all --null-repetitions 1 --graph-repetitions 1 --n-null 256 --n-graph 1200
~~~

The launcher applies its own stage settings automatically. The broader smoke
run's `run_status.json` should report two null
and five graph repetitions: one per scenario. This is not full calibration.
The test report explains optional trained-model skips.

### Core inputs already downloaded as ZIPs

The downloader can verify and extract an offline copy of the versioned archive:

~~~powershell
& .\.venv-reviewer-full\Scripts\python.exe scripts/download_artifacts.py --asset core-inputs.zip --from-directory "D:\downloaded-zips"
~~~

~~~bash
.venv-reviewer-full/bin/python scripts/download_artifacts.py --asset core-inputs.zip --from-directory /path/to/downloaded-zips
~~~

It verifies both the ZIP and extracted files using
[release-manifest.json](../artifacts/release-manifest.json). Existing different
files are rejected. The output defaults to `artifacts/data`; pass that directory
to the quickstart's `--data-root` option. Package installation still needs an
available package source when creating a new environment.

### CERES and historical WHEC comparisons

Windows:

~~~powershell
& .\.venv-reviewer-full\Scripts\python.exe scripts/download_artifacts.py --asset all
& .\.venv-reviewer-full\Scripts\python.exe scripts/verify_artifacts.py --data-root artifacts/data
& .\.venv-reviewer-full\Scripts\python.exe scripts/reproduce.py --experiment all --data-root artifacts/data --output outputs/my-full-reproduction
~~~

Linux:

~~~bash
.venv-reviewer-full/bin/python scripts/download_artifacts.py --asset all
.venv-reviewer-full/bin/python scripts/verify_artifacts.py --data-root artifacts/data
.venv-reviewer-full/bin/python scripts/reproduce.py --experiment all --data-root artifacts/data --output outputs/my-full-reproduction
~~~

This adds both CERES climatology settings and the historical frozen WHEC tests.
The runner preserves and verifies the original WHEC `design.json`. Select
`--experiment ceres` or `--experiment whec` to run either alone. Allow several
GB for downloads/extraction and more memory than the core run. Exact sizes are
in the manifest.

The extended inputs include a deterministic 2017–2025 ERA5 cloud slice in
percent, with parent NetCDF hashes, retaining the original loader's values and
time order. CERES, vectors, WHEC series and preprocessing parameters are archived
inputs. Raw acquisition, conservative remapping, preprocessing and WHEC index
construction are separate stages in the [raw-data guide](RAW_DATA.md).
See [source information](RAW_DATA_SOURCES.md) and [artifact coverage](ARTIFACTS.md).

## Check the evidence

A scientific comparison must end with `status: passed`, all planned jobs
completed and each required table comparison passed. The runner records maximum
numeric differences, input hashes, analysis source hashes, exit codes and
elapsed time. Historical WHEC also checks its design, periods and decisions.
Keep logs and receipts with the results; report a failed check through
[troubleshooting](COMPATIBILITY.md).

The [GitHub workflows](GITHUB_EXECUTION.md) offer hosted checks and
processed-input comparisons. Figure redraws have their own commands in
the [circular-network](../visualization/circular_networks/) and
[seasonal/state](../visualization/seasonal_state_contrasts/) guides.

## Historical scripts and optional models

The [module guide](../revision/README.md) describes additional experiments.
Some helpers expect sibling `revision_outputs/` directories or explicit
input/output paths. Their preparation is separate from the reviewer launcher.
Weather-token, intervention and cloud-interface prototypes have their own
inputs, training and checks. Passing software checks does not establish
interventional identification or make previously inspected years independent.
