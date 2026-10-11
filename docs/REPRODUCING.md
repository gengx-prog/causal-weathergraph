# Reproduce and test the experiments

**Scope:** This page documents the October 8 computational inputs and analyses.
The later [WHEC temporal-control correction](WHEC_AMENDMENT_20261009.md) has
separate entry points and reference data. The historical 24-table workflow does
not cover every later experiment. Historical numerical receipts do not replace
fresh validation of the current checkout.

All materials and commands are supplied for research and reference. The setup
below is one recorded route, not a mandatory or exclusive reproduction
environment. An equivalent compatible setup may be used. Reproducing a specified
result still calls for its corresponding scientific inputs, preprocessing,
analysis choices, and comparison criteria. See the
[reference and compatibility notice](COMPATIBILITY.md#reference-status-and-compatibility)
for conditions that can prevent execution in another environment.

Start here after cloning the repository. Run commands from its root. The
[software guide](SOFTWARE.md) gives Python/dependency versions and isolated
installation steps for the supported commands.
To follow the reference profile, use CPython 3.13.7 and
`python -m pip install -r requirements-test.txt` in a new virtual environment.
In PowerShell, replace `python` below with
`& .\.venv\Scripts\python.exe` if that environment is not activated.

For GitHub Actions, first read [the execution status and remedies](GITHUB_EXECUTION.md).
It distinguishes recorded workflows,
raw-data resource limits and optional prototype models that are not included.

## 1. Check the checkout without meteorological downloads

```bash
python scripts/verify_artifacts.py
python -m pip check
python -m pytest tests -q -rs
python -m revision.run_simulations --output outputs/my-smoke --stage all --null-repetitions 1 --graph-repetitions 1 --n-null 256 --n-graph 1200
```

The small simulation must report two completed null repetitions and five graph
repetitions in `outputs/my-smoke/run_status.json`. These are one repetition for
each scenario, not the full published benchmark. The 13 cloud simulator tests
that need separately trained production models are expected to skip. Unit tests,
file hashes and a synthetic smoke run do not reproduce the complete experiments.
Use a new output directory on each attempt.

On GitHub, every push/pull request runs the CPU checks on Windows and Ubuntu.
For real-data verification, open **Actions → Experiment reproduction from archived
inputs → Run workflow**, then select `core` or `all`. This workflow downloads
the published release and uploads the comparison report, logs, result CSVs and
runner environment as an Actions artifact.

## 2. Recompute the core graph results from real archived inputs

```bash
python scripts/download_artifacts.py --asset core-inputs.zip
python scripts/verify_artifacts.py --data-root artifacts/data --asset core-inputs.zip
python scripts/reproduce.py --experiment core --data-root artifacts/data --output outputs/my-core-reproduction
```

The versioned [release](https://github.com/gengx-prog/causal-weathergraph/releases/tag/reproducibility-20261008.1)
contains the actual 68,668 × 66 × 4 discovery-fitted regional series, the
candidate graph, the screening reference, and the original result tables.
The downloader verifies both the ZIP and every extracted file using
[`release-manifest.json`](../artifacts/release-manifest.json). Existing files
with different contents are rejected. Downloads and extracted arrays are ignored
by Git. To work offline, transfer the ZIP and use
`--from-directory /path/to/downloaded/zips` with the same downloader.

`reproduce.py` runs screen–confirm–replicate and symmetric full-VAR analyses,
then compares every core reference CSV, including all 2,574 candidate-lag tests
and the symmetric family. Row order, labels, Boolean decisions and numeric
values are checked; the numeric tolerance is `rtol=1e-7, atol=1e-10`.
`verification.json` records each table, maximum absolute numeric difference,
runner hash, exit status and duration. Runtime and memory use depend on the
selected workload and execution settings.

## 3. Recompute the satellite comparison and WHEC tests

```bash
python scripts/download_artifacts.py --asset all
python scripts/verify_artifacts.py --data-root artifacts/data
python scripts/reproduce.py --experiment all --data-root artifacts/data --output outputs/my-full-reproduction
```

This adds both CERES climatology settings and the frozen WHEC tests. The runner
copies the original WHEC `design.json` into the new output directory; its
published SHA-256 must still match. All paths are explicit and use downloaded
inputs. Choose `--experiment ceres` or `--experiment whec` to run either alone.
Allow several GB of download/extraction space and more memory than the core run.
Download sizes and complete file inventories are in the release manifest.

The extended-input release includes a deterministic 2017–2025 ERA5 cloud slice
in percent, with hashes of its parent NetCDF files. It preserves the original
loader's values and time order. CERES, regional vectors, WHEC series and fitted
preprocessing parameters are archived inputs; this workflow does **not** rerun
raw-data acquisition, conservative remapping, preprocessing, or WHEC index
construction. The retained original study inputs and provider subsets are now
available in the separate [raw-data archive](RAW_DATA.md), with [official source
links](RAW_DATA_SOURCES.md). See [artifact coverage](ARTIFACTS.md) for the
processed-input scope.

## Historical scripts and additional experiments

The [analysis module guide](../revision/README.md) describes additional
experiments. Some helpers expect sibling `revision_outputs/` directories or require explicit
input/output paths; consult their command options before running them. The commands above are the portable
entry points with recorded numerical checks. Separate weather-token, intervention
and cloud-interface prototypes have their own inputs and are not part of these
core-result replication checks. Passing software checks does not establish
interventional identification or make previously inspected years independent.

## Check experimental figures

The experimental visualizations have their own archived data and numerical
checks. They do not rerun the underlying discovery or calibration calculations.

```bash
python -m pip install -r requirements-figures.txt
python scripts/verify_experiment_figures.py --output outputs/experiment-figure-checks
```

See the [circular-network renderer](../visualization/circular_networks/) and
[seasonal/state renderer](../visualization/seasonal_state_contrasts/) for their
input tables, display encodings and checks. The experiment-figure workflow
runs these checks independently of the core/CERES/WHEC numerical workflow.
