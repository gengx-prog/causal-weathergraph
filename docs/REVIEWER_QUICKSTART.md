# Reviewer quickstart

Start with the small demo to check installation and the numerical workflow.
Add `--core` to compare the paper's core atmospheric results. Both use CPU
computation.

## One-command demo

Prerequisites: a downloaded or cloned copy of this repository, **CPython 3.13**,
and internet access for the initial package installation. Open a terminal in
the repository folder, where `README.md` is located.

Windows PowerShell:

```powershell
py -3.13 scripts/reviewer_quickstart.py
```

Linux:

```bash
python3.13 scripts/reviewer_quickstart.py
```

The launcher performs these steps:

1. Checks Python and creates a new isolated `.venv-reviewer-*` environment. It
   invokes that environment directly, without shell activation.
2. Installs the pinned NumPy, SciPy and pandas profile from
   [`requirements-demo.txt`](../requirements-demo.txt) and runs `pip check`.
3. Runs [`scripts/run_demo.py`](../scripts/run_demo.py), a three-node synthetic
   lagged-regression experiment using the project's scientific routines.
4. Preserves the results in a new `outputs/reviewer-demo-*` directory and
   prints its location.

The demo needs no atmospheric inputs, provider credentials, GPU, CUDA or model
training. It checks a small generated example; it does not substitute for the
atmospheric comparisons below.

## Inspect the result

Inspect the generated `verification.json` and process exit status to determine
whether the checks passed. The launcher output directory includes:

| File | Purpose |
| --- | --- |
| `install.log` | Dependency installation output and errors. |
| `pip-check.log` | Package-consistency check. |
| `run.log` | Demo progress, results and errors. |
| `pip-freeze.txt` | Exact installed package versions. |
| `environment.json` | Interpreter and execution-environment record. |
| `verification.json` | Completion and numerical-check results. |
| `demo/synthetic_data.csv` | Generated three-node time series. |
| `demo/truth.csv` | Known generating relationships. |
| `demo/fitted_tests.csv` | Fitted regression and inference results. |
| `demo/summary.json` | The demo's 13 numerical checks and summary. |

The core option adds `download.log` when downloading inputs, `core-run.log`
for analysis progress and `core/verification.json` for per-table comparisons.
The top-level receipt records the environment directory, each step's exit code
and timing, and hashes of the source files and evidence.

Read the numerical checks as well as completion status. Installation and
scientific execution are timed separately; network speed and package caching
affect setup time. Repeated launches use new result directories, preserving
earlier evidence. The demo uses one numerical thread; the core runner uses two.
The default also creates a fresh virtual environment for each attempt. Use
`--venv .venv-reviewer-shared` only when you intentionally want to create or
reuse a named dedicated environment; it must use CPython 3.13 without system
site packages. The selected directory is recorded in `verification.json`.

## Recompute the core atmospheric results

Windows:

```powershell
py -3.13 scripts/reviewer_quickstart.py --core
```

Linux:

```bash
python3.13 scripts/reviewer_quickstart.py --core
```

The core option also installs [`requirements-test.txt`](../requirements-test.txt),
obtains the versioned `core-inputs.zip`, verifies its files and invokes
[`scripts/reproduce.py`](../scripts/reproduce.py). It compares all 10 core CSV
tables, including screen–confirm–replicate and symmetric full-VAR results.
Numeric comparisons use `rtol=1e-7, atol=1e-10`; row/column order, labels and
Boolean or integer decisions are checked too.

The archive has 16 files: approximately **145 MB to download** and **151 MB
after extraction** (decimal units), plus environment and computation space.
It contains processed inputs and reference outputs. This route does not
download the 108.69 GB raw archive or repeat raw-field preprocessing. Exact
sizes and hashes are in the [release manifest](../artifacts/release-manifest.json).

For **already extracted inputs**, supply their root directory. It must contain
`inputs/region_trainfit.npz` and the other manifest paths; the launcher verifies
them before analysis:

```powershell
py -3.13 scripts/reviewer_quickstart.py --core --data-root "D:\review-data"
```

```bash
python3.13 scripts/reviewer_quickstart.py --core --data-root /path/to/review-data
```

This skips the input download, while retaining dependency installation and file
verification. For a downloaded ZIP, use the downloader's `--from-directory`
route in [REPRODUCING.md](REPRODUCING.md).

## If a step fails

Share the command, repository commit, Python version, verification receipt and
relevant error trace through the
[issue tracker](https://github.com/gengx-prog/causal-weathergraph/issues).
Remove credentials and private paths before sharing logs. Maintainers can use
this evidence to investigate and fix a repeatable installation or execution
problem in the documented route.

See [troubleshooting](COMPATIBILITY.md) for specific remedies and
[GitHub execution](GITHUB_EXECUTION.md) for the hosted CPU workflows. Docker and
Colab are not prerequisites; this guide does not claim a tested container or
notebook deployment.
