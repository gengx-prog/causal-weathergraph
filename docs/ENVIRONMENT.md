# Experimental and validation environments

Use **CPython 3.13.7 (64-bit)** and [`requirements-test.txt`](../requirements-test.txt)
for the repository's portable CPU checks. This profile installs the scientific
dependencies and pytest into an ordinary virtual environment. No GPU, provider
account, raw ERA5/CERES files, or inherited system packages are required for these
checks. Python 3.10+ in the original exploratory guide is a historical broad
requirement; the pinned validation profile targets Python 3.13.

## What each environment record means

| Record | Meaning |
| --- | --- |
| [`revision/environment.freeze.txt`](../revision/environment.freeze.txt) | Preserved revision-era package inventory. It includes unrelated workstation packages and the CUDA build of PyTorch. It is provenance, not the clean-install instructions. |
| [`environment/local-author-20261008.json`](../environment/local-author-20261008.json) | Observed author-machine environment on October 8, 2026. The existing revision virtual environment inherits system packages. Current hardware and package availability cannot establish what was used by every historical experiment. |
| [`environment/constraints-core-py313.txt`](../environment/constraints-core-py313.txt) | Exact direct dependency versions for the portable CPU profile. Zarr 3.1.5 is added for the existing general-purpose requirements; Zarr was absent from the observed author environment. Transitive packages are resolved by pip. |
| [`environment/local-clean-20261008.json`](../environment/local-clean-20261008.json) | Environment observed after a fresh, independent installation for release validation, with system package inheritance disabled. |
| [`environment/clean-installed-20261008.txt`](../environment/clean-installed-20261008.txt) | All package names and versions observed in that clean Windows installation, including transitive dependencies. This is an installation receipt; the cross-platform installation entry point remains `requirements-test.txt`. |
| [`environment/validation-clean-20261008.json`](../environment/validation-clean-20261008.json) | Actual local validation outcomes and their scope. GitHub runner results are recorded separately by the workflow. |

The original experiment commands, designs, seeds, input fingerprints, and output
manifests remain the source of provenance for individual experiments. Preserve
them and use new output directories for reproduction.

## Observed author machine

| Component | Observation on October 8, 2026 |
| --- | --- |
| Operating system | Microsoft Windows 11 Home, 64-bit, version 10.0.26200 |
| CPU | Intel Core Ultra 9 275HX, 24 physical cores / 24 logical processors |
| RAM reported by OS | 68,137,205,760 bytes (63.46 GiB) |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU; 24,463 MiB reported by `nvidia-smi` |
| NVIDIA driver | 610.88 |
| Python | CPython 3.13.7, AMD64 |
| NumPy / pandas / SciPy | 2.4.2 / 3.0.0 / 1.17.0 |
| scikit-learn / statsmodels | 1.8.0 / 0.14.6 |
| xarray / netCDF4 | 2026.9.0 / 1.7.4 |
| Tigramite / OR-Tools | 5.2.10.1 / 9.15.6755 |
| PyTorch, optional token experiment | 2.10.0+cu128, CUDA build 12.8, cuDNN 91002; CUDA available at collection |
| Existing revision environment | `include-system-site-packages = true` |

These are observed resources, not minimum hardware requirements. The portable
tests run on CPU. The separate neural weather-token experiment needs PyTorch and
its corresponding data/model inputs; it is outside the CPU validation profile.
Rebuilding editorial DOCX/PDF artifacts also needs additional document/LaTeX
tools and is not part of the CPU checks.

## Fresh installation and verification

Run from the repository root. Create a new virtual environment **without**
`--system-site-packages`. Internet access is needed for package installation;
the unit checks and small synthetic simulation do not fetch scientific data.

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
$validationPython = '.\.venv\Scripts\python.exe'
& $validationPython -m pip install -r requirements-test.txt
& $validationPython -m pip check
$env:PYTHONNOUSERSITE = '1'
$env:MPLBACKEND = 'Agg'
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:NUMEXPR_NUM_THREADS = '2'
& $validationPython scripts/collect_environment.py --output outputs/my-environment.json
& $validationPython -m pytest tests -q -rs
& $validationPython -m revision.run_simulations --output outputs/my-smoke --stage all --null-repetitions 1 --graph-repetitions 1 --n-null 256 --n-graph 1200
```

Linux/macOS shell:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pip check
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
python scripts/collect_environment.py --output outputs/my-environment.json
python -m pytest tests -q -rs
python -m revision.run_simulations --output outputs/my-smoke --stage all --null-repetitions 1 --graph-repetitions 1 --n-null 256 --n-graph 1200
```

Check each command's exit status before proceeding. Choose another smoke output
directory for a new run. The local installation was checked on Windows; Linux
and Windows GitHub runners are configured separately. macOS commands use the same
standard virtual-environment interface, but macOS has not been validated locally.

`-rs` displays tests skipped because optional recorded model artifacts are absent.
Those skips do not verify the omitted model artifacts. The smoke run uses only
one repetition per scenario and validates command execution and output formation;
it does not reproduce the paper's formal repetition counts or scientific results.
Full reproduction additionally needs the inputs and paths described in the
[reproduction guide](../revision/REPRODUCE.md).

The October 8 clean Windows validation installed this profile successfully and
reported no broken dependencies. The latest recorded unit run passed **112 tests**
and skipped **13** optional production-artifact checks, with zero failures or
errors. The synthetic smoke run completed both null scenarios (12 summary rows)
and all five graph scenarios (30 summary rows). The JSON validation receipt
preserves the initial 106-test run as well as the subsequent 125-test run after
portable cloud-input checks were added. These counts describe the recorded
checkout; later test additions can change them.

## Recording another machine

[`scripts/collect_environment.py`](../scripts/collect_environment.py) uses the
Python standard library and inspects optional packages only when installed.
It records OS, CPU/RAM, selected package versions, virtual-environment isolation,
thread settings and available NVIDIA information. It deliberately excludes
hostnames, user names, executable paths, serial numbers, network identifiers,
environment secrets and provider credentials. Unsupported inventory fields are
null rather than guessed. CUDA build/runtime information describes PyTorch;
it does not claim an independently installed CUDA toolkit.

GitHub Actions checks installation, dependency consistency, unit checks, artifact
integrity and the synthetic smoke run, then uploads logs and environment records.
A configured workflow is not evidence that GitHub has already run it successfully;
inspect the actual Actions run for the commit under review.
