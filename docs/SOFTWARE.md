# Software installation and dependencies

The recorded reference profile uses **CPython 3.13.7** and
[`requirements-test.txt`](../requirements-test.txt) for the portable statistical
checks. The versions and commands on this page are provided for reference, not
as mandatory or exclusive conditions for reproduction. Equivalent compatible
software may be used, with its outputs checked against the relevant experimental
criteria. The pinned profile is an installation example, not a guarantee that
every platform can install or run it. See the
[reference and compatibility notice](COMPATIBILITY.md#reference-status-and-compatibility).

For the example profile, create an isolated virtual environment without
`--system-site-packages`. The unit checks and small synthetic simulation do not
require meteorological downloads, provider accounts or a GPU.

## Dependency records

| File | Scope |
| --- | --- |
| [`environment/constraints-core-py313.txt`](../environment/constraints-core-py313.txt) | Exact direct versions for the scientific/test profile. Transitive packages are resolved by pip; this is not a cross-platform lock of all binary dependencies. |
| [`environment/clean-installed-20261008.txt`](../environment/clean-installed-20261008.txt) | Package names and versions from the recorded clean installation, including transitive dependencies. Use `requirements-test.txt` as the installation entry point. |
| [`environment/calibration-software-20261009.json`](../environment/calibration-software-20261009.json) | Python and scientific package versions, plus the BLAS thread setting, recorded for the October 9 calibration. It does not describe every experiment. |
| [`environment/validation-clean-20261008.json`](../environment/validation-clean-20261008.json) | Historical clean-installation, unit-check and synthetic-smoke outcomes. Later code changes require fresh checks. |
| [`revision/environment.freeze.txt`](../revision/environment.freeze.txt) | Historical package inventory, including optional packages. It is not the clean-install profile. |

The core constraints include NumPy 2.4.2, pandas 3.0.0, SciPy 1.17.0,
scikit-learn 1.8.0, statsmodels 0.14.6, xarray 2026.9.0, netCDF4 1.7.4,
Tigramite 5.2.10.1 and OR-Tools 9.15.6755. See the constraints file for the
complete direct-dependency list. Zarr 3.1.5 supports the general-purpose
requirements. Optional data acquisition and neural experiments may additionally
need `cdsapi` or PyTorch; neither is a prerequisite for the portable checks.

## Install and check

Run commands from the repository root. Package installation requires internet
access. Use a new output directory for each simulation attempt.

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
python -m pytest tests -q -rs
python -m revision.run_simulations --output outputs/my-smoke --stage all --null-repetitions 1 --graph-repetitions 1 --n-null 256 --n-graph 1200
```

Check each command's exit status. The shell instructions use standard virtual
environments; they do not establish that every platform has been tested.
`-rs` reports tests skipped when optional trained-model artifacts are absent.
Skipped checks do not verify those models. The smoke run checks execution and
output formation with one repetition per scenario; it does not reproduce full
calibration or meteorological results.

Use the [reproduction guide](REPRODUCING.md) for archived scientific inputs and
numerical comparisons. See [workflow scope](GITHUB_EXECUTION.md) for recorded
checks and [troubleshooting](COMPATIBILITY.md) for diagnosis. Preserve each
experiment's own thread settings and compare floating-point results using its
stated tolerances.
