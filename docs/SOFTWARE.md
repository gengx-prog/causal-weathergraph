# Software installation and dependencies

Use **CPython 3.13** for the documented CPU workflow. The existing recorded
scientific environment uses CPython 3.13.7. The reviewer launcher creates an
isolated environment and installs pinned dependencies, so no activation or
manual package selection is needed.

Windows PowerShell:

~~~powershell
py -3.13 scripts/reviewer_quickstart.py
~~~

Linux:

~~~bash
python3.13 scripts/reviewer_quickstart.py
~~~

Run from the repository root. Initial installation needs internet access.
The default demo generates its own data. Append `--core` to install the broader
analysis profile and compare 10 core atmospheric result tables. The
[reviewer quickstart](REVIEWER_QUICKSTART.md) explains the saved evidence and
existing-input option.

## Dependency profiles

| File | Scope |
| --- | --- |
| [requirements-demo.txt](../requirements-demo.txt) | Small CPU demo: NumPy 2.4.2, SciPy 1.17.0 and pandas 3.0.0. |
| [requirements-test.txt](../requirements-test.txt) | Broader CPU analysis/test profile, including netCDF/xarray, statistical, graph and test tools. Used by `--core` and scientific CI. |
| [constraints-core-py313.txt](../environment/constraints-core-py313.txt) | Exact direct versions shared by the profiles. Constraints limit requested packages; they do not install every listed package. |
| [requirements-figures.txt](../requirements-figures.txt) | Standalone experimental figure checks. |
| [requirements.txt](../requirements.txt) | General dependency list; use the constrained profiles above for the documented reviewer route. |

Direct versions are pinned; pip resolves transitive dependencies. Each launcher
run saves `pip-freeze.txt` with the installed set and performs `pip check`.
The interpreter record, installation log and verification receipt make the
environment for that run inspectable.

By default the launcher creates a new `.venv-reviewer-*` environment, without
`--system-site-packages`, and invokes its Python directly. Explicit `--venv`
opts into creating or reusing a named dedicated environment. Its directory is
recorded in `verification.json`. The demo uses one numerical thread; the core runner uses
two. Keep experiment-specific settings for other analyses, including the
October 9 calibration's one BLAS thread per worker.

## Additional checks with the full profile

The core option installs the dependencies needed by the unit tests. For a named
environment that can be used in the commands below, explicitly select
`.venv-reviewer-full` when running the core option.

Windows:

~~~powershell
py -3.13 scripts/reviewer_quickstart.py --core --venv .venv-reviewer-full
& .\.venv-reviewer-full\Scripts\python.exe -m pytest tests -q -rs
~~~

Linux:

~~~bash
python3.13 scripts/reviewer_quickstart.py --core --venv .venv-reviewer-full
.venv-reviewer-full/bin/python -m pytest tests -q -rs
~~~

The report lists checks skipped because separately trained model artifacts are
absent. A skipped check does not verify that model. The broader synthetic
benchmark and real-data comparisons are documented in
[REPRODUCING.md](REPRODUCING.md).

## Historical environment records

| File | Recorded scope |
| --- | --- |
| [clean-installed-20261008.txt](../environment/clean-installed-20261008.txt) | Full package inventory from the recorded clean installation. |
| [validation-clean-20261008.json](../environment/validation-clean-20261008.json) | Historical installation, unit-test and synthetic-smoke outcomes. |
| [calibration-software-20261009.json](../environment/calibration-software-20261009.json) | Python, packages and BLAS setting for that calibration. |
| [revision/environment.freeze.txt](../revision/environment.freeze.txt) | Historical workstation inventory including optional frameworks; not the reviewer installation file. |

The CPU demo and core analysis need no GPU or CUDA. The optional weather-token
experiment has a separate PyTorch setup, inputs and checkpoints. Provider
packages and access are relevant to raw-data acquisition, not the processed
public inputs used by the core route. See [workflow scope](GITHUB_EXECUTION.md)
and [troubleshooting](COMPATIBILITY.md).
