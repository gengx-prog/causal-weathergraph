# Causal WeatherGraph experiments

Experimental code, atmospheric data, numerical results and reproducibility tools
for regional wind, humidity and cloud dependencies. The analyses examine lagged
observational associations; they do not establish interventional causation or
physical propagation times.

## Start with one command

Download or clone this repository, install **CPython 3.13**, and open a terminal
in the repository folder. The launcher creates an isolated environment, installs
the pinned CPU dependencies and runs a small three-node regression experiment
using the project's scientific code.

Windows PowerShell:

~~~powershell
py -3.13 scripts/reviewer_quickstart.py
~~~

Linux:

~~~bash
python3.13 scripts/reviewer_quickstart.py
~~~

No environment activation, GPU, CUDA, provider account or atmospheric-data
download is needed. Initial dependency installation uses the internet.
Results, logs, installed versions and a verification receipt are preserved in a
new `outputs/reviewer-demo-*` directory. Installation time is recorded separately
from computation.

To recompute the **10 core result tables from archived atmospheric inputs**,
append `--core`:

~~~powershell
py -3.13 scripts/reviewer_quickstart.py --core
~~~

On Linux, use `python3.13` in place of `py -3.13`. This option installs the broader
CPU profile, obtains verified inputs and compares the computed tables with their
references. The [reviewer quickstart](docs/REVIEWER_QUICKSTART.md) explains the
outputs, download size and existing-input option.

复现入口自动建立隔离环境、安装依赖并运行小型数值实验，无需 GPU 或 CUDA。
加上 `--core` 可核对 10 张核心结果表。日志、环境及核对记录保存在新输出目录中。

## Reproduce the other experiments

The [reproduction guide](docs/REPRODUCING.md) covers the core, CERES and historical
WHEC analyses. Its `all` profile compares 24 tables from the October 8
configuration. The [WHEC correction guide](docs/WHEC_AMENDMENT_20261009.md)
documents the separate October 9 temporal-control correction. Each route
identifies its inputs and scientific scope.

## Data and results

- [Quickstart release and verified execution bundle (13 October)](https://github.com/gengx-prog/causal-weathergraph/releases/tag/quickstart-20261013)
- [Processed inputs and experiment evidence](https://github.com/gengx-prog/causal-weathergraph/releases/tag/reproducibility-20261008.1)
- [Additional results, circular plots and corrected WHEC inputs](https://github.com/gengx-prog/causal-weathergraph/releases/tag/revision-20261010)
- [Raw ERA5/WB2 and CERES data](docs/RAW_DATA.md), with [provider and licensing information](docs/RAW_DATA_SOURCES.md)
- [Artifact inventories and SHA-256 verification](docs/ARTIFACTS.md)
- [Analysis modules](revision/README.md) and [validation records](validation/README.md)

Raw archives total approximately 108.69 GB. The demo generates its own input;
the core option uses the much smaller processed-input archive.

## Installation and help

The [software guide](docs/SOFTWARE.md) explains the dependency profiles.
The [troubleshooting guide](docs/COMPATIBILITY.md) explains how to locate the
failing stage and report it. Failures in the documented route should be reported
so maintainers can investigate the code, dependencies and instructions.
Optional neural models and trained prototypes have separate setup steps.

## Visualize experimental results

Use the interpreter from your chosen virtual environment:

~~~bash
python -m pip install -r requirements-figures.txt
python scripts/verify_experiment_figures.py --output outputs/circular-figure-check
~~~

The `visualization/` directory contains source values and renderers for circular
regional networks and seasonal/state contrasts, with numerical and vector-export
checks.

## Automated checks

- [One-command demo on Windows and Ubuntu](https://github.com/gengx-prog/causal-weathergraph/actions/workflows/reviewer-demo.yml)
- [CPU tests and artifact integrity](https://github.com/gengx-prog/causal-weathergraph/actions/workflows/reproducibility.yml)
- [Experiment reproduction](https://github.com/gengx-prog/causal-weathergraph/actions/workflows/experiment-reproduction.yml)
- [Experimental figures](https://github.com/gengx-prog/causal-weathergraph/actions/workflows/experiment-figures.yml)
- [Raw-data checks](https://github.com/gengx-prog/causal-weathergraph/actions/workflows/raw-data-check.yml)

See [workflow scope and receipts](docs/GITHUB_EXECUTION.md). The exploratory
pipeline remains available through `scripts/run_all.py` and
`configs/default.yaml`; configure its data location before running it.

## Recorded local verification before publication

The [13 October local receipt](validation/quickstart-local-20261013.json)
records two fresh CPython 3.13.7 environments on Windows 11: the generated-data
demo passed 13 numerical checks and the public-input core run passed all 10
reference-table comparisons. A separately run test suite passed 200 tests;
13 checks requiring optional trained artifacts were skipped. Installed sets
are retained in [demo](environment/quickstart-demo-installed-20261013.txt) and
[core](environment/quickstart-core-installed-20261013.txt) inventories.

The [release bundle](https://github.com/gengx-prog/causal-weathergraph/releases/tag/quickstart-20261013)
preserves the source and full logs exactly as prepared for these local checks.
Its references to changes not yet being pushed describe that earlier preparation
stage; the quickstart is now published in this repository. See the
[publication notes](docs/QUICKSTART_RELEASE_20261013.md) for the bundle hash,
verification scope and distinction between archived evidence and current source.

Hosted results are reported separately in the
[Windows/Ubuntu quickstart runs](https://github.com/gengx-prog/causal-weathergraph/actions/workflows/reviewer-demo.yml)
and [CPU test runs](https://github.com/gengx-prog/causal-weathergraph/actions/workflows/reproducibility.yml).
Inspect the run for the commit being reviewed; the local receipt does not by
itself establish a hosted or cross-platform pass.
