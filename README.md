# Causal WeatherGraph experiments

Experimental code, atmospheric data, numerical results, visualization tools,
and reproducibility records for regional wind, humidity, and cloud dependencies.
The analyses examine lagged observational associations; they do not establish
interventional causation or physical propagation times.

## Run the experiments

Use a fresh Python 3.13.7 environment for the statistical checks:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
python -m pip install -r requirements-test.txt
python scripts/verify_artifacts.py
python -m pytest tests -q -rs
python scripts/download_artifacts.py --asset core-inputs.zip
python scripts/reproduce.py --experiment core --output outputs/core-check
```

The [reproduction guide](docs/REPRODUCING.md) covers the core, CERES, and WHEC
analyses. The `all` profile compares 24 reference CSV tables from the October 8
experimental configuration. The [WHEC correction guide](docs/WHEC_AMENDMENT_20261009.md)
describes the separate October 9 configuration and its reference comparisons.
These profiles do not rerun every archived experiment or acquire the full raw data.

## Data and results

- [Processed inputs and experiment evidence](https://github.com/gengx-prog/causal-weathergraph/releases/tag/reproducibility-20261008.1)
- [Additional experiment results, circular plots, and corrected WHEC inputs](https://github.com/gengx-prog/causal-weathergraph/releases/tag/revision-20261010)
- [Raw ERA5/WB2 and CERES data](docs/RAW_DATA.md), with [provider and licensing information](docs/RAW_DATA_SOURCES.md)
- [Artifact inventories and SHA-256 verification](docs/ARTIFACTS.md)
- [Analysis modules](revision/README.md) and [validation records](validation/README.md)

Raw input archives total approximately 108.69 GB. Select only the datasets needed
for an experiment; processed inputs provide a smaller starting point.

## Software and compatibility

See the [software guide](docs/SOFTWARE.md) for Python, dependency constraints
and isolated installation instructions. The core statistical analyses use the
CPU; optional neural models have separate PyTorch and input requirements.
The [compatibility guide](docs/COMPATIBILITY.md) covers missing inputs,
dependency conflicts and workload resource limits. Reproducible software defects
should be reported through the issue tracker.

## Visualize experimental results

```bash
python -m pip install -r requirements-figures.txt
python scripts/verify_experiment_figures.py --output outputs/circular-figure-check
```

The `visualization/` directory contains the source values and renderers for circular
regional networks and seasonal/state contrast plots. Results are exported as PDF,
SVG, and PNG, with numerical checks against the recorded source values.

## Automated checks

- [CPU tests and artifact integrity](https://github.com/gengx-prog/causal-weathergraph/actions/workflows/reproducibility.yml)
- [Experiment reproduction](https://github.com/gengx-prog/causal-weathergraph/actions/workflows/experiment-reproduction.yml)
- [Experimental figures](https://github.com/gengx-prog/causal-weathergraph/actions/workflows/experiment-figures.yml)
- [Raw-data checks](https://github.com/gengx-prog/causal-weathergraph/actions/workflows/raw-data-check.yml)

The exploratory pipeline remains available through `scripts/run_all.py` and
`configs/default.yaml`. Configure the data location before running it.
