# Analyses supporting the revised manuscript

For a fresh GitHub checkout, use the **[portable reproduction guide](../docs/REPRODUCING.md)**
and **[recorded environment](../docs/ENVIRONMENT.md)**. The versioned release
now supplies processed core, satellite and WHEC inputs. `scripts/reproduce.py`
passes explicit paths and compares rerun CSVs with the archived evidence.

Author: Xinchen Geng, University of Southern California.

This directory contains the analysis code used during the revision of **Causal WeatherGraph: screen–confirm–replicate discovery of lagged wind–humidity–cloud dependencies in global atmospheric reanalysis**. The final manuscript and machine-readable results are linked from the [repository README](../README.md).

## Main entry points

| Analysis | Files |
| --- | --- |
| Discovery-period preprocessing and diurnal sensitivity | `prepare_inputs.py`, `prepare_inputs_diurnal.py` |
| Screen–confirm–replicate graph evaluation | `run_three_stage.py`, `full_var_tests.py` |
| Autocorrelation-consistent inference and graph controls | `inference.py`, `run_inference.py`, `graph_nulls.py`, `run_graph_nulls.py` |
| Native ERA5 conservative regridding and source overlap | `download_era5_primary_native.py`, `regrid_era5_primary_native.py`, `run_native_route_check.py` |
| CERES cloud substitution | `prepare_ceres_cloud_6h.py`, `run_ceres_cloud_substitution.py` |
| Wind–humidity eddy convergence (WHEC) | `make_whec_index.py`, `run_whec_test.py` |
| Regime, latitude, hemisphere and lag sensitivity | `run_regime_contrasts.py`, `run_latitude_contrasts.py`, `run_hemisphere_contrasts.py`, `run_joint_window_by_segment.py` |
| Known-structure benchmarks | `run_simulations.py`, `run_rpcmci_benchmark.py`, `run_castle_benchmark.py` |
| Revised figures and tables | `make_framework_figure_v2.py`, `make_paper_figures_v2.py`, `make_paper_tables_v2.py`, `make_supp_tables_v2.py`, `make_whec_figure.py` |

## Environment and inputs

Run Python modules from the repository root. [`requirements.txt`](requirements.txt) includes the base dependencies, core revision dependencies and test runner. Data acquisition and optional document/forecasting helpers have additional dependencies recorded in [`environment.freeze.txt`](environment.freeze.txt), including `cdsapi`, `beautifulsoup4`, `numcodecs`, `python-docx`, `Pillow`, `PyMuPDF`, `pypdf` and `torch`. That file records the research environment, including packages unrelated to the core analyses. The framework figure renderer also expects a local Chrome installation at the path specified in its source.

```bash
python -m pip install -r revision/requirements.txt
python -m pytest -q
```

Raw meteorological fields and large intermediate arrays are external inputs. The scripts preserve the recorded research directory conventions, including defaults under sibling `revision_outputs/` and `supplementary_data/` directories. Shell orchestration scripts contain the original Windows/Git Bash paths. Supply the available command-line input/output options, or adapt fixed paths in a separate reproduction copy before running elsewhere. Do not reuse recorded result directories for a new analysis.

For example, after preparing the required regional input array:

```bash
python -m revision.run_three_stage --input /path/to/region_trainfit.npz --output /path/to/new/three_stage
python -m revision.make_whec_index --data-root /path/to/native/data_root --output /path/to/new/whec_index
```

`run_whec_test.py` verifies the SHA-256 of `design.json` in its output directory before execution. The original frozen design is `S6a_whec_design.json` in the supplementary data ZIP; its hash is `0507157c5a8ec0b248a36bca29818af6623ba86b1e19576f57fd743d92beee92`. Preserve those bytes when copying it as `design.json`. This runner also expects the recorded regional inputs, WHEC index and satellite input paths described in its source; changing only `--output` does not relocate its inputs.

[`REPRODUCE.md`](REPRODUCE.md) retains the earlier experiment commands and limitations. It includes local historical paths and references to intermediate reports retained in the research workspace. The final revision package and this guide are the entry points for the published revision.

## Reported data

[`4_Supplementary_Data.zip`](../manuscript/submission_20261002/4_Supplementary_Data.zip) includes its own README and manifest, region definitions, screening/confirmation/replication results, symmetric full-VAR tests, regime tests, CERES comparisons, source-route diagnostics, and WHEC design and results. The published data package preserves the original analysis outputs.

The graph estimates describe conditional linear predictive dependencies in observational records. They do not establish atmospheric intervention effects.
