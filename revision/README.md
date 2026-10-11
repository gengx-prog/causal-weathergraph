# Atmospheric dependency experiments

For a fresh checkout, use the [portable reproduction guide](../docs/REPRODUCING.md)
and [software installation guide](../docs/SOFTWARE.md). The processed-input release
supplies core, satellite and WHEC inputs. `scripts/reproduce.py` passes explicit
paths and compares rerun CSVs with archived numerical evidence. The separate
[raw-data archive](../docs/RAW_DATA.md) retains ERA5/WB2 and CERES acquisitions,
with [official source descriptions](../docs/RAW_DATA_SOURCES.md).

## Main entry points

| Analysis | Files |
| --- | --- |
| Discovery-fitted preprocessing and diurnal sensitivity | `prepare_inputs.py`, `prepare_inputs_diurnal.py` |
| Screen-confirm-replicate graph evaluation | `run_three_stage.py`, `full_var_tests.py` |
| Dependence-aware inference and graph controls | `inference.py`, `run_inference.py`, `graph_nulls.py`, `run_graph_nulls.py` |
| Native ERA5 regridding and source overlap | `download_era5_primary_native.py`, `regrid_era5_primary_native.py`, `run_native_route_check.py` |
| CERES cloud substitution | `prepare_ceres_cloud_6h.py`, `run_ceres_cloud_substitution.py` |
| Original WHEC calculation | `make_whec_index.py`, `run_whec_test.py` |
| Corrected WHEC temporal control | `make_whec_index_20261009.py`, `run_whec_test_20261009.py`, `verify_whec_20261009.py` |
| State, latitude, hemisphere and lag sensitivity | `run_regime_contrasts.py`, `run_latitude_contrasts.py`, `run_hemisphere_contrasts.py`, `run_joint_window_by_segment.py` |
| Known-structure benchmarks | `run_simulations.py`, `run_rpcmci_benchmark.py`, `run_castle_benchmark.py` |

## Environment and input boundaries

Run Python modules from the repository root. Install `requirements-test.txt` in
a fresh virtual environment for the pinned scientific/test profile. The dated
`environment.freeze.txt` records additional optional packages; it is not the
clean-install profile. Acquisition and optional neural experiments may also
need `cdsapi` or PyTorch, respectively.

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

Raw fields and large intermediate arrays are separate inputs. Historical helpers
may expect sibling `revision_outputs/` and `supplementary_data/` directories or
platform-specific shell conventions. Supply explicit input/output options where
available, and verify adaptations of fixed paths in a separate reproduction copy.
Use a new output directory for every rerun.

```bash
python -m revision.run_three_stage --input /path/to/region_trainfit.npz --output /path/to/new/three_stage
python -m revision.make_whec_index --data-root /path/to/native/data_root --output /path/to/new/whec_index
```

The original `run_whec_test.py` requires the exact frozen `design.json`, with
SHA-256 `0507157c5a8ec0b248a36bca29818af6623ba86b1e19576f57fd743d92beee92`.
It also expects recorded regional, index and satellite inputs; changing only its
output directory does not relocate those inputs. The portable reproduction
runner configures the historical inputs explicitly. The corrected temporal
control has [separate commands and evidence](../docs/WHEC_AMENDMENT_20261009.md).
Do not substitute the original and corrected indices for each other.

Individual module READMEs describe additional experiments and limitations.
The [artifact guide](../docs/ARTIFACTS.md) identifies the inputs, designs and
numerical references included in each data archive.

The graph estimates describe conditional linear predictive dependencies in
observational records. They do not establish atmospheric intervention effects.
