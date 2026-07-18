# Causal WeatherGraph

**Regime-Aware Spatio-Temporal Causal Discovery for Global Atmospheric Dynamics**

Causal WeatherGraph is a lightweight Python research codebase for discovering regime-dependent lagged directed dependencies in global atmospheric fields. The central scientific question is whether observational reanalysis time series contain robust evidence for a moisture transport pathway:

```text
Wind -> Humidity -> Cloud Cover
```

Temperature is treated as an important contextual/background variable. The code is designed for scientific exploration, not weather forecasting and not reproduction of any existing model.

## Scientific Motivation

Atmospheric transport is dynamic and regime dependent. A relationship that is visible in humid tropical states may weaken in polar or low-cloud states; similarly, seasonal circulation can change which regions behave as upstream sources for downstream humidity or cloud changes.

This repository builds a first-pass pipeline for:

- finding which variables tend to precede others in time,
- identifying spatial source regions for humidity and cloud-cover changes,
- comparing directed dependency graphs across regimes,
- quantifying whether Wind -> Humidity and Humidity -> Cloud links strengthen in specific regimes.

Throughout the code and outputs, edges should be interpreted cautiously as **lagged causal dependencies** or **causal-discovery-inspired directed dependencies** from observational time series. They are not interventional proof.

## What The Code Does

The pipeline:

1. Inspects local atmospheric data files under `/home/vipuser/Data`.
2. Loads NetCDF, Zarr, NumPy, CSV, or parquet files when possible.
3. Detects common aliases for temperature, humidity, wind, and cloud cover.
4. Computes wind speed from u/v wind components when both are available.
5. Converts data to a unified `[time, node, variable]` array.
6. Handles missing values, removes seasonal anomalies, optionally differences, and standardizes time series.
7. Aggregates grid cells or nodes into regions, defaulting to 64 latitude-longitude bins.
8. Defines regimes such as all, season, high/normal humidity, and high/normal cloud.
9. Constructs targeted candidate edges rather than testing every possible all-to-all edge.
10. Runs Granger-style lagged OLS tests with FDR correction.
11. Computes regime metrics and Wind -> Humidity -> Cloud chain summaries.
12. Generates figures for maps, regimes, lags, networks, and chains.

## Installation

Use Python 3.10+.

```bash
cd /home/vipuser/论文2026_516/Causal/causal-weathergraph
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your shell does not provide `python`, use `python3` or your Conda Python for the same commands.

Additional packages:

- `cartopy` for future map projections,
- `tigramite` for the confirmatory PCMCI experiment reported in the manuscript.

## Data Location

The default config expects data here:

```text
/home/vipuser/Data
```

The code does not download any dataset. It recursively scans the local directory and tries to infer usable atmospheric variables from file contents and filenames.

## Quick Start

Run the full pipeline:

```bash
python scripts/run_all.py --config configs/default.yaml
```

Run individual steps:

```bash
python scripts/inspect_data.py --data_root /home/vipuser/Data
python scripts/run_preprocess.py --config configs/default.yaml
python scripts/run_causal_discovery.py --config configs/default.yaml
python scripts/run_regime_analysis.py --config configs/default.yaml
```

## Expected Outputs

Main outputs are written under `outputs/`:

- `outputs/data_summary.json`: scanned files, dimensions, variables, and time ranges.
- `outputs/processed/region_series.npz`: region-level `[time, region, variable]` series.
- `outputs/regions/region_metadata.csv`: region centroids and node counts.
- `outputs/causal_edges/candidate_edges.csv`: candidate directed edges tested.
- `outputs/causal_edges/all_edges.csv`: all tested lagged edges.
- `outputs/causal_edges/significant_edges.csv`: FDR-significant edges.
- `outputs/reports/summary_metrics.csv`: per-regime summary metrics.
- `outputs/reports/regime_comparison.csv`: pairwise graph Jaccard distances.
- `outputs/reports/causal_chain_summary.csv`: Wind -> Humidity -> Cloud chain counts.
- `outputs/figures/`: causal maps, heatmap, lag histogram, network plot, and chain figure.

## Causal Edge Table Columns

Important columns in `all_edges.csv` and `significant_edges.csv`:

- `source_region`, `target_region`: directed region pair.
- `source_var`, `target_var`: directed variable pair.
- `edge_type`: readable label such as `wind_to_humidity`.
- `lag`: tested temporal lag.
- `p_value`: nested OLS F-test p-value.
- `q_value`: FDR-adjusted p-value within each regime and target variable.
- `significant`: true when `q_value < alpha`.
- `F_statistic`: nested model F statistic.
- `effect_coefficient`: coefficient for the source lag in the full model.
- `effect_abs`: absolute coefficient magnitude.
- `effect_score`: signed score combining coefficient sign, effect size, and `-log10(q_value)`.
- `regime`: regime mask used for the test.
- `n_samples`: effective samples after lagging, masking, and finite-value filtering.
- `distance_km`: source-target centroid distance.
- `method`: causal discovery backend, default `granger_ols`.

## Interpretation Notes

This is observational causal discovery. A significant edge means a lagged source series improves prediction of a target series under the configured controls and regime mask. It should be described as a **lagged directed dependency**, **lagged causal dependency**, or **causal-discovery-inspired dependency**.

It should not be described as proof that intervening on one atmospheric variable would necessarily change another. Confounding, measurement choices, spatial aggregation, temporal resolution, and regime definitions all matter.

## Suggested Paper Framing

Useful framing phrases:

- regime-aware causal discovery,
- moisture transport pathway,
- Wind -> Humidity -> Cloud causal chain,
- dynamic atmospheric causal graph,
- regime-dependent lagged atmospheric dependencies.

## Configuration

Edit `configs/default.yaml` to control:

- local data root and subsampling,
- preprocessing options,
- number and type of spatial regions,
- regime quantiles,
- maximum lag and FDR alpha,
- bootstrap stability,
- visualization edge limits.

For a fast smoke test on a very large dataset, set:

```yaml
data:
  max_time_steps: 500
  spatial_subsample: 4
causal:
  max_lag: 6
```

Then rerun the pipeline.
