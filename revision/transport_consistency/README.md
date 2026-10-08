# Bounded horizontal moisture transport experiment

The completed experiment uses two fixed Atlantic footprints, six-hour ERA5
fields over 1979–2025, and a separate 2019 CERES cross-product check. It tests
local six-hour humidity change and cloud prediction. It does not validate the
previous cross-region graph paths or identify intervention effects.

All evidence is in `../../../revision_outputs/transport_consistency_v1/`
relative to this directory. The two same-information humidity comparisons
reduce equal-region evaluation MSE by 6.35% (ridge) and 2.79% (histogram
gradient boosting). Cloud changes are small and inconsistent. The unfitted
horizontal-budget proxy performs worse than persistence. Full regional,
seasonal, source-segment and unfavorable results are retained.

## Design and evidence

- `design.json`: frozen before new transport values or model results; retrospective data had already been examined.
- `inputs/input_manifest.json`: 15 source hashes, unit/calendar checks, resolved-grid finite-volume operator and area/covariance checks.
- `inputs/input_data.npz`: 68,668 six-hour times, 2 regions, 36 core cells and 60 core-plus-cardinal-halo inputs per region.
- `model/run_manifest.json`: 40 fitted models, configuration and result hashes.
- `model/sample_ledger.csv`: all daily candidate origins and split/source exclusion reasons.
- `model/predictions.csv.gz`: implementable forecasts and spatial mismatch sensitivity; future information is excluded.
- `model/diagnostic_predictions.csv.gz`: deliberately unavailable target-time flux. For humidity it contains future outcome information; it is not a causal negative control.
- `model/metrics.csv`, `model/paired_comparisons.csv`: all outcomes and paired, source-stratified 30-day-block intervals.
- `external/`: all 2019 cloud comparisons, with 06:00 ERA5 and nominal 06:30 CERES supports distinguished.
- `independent_audit.json`: independent grid/boundary, fitting, parameter selection, prediction, uncertainty and original-CERES recomputation.
- Chinese DOCX/PDF/Markdown and English supplementary PDF: methods, results, reviewer-response language and limits.

The derivation has a numerical horizontal flux closure. It does not close the
physical moisture budget, recover subgrid turbulence, or establish native-grid
above-ground pressure-level validity. Models use raw-unit physical controls,
with any scaling fitted on 1979–2014 only. Different source remapping routes
and previously examined evaluation years remain explicit limitations.

## Reproduce without overwriting the recorded run

From the repository root, create an empty sibling output directory and copy
the existing frozen design there without changing its contents. Then run:

```powershell
& '../.venv/Scripts/python.exe' 'revision/transport_consistency/features.py' --execute --plan '../revision_outputs/transport_consistency_reproduction/design.json' --output '../revision_outputs/transport_consistency_reproduction/inputs'
& '../.venv/Scripts/python.exe' 'revision/transport_consistency/experiment.py' --input '../revision_outputs/transport_consistency_reproduction/inputs/input_data.npz' --design '../revision_outputs/transport_consistency_reproduction/design.json' --output '../revision_outputs/transport_consistency_reproduction/model'
& '../.venv/Scripts/python.exe' 'revision/transport_consistency/external_check.py' --root '../revision_outputs/transport_consistency_reproduction'
& '../.venv/Scripts/python.exe' 'revision/transport_consistency/audit.py' --output '../revision_outputs/transport_consistency_reproduction'
```

The raw ERA5 files and already audited original CERES deliveries are local
prerequisites; the experiment does not acquire new data. Model production uses
two threads. The full audit refits the eight boosting models and independently
solves all ridge penalties; it requires model and external outputs and fails
if a required stage is absent. Independent calculations do not imply
independent scientific observations or calibrated causal inference.

`build_report.py` regenerates the report for the canonical recorded output
only, after the full audit passes. Its DOCX is exported with Word to PDF and
checked separately for content and layout. Source diffs and Git preserve the
added code and manuscript modules. The main manuscript and full bibliography
are not silently replaced by the supplementary experiment.
