# Controlled intervention benchmark

This supplement tests specified total wind-policy effects inside an idealized
moist-tracer simulator. It does not identify observational ERA5 effects, validate
the original discovered graph, estimate natural mediation, or implement a
Navier--Stokes or operational cloud model.

The frozen design is in
`../../../revision_outputs/intervention_benchmark_v1/design.json` relative to
this folder. Production uses 48 periodic cells, conservative upwind transport,
externally forced temperature, saturation phase exchange and an explicit water
source/rain ledger. A=1 versus A=0 means +2 versus -2 m/s during the first hour,
with paired initial states and identical exogenous noise. Model outputs are
dry-air vapor mixing ratio and a condensate-derived cloud proxy.

Four scenarios distinguish randomization, assignment confounding, prescribed
condensate feedback, and an exact structural null. Independent replications have
2,048 training, 512 unused validation and 512 test episodes. Scenarios share
exogenous draws within a replication. The directly observed common driver Z
is sufficient for adjustment by construction; local covariates still contain
proxies for Z. Full-state comparisons change available information.

Eight estimators include explicit endpoint adapters of the existing OLS/FWL
routine, separately named HGB g-computation, and known-propensity oracle IPW
and AIPW. Training only sees assigned factual outcomes. Paired potential
outcomes are used only on the held-out test set for effect evaluation. The
original significance-weighted ranking score is not an intervention effect.

## Reproduction

From the repository root, create a new empty output directory and copy the
recorded `design.json` there without changing it. Then run:

```powershell
& '../.venv/Scripts/python.exe' 'revision/intervention_benchmark/run.py' --output '../revision_outputs/intervention_benchmark_reproduction'
```

The program refuses to overwrite a recorded run. It uses two CPU threads, no
downloaded meteorological files, and no fitted observational parameters. All
generator assumptions are explicit in `design.json` and `simulation.py`.
The separate `numerical_checks.py` recreates the canonical numerical-check
artifacts: finite-volume invariants, analytic advection and the prespecified
48/96-cell, 300/150-second effect sensitivity. It writes to the canonical
output directory, so preserve that directory before rerunning these checks.

## Evidence and interpretation

- `data/<scenario>/rep_XXX/episodes.npz`: initial covariates, assignments,
  known propensities, factual outcomes and paired outcomes.
- `trajectories.npz`: eight complete paired trajectories per case;
  `numerics.json`: invariants and exact-null full-state hashes.
- `models/<scenario>/rep_XXX/`: saved fits, predictions, per-fit metrics and hashes.
- `all_metrics.csv`, `summary.csv`: all four scenarios, two outcomes, two horizons
  and eight methods. No outcome-dependent exclusions or hyperparameter tuning.
- `resolution_sensitivity.csv`: every prespecified numerical sensitivity result.
- `independent_audit.json`: exact scope and status of independent recomputation.
- Local Chinese Word/PDF and English supplementary PDF: scientific explanation,
  results, adverse findings and response language, separate from the main paper.

HC3 intervals concern linear projection coefficients. Oracle score intervals
target population effects under IID episodes and known assignment probabilities.
They are not confidence intervals for a noise-free finite test-bank truth.
HGB g-computation has no per-fit interval. Twenty-replication Monte Carlo
summaries and Wilson intervals do not establish 5% calibration or graph-wide
FDR. Episode-twin prediction errors are not conditional-average-effect risk.
The model has no latent heating, vertical motion, momentum equation or energy
closure; its cloud proxy is not satellite cloud truth.
