# Revision experiment reproduction

> Historical research-workspace log. For a new clone, follow
> [the portable reproduction guide](../docs/REPRODUCING.md), including the
> tested independent environment and versioned data downloads. Absolute paths
> and earlier execution statuses below are retained for provenance.

## Controlled intervention truth benchmark

`revision/intervention_benchmark/README.md` describes the frozen lightweight
simulator, eight effect-estimation adapters and reproduction command. The
canonical outputs are in `../revision_outputs/intervention_benchmark_v1/`.
Use a new output directory and copy the original design before running
`run.py --output <new-directory>`; recorded runs cannot be overwritten.
The independent audit distinguishes all-metric verification from sampled
trajectory reconstruction and eight independently retrained boosting models.
All reported causal effects are internal to the specified simulator and wind
policy; they do not establish observational atmospheric causality or mediation.

## Bounded physical transport consistency

The completed two-region experiment and exact reproduction commands are in
`revision/transport_consistency/README.md`. Its frozen design and all inputs,
predictions, paired comparisons, original-CERES checks, independent numerical
audit, and supplementary reports are in
`../revision_outputs/transport_consistency_v1/`. Use a new output directory for
reproduction. The audit requires input, model and external stages to exist and
independently refits all four ridge penalties and eight boosting fits. Numerical
consistency does not establish causal effects or full moisture-budget closure.

The new global CERES and same-date ERA5 overlap downloads have completed;
their latest on-disk status files and refreshed `revision/download_receipt.json`
take precedence over historical queue snapshots. Their broader scientific
comparisons remain separate from this two-region transport experiment.

## Local wind/humidity cloud estimator

The estimator in `cloud_simulator/` uses existing physical-unit inputs and a separate output directory. Launch the finished local interface with:

```powershell
& '../.venv/Scripts/python.exe' 'revision/cloud_simulator/launch.py'
```

Or double-click `revision/cloud_simulator/Open_Cloud_Simulator.cmd`. The launch helper opens Chrome when available and starts a hidden Python process listening only on `127.0.0.1:8765`. The inference API loads trusted locally generated model files and checks the model hash. It does not fetch new meteorological data.

To reproduce training, use a **new** output directory; the recorded production directory cannot be silently overwritten:

```powershell
& '../.venv/Scripts/python.exe' 'revision/cloud_simulator/engine.py' --output '../revision_outputs/cloud_simulator_reproduction' --horizons 0
& '../.venv/Scripts/python.exe' -m pytest tests/test_cloud_simulator.py -q
```

The production run trained six fixed estimators in about 8.7 seconds on this machine. Frozen training settings and exact feature/target/index fingerprints are in `../revision_outputs/cloud_simulator_v1/protocol.json`; the exact training engine is preserved beside its models. Tests verify the recorded production artifacts when present. Use `app.py --artifacts <new_directory> --port 8766` to inspect a reproduction independently. Only horizon 0 was trained and validated for the first interface; selecting a future horizon through the Python CLI is a new experiment. This is contemporaneous conditional regression, not the earlier token forecast task or an atmospheric intervention simulator.

Run modules from the worktree root, using `../.venv/Scripts/python.exe`. Python 3.13.7 and the full installed package versions are recorded in `environment.freeze.txt`; the environment inherits system packages. Essential analysis dependencies are NumPy, SciPy, pandas, xarray, netCDF4, statsmodels, Tigramite, OR-Tools, psutil, threadpoolctl and matplotlib. Editorial checks additionally use bibtexparser 2 and pylatexenc. Tests use pytest.

The original repository commit is `e4c7bac`. The revision branch is `major-revision-20260929`; manuscript inputs are archived in `manuscript/baseline`. The original manuscript, raw data and original repository outputs were not edited. The main experiment protocol is `protocol.json`; later statistical diagnostics have explicitly labelled protocol amendments. These are retrospective revision protocols, not original-study preregistration.

## Inputs and execution

Use new output directories when rerunning. In particular, the E2 cache identity does not include estimator code hashes, so never reuse its previous output directory after changing estimator code. Most examples below show the recorded output names to make provenance easy to follow; substitute an entirely new output root to reproduce without overwriting evidence. Run each line as a separate command and check its exit status.

```powershell
$revisionPython = 'D:\Paper2\Major Revision\.venv\Scripts\python.exe'
$revisionOutput = 'D:\Paper2\Major Revision\revision_outputs'
$legacyOutput = 'D:\Paper2\vipuser\论文2026_516\Causal\causal-weathergraph\outputs'
$env:OPENBLAS_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'

& $revisionPython -m revision.prepare_inputs --data-root 'D:\Paper2\vipuser\Data' --legacy "$legacyOutput\processed\region_series.npz" --output "$revisionOutput\inputs"
& $revisionPython -m revision.run_inference --input "$legacyOutput\processed\region_series.npz" --legacy-edges "$legacyOutput\causal_edges\all_edges.csv" --output "$revisionOutput\inference" --bootstrap-replicates 200
& $revisionPython -m revision.run_cluster_inference --input "$legacyOutput\processed\region_series.npz" --output "$revisionOutput\inference_exploratory_cluster"
& $revisionPython -m revision.run_graph_nulls --input "$legacyOutput\processed\region_series.npz" --output "$revisionOutput\graph_nulls" --candidate-replicates 100 --topology-replicates 200
& $revisionPython -m revision.run_holdout --input "$revisionOutput\inputs\region_trainfit.npz" --output "$revisionOutput\holdout_aligned" --pcmci
& $revisionPython -m revision.run_regime_contrasts --input "$revisionOutput\inputs\region_trainfit.npz" --output "$revisionOutput\regime_contrasts"
& $revisionPython -m revision.run_hemisphere_contrasts --input "$revisionOutput\inputs\region_trainfit.npz" --output "$revisionOutput\hemisphere_contrasts"
& $revisionPython -m revision.run_path_diagnostics --input "$revisionOutput\inputs\region_trainfit_vectors.npz" --discovery "$revisionOutput\holdout_aligned\discovery_edges.csv" --output "$revisionOutput\path_diagnostics"
& $revisionPython -m revision.run_full_var --input "$revisionOutput\inputs\region_trainfit.npz" --output "$revisionOutput\full_var" --history-lags 3 --source-lags 3
& $revisionPython -m revision.run_full_var --input "$revisionOutput\inputs\region_trainfit.npz" --output "$revisionOutput\full_var_lag12" --history-lags 12 --source-lags 3 --start-lag 12
& $revisionPython -m revision.run_full_var --input "$revisionOutput\inputs\region_trainfit.npz" --output "$revisionOutput\full_var_lag3_aligned12" --history-lags 3 --source-lags 3 --start-lag 12
& $revisionPython -m revision.compare_full_var_histories
& $revisionPython -m revision.run_simulations --output "$revisionOutput\simulations" --stage all --null-repetitions 200 --graph-repetitions 50
& $revisionPython -m revision.run_castle_benchmark --source 'D:\Paper2\Major Revision\third_party\CaStLe' --output "$revisionOutput\castle_response_aligned" --repetitions 25 --length 256
& $revisionPython -m revision.run_rpcmci_benchmark --output "$revisionOutput\rpcmci" --replicates 20 --n-samples 2048 --iterations 10 --anneals 3 --workers 4
& $revisionPython -m revision.data_shift_diagnostics --inputs "$revisionOutput\inputs" --output "$revisionOutput\data_shift_diagnostics"
& $revisionPython -m revision.make_revision_figures --root $revisionOutput
& $revisionPython -m pytest -q
```

`run_full_var` currently reads its comparison input from the sibling `holdout/holdout_all_candidates.csv`. For a clean rerun, generate the corrected holdout in that directory, or copy the corrected output there before the VAR commands. The original execution used the archived first holdout's unchanged Granger/forecast columns. `compare_full_var_histories` currently resolves the recorded `../revision_outputs` root internally; change that root explicitly if reproducing elsewhere. Dense VAR runners fix BLAS to four threads internally, whereas most other runners use two.

The CaStLe runner expects the repository root containing `src/stencil_functions.py`, not the `src` directory or a pip package. Its manifest records the official commit, function extraction, exact local dependency adapter and common scoring family. Keep the MIT license with the source. The original function bodies and concatenation behavior are preserved; adapter details must be described if used in a paper.

On 2026-09-30, the CaStLe comparison's local OLS response window was corrected to match local PCMCI's 254 responses at T=256. The command above now names the aligned output; the older `castle/` files and executing source snapshot remain historical evidence. Reproducing that earlier version requires its archived runner. Official CaStLe's pooled responses and cell-boundary seams remain different from local sampling. `castle_response_aligned/comparison_report.json` records unchanged official CaStLe/local PCMCI metrics and all changed local OLS means. Existing figures have not automatically been regenerated from the new output.

The hemisphere extension is a post-diagnostic retrospective contrast on 425 mirrored candidate pairs, with same-calendar covariance retained before HAC estimation. It reports signed local-winter-minus-summer slope differences, not causal effects or frozen predictive holdout performance. The output `independent_audit.json` records the separate numeric review; it is not generated automatically by the analysis command. Inference calibration and continuous/bandwise latitude interactions remain unresolved.

The larger-N null simulations and block-cluster extension were added after the first diagnostic, with exact commands in `../revision_outputs/simulations_post_diagnostic/reproduction_commands.json`. They must not be relabelled as the original frozen primary suite. Each simulation output keeps its original executing source snapshot; future code versions may reject resume because hashes differ, as intended.

The RPCMCI diagnostic audit script is archived in `../revision_outputs/rpcmci/audit_postprocessing.py` and expects to run from that output location after all 20 fits. It preserves the original JSONL, verifies model-output fingerprints and refreshes summaries only. The audited off-by-one diagnostic boundary changed zero actual result rows.

## Version and evidence distinctions

- `holdout_aligned` supersedes the initial PCMCI response-window implementation. All original non-PCMCI values were checked exactly equal; the new response prefix uses only already-observed history. Both ΔR² denominator conventions are exported.
- Full VAR history sensitivity must compare `full_var_lag3_aligned12` against `full_var_lag12`, not the original start-lag-3 fit.
- E1/E2 use legacy full-period climatology for reproduction and retrospective sensitivity. E3/E4/E5 use training-only transformations. Never describe all experiments as untouched holdout validation.
- E5 geography calipers were added after fitting without changing chosen paths or model fits. The design records this amendment. Unmatched paths remain unmatched.
- Primary, smoke and pilot output directories are separate. Smoke/pilot results are excluded from the final evidence index and formal replicate counts.
- Elapsed times and memory measurements have different scopes across runners; they are execution records, not a fair algorithm speed ranking. E2/E5 do not have a recorded peak-memory measure.
- The original `src/causal_weathergraph/causal/bootstrap.py` is unchanged. Revision analyses use the tested calendar-design-row bootstrap in `revision/inference.py`.

`finalize_evidence.py` verifies exact non-PCMCI equality between holdout versions, complete PCMCI response counts, matching execution hashes for root-owned source archives, and all planned RPCMCI repetitions. It writes `validation/artifact_index.csv` with SHA-256 for non-smoke tables, diagnostics, source snapshots and figures. Raw data and large binary model files are traced through individual manifests rather than copied into Git. Independent numerical and provenance checks are recorded separately in `validation/reviewer2_independent_audit.json`.

To inspect every tracked change, run `git diff e4c7bac -- revision tests manuscript`. This includes the archived manuscript baseline, experiment code, protocols, tests and this report; it is not a highlighted revised manuscript. Final English proofreading, literature relevance review, first-citation numbering, the marked manuscript and formal response letter remain separate manuscript work.

## September 30: downloaded controls and external subsets

The completed 144-file CDS extension has been checked and conservatively remapped to the earlier 64-by-32 grid. Its frozen plan, per-source full-value/SHA checks, area-integral checks and independent derived-data audit are in `../supplementary_data/era5_controls/cds_extension/regridded/`. `regrid_era5_cds_controls.py --run` uses the recorded fixed workspace paths and a resumable plan; it is not a new download command. To rebuild that preprocessing from scratch without altering receipts, use an isolated copy of the workflow with distinct output paths. The original source files remain preserved.

The commands below use new output directories, reuse the downloaded sources and do not initiate network transfers. The standardization parameters use 1979–2018 only. Execute sequentially from the repository root, retaining the variables defined above.

```powershell
& $revisionPython -m revision.prepare_physical_controls --output "$revisionOutput\physical_controls_inputs_reproduction"
& $revisionPython -m revision.run_physical_controls --physical "$revisionOutput\physical_controls_inputs_reproduction\region_controls_trainfit.npz" --output "$revisionOutput\physical_controls_experiments_reproduction"
& $revisionPython -m revision.run_ceres_pilot_comparison --freeze --output "$revisionOutput\ceres_pilot_comparison_reproduction"
& $revisionPython -m revision.run_ceres_pilot_comparison --run --output "$revisionOutput\ceres_pilot_comparison_reproduction"
& $revisionPython -m revision.run_external_nino_regimes --freeze --output "$revisionOutput\external_nino_regimes_reproduction"
& $revisionPython -m revision.run_external_nino_regimes --run --output "$revisionOutput\external_nino_regimes_reproduction"
& $revisionPython -m pytest -q tests/test_physical_controls.py
```

The recorded original outputs are `physical_controls_inputs/`, `physical_controls_experiments/`, `ceres_pilot_comparison/` and `external_nino_regimes/`. The physical experiment retains the original 36 pairs, both directions and six signed lags; it separately fits each source lag. Its 60 paths retain temporal mediator alignment. The pair model includes own history alone or six additional fields in two predeclared history windows; the path augmentation precedes its earliest wind input. Future-source models are reverse-time diagnostics. Evaluation coefficient refits are distinct from training-frozen losses. Source-segment summaries remove rows whose inputs straddle the source change. All results are retrospective sensitivity analyses on a fixed subset, not whole-graph causal/FDR validation.

The Niño experiment is separate from the six-field conditioning experiment: it uses target history and state interactions, not the new physical controls jointly. Its 1,296 rows include three HAC bandwidths for the same 432 period/contrast estimates. `clarify_nino_population_counts.py` was applied once after the original execution to distinguish a requested but unavailable antecedent month from observed native months; original counts and the amendment are preserved. This reporting helper has a fixed `OUT` path and intentionally refuses a second application. For a new output directory, copy the helper, update `OUT` explicitly, and run that copy once. No model coefficient changes.

Independent audits are saved as `physical_controls_inputs/validation_saved_sources_independent.json`, `physical_controls_experiments/independent_audit.json` and `ceres_pilot_comparison/independent_validation.json`. The Niño runner exports independent grouped-OLS and direct-Bartlett checks. `audit_physical_controls.py` and the saved independent audit scripts currently address the recorded output roots; adjust a copy's paths before auditing reproduction directories. Hash, row-count and arithmetic checks cover all tables; independent numerical refits are sampled. All physical designs are full rank but have condition numbers up to approximately 6.3e5; passing QR/SVD checks does not remove this near-collinearity limitation.

`summarize_new_data_experiments.py` reads the recorded result directories and creates `../新增数据处理与补充实验结果_20260930.md`, its Word counterpart, and `revision/NEW_DATA_EXPERIMENTS_ZH.md`. It requires the physical-model independent audit to have passed. Run `& $revisionPython -m revision.summarize_new_data_experiments` to regenerate that report; this refreshes the report files, not scientific outputs. CERES covers January 2019 and two fixed Atlantic regions, using two declared daily sampling comparisons; exact timestamp equivalence and long-term validation are not claimed.

## September 30 local time / October 1 UTC: reviewer-directed second round

The reviewer ledger in `../revision_outputs/reviewer_experiment_round2/reviewer_coverage_final.json` maps 56 reviewer subitems and six editor requirements to completed experiments, existing evidence and remaining work. The 44 Reviewer 1 subitems are an internal decomposition of the supplied document, not 44 numbered questions in the attachment. `coverage_independent_audit.json` checks the original DOCX/PDF quotations. No new downloads are required for this batch.

```powershell
& $revisionPython -m revision.run_latitude_contrasts --freeze --run --output "$revisionOutput\latitude_contrasts_reproduction"
& $revisionPython -m revision.run_external_nino_physical_joint --freeze --output "$revisionOutput\external_nino_physical_joint_reproduction"
& $revisionPython -m revision.run_external_nino_physical_joint --run --output "$revisionOutput\external_nino_physical_joint_reproduction"
& $revisionPython -m revision.run_lag_window_sensitivity --output "$revisionOutput\lag_window_sensitivity_reproduction"
& $revisionPython -m revision.run_long_lag_benchmark --repetitions 100 --output "$revisionOutput\long_lag_benchmark_reproduction"
```

`run_graph_overlap_null_round2.py` has a fixed `OUT` constant and refuses to overwrite a frozen directory. To reproduce, copy the script within `revision/` under a new filename, change that copy's `OUT` to a new directory, then run that copy from the repository root. This records its new code hash; preserve the executing original snapshot. Do not delete the original directory to bypass its refusal. The fixed plan uses four chains, 200 stored states per chain, burn-in of 200 attempts per selected edge, and gaps of 20 attempts per edge, for each of two graph families. Rejected and label-only moves consume attempts. Exact original-support enumeration is separate from expanded-support MCMC. The latter did not establish sufficient mixing or state-space connectivity and must not be used as a calibrated enrichment test.

Latitude summaries cover only three absolute target-latitude rings; aggregate influence scores retain same-calendar dependence across edges and hemispheres. Niño joint controls precede tested source lags and allow every state its own regression coefficients; all three HAC bandwidths are retained. The source-window experiment compares 3/6/12 jointly entered source lags under three control schemes on the same real-data response start at `t=15`; the synthetic long-lag companion uses `t=12`. A benchmark window that excludes the true lag is explicitly outside recovery scope, not a measured failure probability. All coefficient and loss uncertainty for the observed data remains exploratory.

The recorded output directories retain independent audit scripts, input/output hashes and executing sources. Some independent scripts use recorded absolute paths; adjust a copy explicitly before auditing reproduction directories. `make_round2_figures.py` renders source-backed standalone scientific PNG/PDF figures, and `summarize_reviewer_round2.py` builds the Chinese Word/Markdown report and 62-item ledger from the recorded results and the reviewed graph interpretation file. The report files are `../逐条审稿意见追加实验与证据_第二轮_20260930.*`. Export the generated Word document using `revision/export_response_pdf.ps1` if a refreshed PDF is needed. These commands refresh presentation artifacts, not scientific outputs.

## October 1: physical-state token forecasting pilot

The new `weather_tokens/README.md` specifies the separate forecasting question and limitations. No download or pretrained language-model weights are required. Run scripts directly so their local data-module imports resolve. The current environment has PyTorch 2.10.0+cu128 and supports the local GPU; execution records exact versions and input hashes.

```powershell
& $revisionPython revision/weather_tokens/train.py --smoke --cases pooled --seeds 17 --output "$revisionOutput\weather_token_pilot_reproduction_smoke"
& $revisionPython revision/weather_tokens/train.py --output "$revisionOutput\weather_token_pilot_reproduction"
& $revisionPython revision/weather_tokens/linear.py --output "$revisionOutput\weather_token_pilot_reproduction"
& $revisionPython revision/weather_tokens/train.py --output "$revisionOutput\weather_token_pilot_reproduction" --summarize-only
```

The final summarizer currently has a fixed `OUT` constant pointing to the recorded pilot. To regenerate a separate reproduction report, copy `summarize.py` and adjust that constant, preserving the original sources and evidence. Check `audit.py --help` and `render_examples.py --help` for output-directory options before auditing or rendering a reproduction. Formal runs refuse to reuse model artifacts whose code, protocol, source hashes or smoke status differ. Initial smoke and deterministic-smoke directories are retained separately and never used in the final scientific tables.

This pilot refits all statistics from the `physical` arrays; the old normalized `data` arrays cannot be exactly inverted after spatial aggregation and contain validation-period fit information for this new split. Comparisons retain the original 12-region support, all three seeds and failed transfer baselines. Do not select a new encoder on the already inspected evaluation period and call the result an untouched confirmation.

## October 1: bounded annual CERES comparison and response refresh

The accepted extension orders are 41742/suborder 99005 (north Atlantic) and 41743/suborder 99006 (south Atlantic), February–December 2019. January stays in its original pilot directory. The new raw-data location is `../supplementary_data/cloud_validation/ceres_syn1deg_ed42_2019_extension/`; `request_plan.json` is immutable, `submission_receipt.json` records acceptance, and `download_status.json` reports the live worker state. No full archive or multiyear download is authorized by this extension. Its raw-data ceiling is 250,000,000 bytes, with an approximately 80 MB provider estimate.

`watch_ceres_annual_download.py --poll-seconds 300 --max-hours 24 --run-analysis` receives the previously authorized email and CERES-domain session cookies once through a short-lived loopback-only endpoint, then keeps them in memory. The browser context must include cookies for the `/ord-tool/order` path; root-path cookie queries omit the provider's relevant cookies. Do not print or persist this session context, or embed it in shell arguments. A restarted worker needs fresh in-memory context. It resumes completed files only when the download manifest, hashes and semantic checks agree; unknown or partial files require inspection rather than automatic overwriting. Provider queue times are outside the worker's control. Scientific `.nc` files and `.nc.part` files share the cap; no unrelated directories are scanned or deleted.

The annual comparison's original `comparison_plan.json` and later `execution_addendum_frozen.json` are both preserved in `../revision_outputs/ceres_annual_comparison_2019/`. The addendum states exactly when final execution hashes were fixed; neither record is described as a pristine-year preregistration. The downloader automatically runs the following after both fixed-region files pass acquisition checks:

```powershell
& $revisionPython revision/run_ceres_annual_comparison.py --run
```

Do not invoke `--freeze` or `--freeze-execution` over the existing records, edit frozen source/evidence files, or reuse a completed result directory. For a fresh reproduction, use a separate output directory, explicitly freeze its plan and execution addendum, and retain the same source manifests. Both daily sampling operators and every month/season must remain; DJF means January, February and December 2019, not one continuous winter. Product discrepancies are not errors against independent truth; monthly demeaning is descriptive within-year centering. The worker's `analysis_completed_pending_independent_audit` state deliberately requires subsequent output review.

The versioned response package can be rebuilt without overwriting the older delivery:

```powershell
& $revisionPython -m revision.build_response_document --output '..\审稿答复文档\2026-10-01_evidence_update'
& .\revision\export_response_pdf.ps1 -DocumentPath 'D:\Paper2\Major Revision\审稿答复文档\2026-10-01_evidence_update\审稿意见逐条答复与技术证据说明.docx' -PdfPath 'D:\Paper2\Major Revision\审稿答复文档\2026-10-01_evidence_update\审稿意见逐条答复与技术证据说明.pdf'
& $revisionPython -m revision.validate_response_document --output '..\审稿答复文档\2026-10-01_evidence_update'
```

These commands refresh the assembled response and presentation artifacts, not experiment outputs or the manuscript body. The checked delivery contains 62 response units and 87 PDF pages; all quotations match their supplied originals and 338 explicit evidence paths exist. Source changes and file hashes accompany the package, while future annual CERES results remain pending.

## October 1: explicitly authorized larger acquisition

The previous paragraph is the earlier delivery snapshot. The two-region annual comparison has since completed and passed `../revision_outputs/ceres_annual_comparison_2019/independent_audit.json`. Re-run its saved independent script to check the recorded products; do not rerun the production analysis over those completed outputs.

The new global CERES directory is `../supplementary_data/cloud_validation/ceres_syn1deg_ed42_global_2001_2025/`. Its immutable `request_plan.json` selects only total cloud, daily 2001–2025 and hourly 2017–2025, as global one-degree regional-grid cells. `submission_receipt.json` maps every job to the accepted order/suborder and actual download directory. `provider_metadata_audit.json` verifies all 23 metadata records; `provider_probe.json` records the real authenticated-session status check. These are acquisition checks, not full scientific validation. The user explicitly expanded the earlier pilot scope; the original pilot directory is preserved.

```powershell
& $revisionPython revision/watch_ceres_global_download.py --probe
& $revisionPython revision/watch_ceres_global_download.py --poll-seconds 300 --max-hours 48
```

Each invocation receives the existing authorized provider context once on loopback port 8767 (`/challenge`, then `/provider-context`); it must stay in memory and must not appear in command-line arguments, source, receipts or logs. Use only cookies applicable to the CERES `/ord-tool/order` path. The formal worker requires a successful probe for the same plan/receipt, a single-process lock, exact order-directory links and 25 GB/1.3 GB limits. It retains interrupted partials for review rather than silently restarting large transfers. `download_status.json` and `download_manifest.json` are authoritative for actual received data. This worker does not run the earlier two-region analysis on the global files. The next scientific stage is defined separately in `../revision_outputs/ceres_global_cloud_plan/methods_plan.json`; periodic longitude handling and explicit product seams require their own verified implementation.

The separate ERA5 package is `../supplementary_data/era5_source_overlap_2022_2023/`, using the already configured CDS client and accepted dataset terms:

```powershell
& $revisionPython revision/download_era5_source_overlap.py --run --poll-seconds 30 --max-hours 24
```

Its existing frozen plan covers 30 requests and 152 unique six-hour times. Do not run `--freeze` over the recorded plan, change its code snapshot, or start another process while `worker.lock` points to a live worker. A later restart reuses recorded provider request IDs and exact-range partials. The 4 GB cap counts files, partials and records. Completed-file checks cover all time/grid/level/unit metadata plus first/last planes; full-value scientific audit, conservative remapping and source comparison remain separate tasks. Provider result URLs and API credentials are never written to receipts.

The global directory also contains `worker_implementation_checks.py/json`: ten executed check groups use synthetic in-memory arrays and the sanitized real provider table, without accessing private context or changing the live worker. Run that check script directly with the revision Python to repeat the recorded implementation audit. The updated response delivery is `../审稿答复文档/2026-10-01_global_expansion/`; apply the earlier build/export/validate commands with this output directory when intentionally refreshing it. Its 62 response units and 88 PDF pages distinguish completed 2019 results from pending expanded analyses. The source diff and artifact hashes accompany the package.
