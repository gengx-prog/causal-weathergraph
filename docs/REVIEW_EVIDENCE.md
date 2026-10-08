# Review topics, code and evidence

Use the [original comments and current response](REPRODUCING.md#review-the-paper-and-original-comments)
for the exact wording. This table is a navigation aid, not a claim that every
scientific criticism has been resolved. The release's experiment evidence
archive preserves recorded output and design files; availability of an archive
does not mean every historical analysis was rerun in the publication check.

| Comments / topic | Implementation | Inspectable results |
| --- | --- | --- |
| R1-M1–M7, R2-1, R3-3: serial dependence, equations, inference | `revision/inference.py`, `full_var_tests.py`, `run_simulations.py` | Supplement S1, S2, S4; release `inference/`, `simulations/`, `simulations_post_diagnostic/` |
| R1-M6, R2-5: held-period evaluation | `revision/run_holdout.py`, `run_three_stage.py` | Release `holdout_aligned/`, `three_stage/`; rerun `scripts/reproduce.py --experiment core` |
| R1-M2/M3, R2-2, R3-7: candidate and topology nulls | `revision/graph_nulls.py`, `run_graph_nulls.py`, `run_graph_overlap_null_round2.py` | Release `graph_nulls/`, `graph_overlap_null_round2/`; supplement graph/null tables |
| R1-D5, R2-3, R3-4: regime/endogenous selection and latitude | `run_regime_contrasts.py`, `run_hemisphere_contrasts.py`, `run_latitude_contrasts.py`, `run_external_nino_regimes.py` | Release corresponding contrast and external-Niño directories; supplement regime tables |
| R2-5, R3-1/R3-2: known-structure comparisons | `run_simulations.py`, `run_rpcmci_benchmark.py`, `run_castle_benchmark.py` | Release `simulations/`, `rpcmci/`, `castle_response_aligned/`; each manifest distinguishes settings and adapters |
| R1-I2/I3, R1-T3, R3-6: data route and cloud-product sensitivity | `prepare_inputs.py`, `run_ceres_cloud_substitution.py`, `run_source_overlap_comparison.py` | Release input manifests, source-route diagnostics and CERES tables; rerun `--experiment ceres` |
| R2-4, R3-5: vector/physical interpretation and path limitations | `run_path_diagnostics.py`, `make_whec_index.py`, `run_whec_test.py` | Supplement S6 and WHEC design; release `path_diagnostics/`, `whec_index/`, `whec_test/`; rerun `--experiment whec` from archived index |
| R1-P10, R1-T5: complete tables and computational documentation | `scripts/verify_artifacts.py`, `download_artifacts.py`, `reproduce.py` | Supplement ZIP, [release file inventory](../artifacts/release-manifest.json), [environment](ENVIRONMENT.md), [validation](../validation/README.md) |
| ED-1–ED-6 and presentation changes | Revised manuscript sources and final response | Clean/marked PDFs, supplementary PDF, highlights, LaTeX, current response |

Core numerical reruns compare complete output tables, not selected favorable
edges. Negative controls and unsuccessful hypotheses remain in the reference
tables. WHEC uses the existing frozen design and preserves its verdict; the
software publication does not change that scientific decision.
