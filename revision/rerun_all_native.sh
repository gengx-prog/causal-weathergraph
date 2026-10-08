#!/usr/bin/env bash
# Rerun every real-data analysis that uses data after 2023-01-10 with the re-acquired
# 2023-01-11--2025 segment (native 0.25-degree CDS fields, WB2 conservative remapping).
# The earlier results are kept in revision_outputs/_archive_cds5p625_route/. Each step is
# logged with its exit status; a failed step does not stop independent later steps.

PY="/d/Paper2/Major Revision/.venv/Scripts/python.exe"
REPO="/d/Paper2/Major Revision/causal-weathergraph"
OUT='D:\Paper2\Major Revision\revision_outputs'
DATA='D:\Paper2\Major Revision\supplementary_data\era5_primary_native\data_root'
EXT='D:\Paper2\Major Revision\supplementary_data\era5_primary_native\era5_cds_native_conservative_6h_64x32_850hPa_2023-01-11_2025'
LEGACY_DIR='D:\Paper2\vipuser\论文2026_516\Causal\causal-weathergraph\outputs'
FULL="$OUT\\inputs\\region_series_fullrecord.npz"
export PYTHONWARNINGS=ignore
cd "$REPO"

FAILED=()
run() {
    local name="$1"; shift
    echo "== $(date -u +%H:%M:%S)Z START $name"
    local t0=$SECONDS
    if "$@"; then
        echo "== $(date -u +%H:%M:%S)Z OK $name ($((SECONDS - t0)) s)"
    else
        echo "== $(date -u +%H:%M:%S)Z FAILED $name (exit $?, $((SECONDS - t0)) s)"
        FAILED+=("$name")
    fi
}

run inputs "$PY" -m revision.prepare_inputs --data-root "$DATA" --legacy "$LEGACY_DIR\\processed\\region_series.npz" \
    --output "$OUT\\inputs" --full-record-output "$FULL"
run inputs_diurnal "$PY" -m revision.prepare_inputs_diurnal --data-root "$DATA" --output "$OUT\\inputs_diurnal" \
    --reference "$OUT\\inputs\\region_trainfit.npz"
run graph_nulls "$PY" -m revision.run_graph_nulls --input "$FULL" --output "$OUT\\graph_nulls" --candidate-replicates 100 --topology-replicates 200
run inference "$PY" -m revision.run_inference --input "$FULL" --legacy-edges "$LEGACY_DIR\\causal_edges\\all_edges.csv" \
    --output "$OUT\\inference" --bootstrap-replicates 200
run cluster_inference "$PY" -m revision.run_cluster_inference --input "$FULL" --output "$OUT\\inference_exploratory_cluster"
run holdout "$PY" -m revision.run_holdout --input "$OUT\\inputs\\region_trainfit.npz" --output "$OUT\\holdout_aligned" --pcmci
run three_stage "$PY" -m revision.run_three_stage --input "$OUT\\inputs\\region_trainfit.npz" --output "$OUT\\three_stage"
run three_stage_diurnal "$PY" -m revision.run_three_stage --input "$OUT\\inputs_diurnal\\region_trainfit.npz" --output "$OUT\\three_stage_diurnal"
for clim in month month_hour; do
    run "ceres_$clim" "$PY" -m revision.run_ceres_cloud_substitution --climatology "$clim" --extension-dir "$EXT" \
        --output "$OUT\\ceres_cloud_substitution" --inputs "$OUT\\inputs"
done
run regime_contrasts "$PY" -m revision.run_regime_contrasts --input "$OUT\\inputs\\region_trainfit.npz" --output "$OUT\\regime_contrasts"
run hemisphere_contrasts "$PY" -m revision.run_hemisphere_contrasts --input "$OUT\\inputs\\region_trainfit.npz" --output "$OUT\\hemisphere_contrasts"
run latitude_contrasts "$PY" -m revision.run_latitude_contrasts --freeze --run --output "$OUT\\latitude_contrasts"
run path_diagnostics "$PY" -m revision.run_path_diagnostics --input "$OUT\\inputs\\region_trainfit_vectors.npz" \
    --discovery "$OUT\\holdout_aligned\\discovery_edges.csv" --output "$OUT\\path_diagnostics"
run full_var "$PY" -m revision.run_full_var --input "$OUT\\inputs\\region_trainfit.npz" --output "$OUT\\full_var" --history-lags 3 --source-lags 3
run full_var_lag12 "$PY" -m revision.run_full_var --input "$OUT\\inputs\\region_trainfit.npz" --output "$OUT\\full_var_lag12" \
    --history-lags 12 --source-lags 3 --start-lag 12
run full_var_lag3_aligned12 "$PY" -m revision.run_full_var --input "$OUT\\inputs\\region_trainfit.npz" --output "$OUT\\full_var_lag3_aligned12" \
    --history-lags 3 --source-lags 3 --start-lag 12
run compare_full_var "$PY" -m revision.compare_full_var_histories
run physical_controls_inputs "$PY" -m revision.prepare_physical_controls --output "$OUT\\physical_controls_inputs"
run physical_controls_experiments "$PY" -m revision.run_physical_controls --physical "$OUT\\physical_controls_inputs\\region_controls_trainfit.npz" \
    --output "$OUT\\physical_controls_experiments"
run lag_window_sensitivity "$PY" -m revision.run_lag_window_sensitivity --output "$OUT\\lag_window_sensitivity"
run nino_freeze "$PY" -m revision.run_external_nino_regimes --freeze --output "$OUT\\external_nino_regimes"
run nino_run "$PY" -m revision.run_external_nino_regimes --run --output "$OUT\\external_nino_regimes"
run nino_joint_freeze "$PY" -m revision.run_external_nino_physical_joint --freeze --output "$OUT\\external_nino_physical_joint"
run nino_joint_run "$PY" -m revision.run_external_nino_physical_joint --run --output "$OUT\\external_nino_physical_joint"
run transport_features "$PY" revision/transport_consistency/features.py --execute --plan "$OUT\\transport_consistency_v1\\design.json" \
    --output "$OUT\\transport_consistency_v1\\inputs" --data-root "$DATA"
run transport_models "$PY" revision/transport_consistency/experiment.py --input "$OUT\\transport_consistency_v1\\inputs\\input_data.npz" \
    --design "$OUT\\transport_consistency_v1\\design.json" --output "$OUT\\transport_consistency_v1\\model"
run transport_external "$PY" revision/transport_consistency/external_check.py --root "$OUT\\transport_consistency_v1"
run overlap_null_round2 "$PY" -m revision.run_graph_overlap_null_round2
run data_shift "$PY" -m revision.data_shift_diagnostics --inputs "$OUT\\inputs" --output "$OUT\\data_shift_diagnostics"
run transport_audit "$PY" revision/transport_consistency/audit.py --output "$OUT\\transport_consistency_v1"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "RERUN OK $(date -u +%FT%TZ)"
else
    echo "RERUN FINISHED WITH FAILURES: ${FAILED[*]} $(date -u +%FT%TZ)"
fi
