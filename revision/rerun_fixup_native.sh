#!/usr/bin/env bash
# Second pass of rerun_all_native.sh: the physical-control chain. prepare_physical_controls.py imports
# prepare_inputs as a sibling module, so it runs as a script; the steps below depend on its output.
PY="/d/Paper2/Major Revision/.venv/Scripts/python.exe"
REPO="/d/Paper2/Major Revision/causal-weathergraph"
OUT='D:\Paper2\Major Revision\revision_outputs'
DATA='D:\Paper2\Major Revision\supplementary_data\era5_primary_native\data_root'
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
        echo "== $(date -u +%H:%M:%S)Z FAILED $name ($((SECONDS - t0)) s)"
        FAILED+=("$name")
    fi
}

# Remove partial outputs of the failed first attempts (the archived originals are untouched).
rm -rf "/d/Paper2/Major Revision/revision_outputs/physical_controls_inputs" "/d/Paper2/Major Revision/revision_outputs/physical_controls_experiments" \
       "/d/Paper2/Major Revision/revision_outputs/lag_window_sensitivity" "/d/Paper2/Major Revision/revision_outputs/external_nino_physical_joint"
find "/d/Paper2/Major Revision/revision_outputs/transport_consistency_v1" -mindepth 1 -maxdepth 1 ! -name design.json -exec rm -rf {} +

run physical_controls_inputs "$PY" revision/prepare_physical_controls.py --output "$OUT\\physical_controls_inputs"
run physical_controls_experiments "$PY" -m revision.run_physical_controls --physical "$OUT\\physical_controls_inputs\\region_controls_trainfit.npz" \
    --output "$OUT\\physical_controls_experiments"
run lag_window_sensitivity "$PY" -m revision.run_lag_window_sensitivity --output "$OUT\\lag_window_sensitivity"
run nino_joint_freeze "$PY" -m revision.run_external_nino_physical_joint --freeze --output "$OUT\\external_nino_physical_joint"
run nino_joint_run "$PY" -m revision.run_external_nino_physical_joint --run --output "$OUT\\external_nino_physical_joint"
run transport_features "$PY" revision/transport_consistency/features.py --execute --plan "$OUT\\transport_consistency_v1\\design.json" \
    --output "$OUT\\transport_consistency_v1\\inputs" --data-root "$DATA"
run transport_models "$PY" revision/transport_consistency/experiment.py --input "$OUT\\transport_consistency_v1\\inputs\\input_data.npz" \
    --design "$OUT\\transport_consistency_v1\\design.json" --output "$OUT\\transport_consistency_v1\\model"
run transport_external "$PY" revision/transport_consistency/external_check.py --root "$OUT\\transport_consistency_v1"
run transport_audit "$PY" revision/transport_consistency/audit.py --output "$OUT\\transport_consistency_v1"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "FIXUP OK $(date -u +%FT%TZ)"
else
    echo "FIXUP FINISHED WITH FAILURES: ${FAILED[*]} $(date -u +%FT%TZ)"
fi
