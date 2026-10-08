#!/usr/bin/env bash
# Rebuild the 2023-01-11--2025 segment from the CDS grid-box-average (conservative)
# route, check it against WB2 on the overlap windows and against the native-
# conservative route after the splice, then rerun the analyses that use the segment.
# Outputs go to new *_conservative directories; earlier outputs are left untouched.
set -euo pipefail

PY="/d/Paper2/Major Revision/.venv/Scripts/python.exe"
REPO="/d/Paper2/Major Revision/causal-weathergraph"
OUT='D:\Paper2\Major Revision\revision_outputs'
GBA='D:\Paper2\Major Revision\supplementary_data\era5_primary_gba'
EXT="$GBA\\era5_cds_gba_6h_64x32_850hPa_2023-01-11_2025"
LEGACY='D:\Paper2\vipuser\论文2026_516\Causal\causal-weathergraph\outputs\processed\region_series.npz'
export PYTHONWARNINGS=ignore

cd "$REPO"
step() { echo "== $(date -u +%H:%M:%S)Z $*"; }

step "assemble segment and overlap files"
"$PY" revision/assemble_era5_primary_gba.py --kind all

step "same-date check against WB2 (152 overlap timestamps)"
"$PY" revision/run_native_route_check.py --native-dir "$GBA\\overlap_64x32" --label cds_grid_box_average \
    --output "$OUT\\source_overlap_gba_route"

step "cross-check against the native-conservative route after the splice"
"$PY" revision/compare_gba_native.py

step "regional inputs, monthly climatology"
"$PY" -m revision.prepare_inputs --data-root 'D:\Paper2\vipuser\Data' --legacy "$LEGACY" \
    --output "$OUT\\inputs_conservative" --extension-dir "$EXT"

step "regional inputs, month x UTC-hour climatology"
"$PY" -m revision.prepare_inputs_diurnal --data-root 'D:\Paper2\vipuser\Data' --output "$OUT\\inputs_diurnal_conservative" \
    --reference "$OUT\\inputs_conservative\\region_trainfit.npz" --extension-dir "$EXT"

step "three-stage, monthly climatology"
"$PY" -m revision.run_three_stage --input "$OUT\\inputs_conservative\\region_trainfit.npz" --output "$OUT\\three_stage_conservative"

step "three-stage, diurnal climatology"
"$PY" -m revision.run_three_stage --input "$OUT\\inputs_diurnal_conservative\\region_trainfit.npz" \
    --output "$OUT\\three_stage_diurnal_conservative"

for clim in month month_hour; do
    step "CERES substitution ($clim)"
    "$PY" -m revision.run_ceres_cloud_substitution --output "$OUT\\ceres_cloud_substitution_conservative" --climatology "$clim" \
        --extension-dir "$EXT" --inputs "$OUT\\inputs_conservative"
done

step "summary of old versus conservative route"
"$PY" revision/summarize_native_route.py --suffix conservative
step "done"
