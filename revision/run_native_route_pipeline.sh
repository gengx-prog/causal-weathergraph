#!/usr/bin/env bash
# Rebuild the 2023-01-11--2025 segment from native CDS fields with WB2 conservative
# remapping, then rerun the analyses that use that segment. Outputs go to new
# *_native directories; the earlier outputs are left untouched.
set -euo pipefail

PY="/d/Paper2/Major Revision/.venv/Scripts/python.exe"
REPO="/d/Paper2/Major Revision/causal-weathergraph"
OUT='D:\Paper2\Major Revision\revision_outputs'
EXT='D:\Paper2\Major Revision\supplementary_data\era5_primary_native\era5_cds_native_conservative_6h_64x32_850hPa_2023-01-11_2025'
LEGACY='D:\Paper2\vipuser\论文2026_516\Causal\causal-weathergraph\outputs\processed\region_series.npz'
export PYTHONWARNINGS=ignore

cd "$REPO"
step() { echo "== $(date -u +%H:%M:%S)Z $*"; }

step "remap extension and assemble segment"
"$PY" revision/regrid_era5_primary_native.py --kind extension --combine

step "regional inputs, monthly climatology"
"$PY" -m revision.prepare_inputs --data-root 'D:\Paper2\vipuser\Data' --legacy "$LEGACY" \
    --output "$OUT\\inputs_native" --extension-dir "$EXT"

step "regional inputs, month x UTC-hour climatology"
"$PY" -m revision.prepare_inputs_diurnal --data-root 'D:\Paper2\vipuser\Data' --output "$OUT\\inputs_diurnal_native" \
    --reference "$OUT\\inputs_native\\region_trainfit.npz" --extension-dir "$EXT"

step "three-stage, monthly climatology"
"$PY" -m revision.run_three_stage --input "$OUT\\inputs_native\\region_trainfit.npz" --output "$OUT\\three_stage_native"

step "three-stage, diurnal climatology"
"$PY" -m revision.run_three_stage --input "$OUT\\inputs_diurnal_native\\region_trainfit.npz" --output "$OUT\\three_stage_diurnal_native"

for clim in month month_hour; do
    step "CERES substitution ($clim)"
    "$PY" -m revision.run_ceres_cloud_substitution --output "$OUT\\ceres_cloud_substitution_native" --climatology "$clim" \
        --extension-dir "$EXT" --inputs "$OUT\\inputs_native"
done

step "summary of old versus native route"
"$PY" revision/summarize_native_route.py
step "done"
