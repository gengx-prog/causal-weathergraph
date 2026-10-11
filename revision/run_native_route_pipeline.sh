#!/usr/bin/env bash
# Rebuild the 2023-01-11--2025 segment from native CDS fields with WB2 conservative
# remapping, then rerun the analyses that use that segment. Outputs go to new
# *_native directories; the earlier outputs are left untouched.
set -euo pipefail

PY="${PYTHON:-python}"
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${CWG_OUTPUT_ROOT:-outputs}"
DATA="${CWG_DATA_ROOT:-data}"
EXT="${CWG_EXTENSION_ROOT:-data/era5_primary_native/era5_cds_native_conservative_6h_64x32_850hPa_2023-01-11_2025}"
# Required input configuration (example):
# export CWG_LEGACY_SERIES="data/legacy/processed/region_series.npz"
LEGACY="${CWG_LEGACY_SERIES:?Set CWG_LEGACY_SERIES to the recorded legacy region_series.npz}"
export PYTHONWARNINGS=ignore

cd "$REPO" || exit 1
step() { echo "== $(date -u +%H:%M:%S)Z $*"; }

step "remap extension and assemble segment"
"$PY" revision/regrid_era5_primary_native.py --kind extension --combine

step "regional inputs, monthly climatology"
"$PY" -m revision.prepare_inputs --data-root "$DATA" --legacy "$LEGACY" \
    --output "$OUT/inputs_native" --extension-dir "$EXT"

step "regional inputs, month x UTC-hour climatology"
"$PY" -m revision.prepare_inputs_diurnal --data-root "$DATA" --output "$OUT/inputs_diurnal_native" \
    --reference "$OUT/inputs_native/region_trainfit.npz" --extension-dir "$EXT"

step "three-stage, monthly climatology"
"$PY" -m revision.run_three_stage --input "$OUT/inputs_native/region_trainfit.npz" --output "$OUT/three_stage_native"

step "three-stage, diurnal climatology"
"$PY" -m revision.run_three_stage --input "$OUT/inputs_diurnal_native/region_trainfit.npz" --output "$OUT/three_stage_diurnal_native"

for clim in month month_hour; do
    step "CERES substitution ($clim)"
    "$PY" -m revision.run_ceres_cloud_substitution --output "$OUT/ceres_cloud_substitution_native" --climatology "$clim" \
        --extension-dir "$EXT" --inputs "$OUT/inputs_native"
done

step "summary of old versus native route"
"$PY" revision/summarize_native_route.py
step "done"
