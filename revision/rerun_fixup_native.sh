#!/usr/bin/env bash
# Second pass of rerun_all_native.sh: the physical-control chain. prepare_physical_controls.py imports
# prepare_inputs as a sibling module, so it runs as a script; the steps below depend on its output.
PY="${PYTHON:-python}"
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${CWG_OUTPUT_ROOT:-outputs}"
DATA="${CWG_DATA_ROOT:-data/era5_primary_native/data_root}"
export PYTHONWARNINGS=ignore
cd "$REPO" || exit 1

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

# Remove only the named partial outputs inside the selected experiment output root.
# Resolve and check every target before removing any; keep the frozen transport design.
"$PY" - "$OUT" <<'PY_CLEANUP' || exit 1
from pathlib import Path
import shutil
import sys

output = Path(sys.argv[1]).resolve()
repository = Path.cwd().resolve()
if output == Path(output.anchor) or output == repository or repository.is_relative_to(output):
    raise SystemExit("Refusing cleanup: output root is a filesystem root or repository ancestor")
names = ("physical_controls_inputs", "physical_controls_experiments",
         "lag_window_sensitivity", "external_nino_physical_joint")
targets = [output / name for name in names]
transport = output / "transport_consistency_v1"
if not transport.resolve().is_relative_to(output):
    raise SystemExit("Refusing cleanup: transport output resolves outside the output root")
if transport.exists():
    targets.extend(path for path in transport.iterdir() if path.name != "design.json")
for target in targets:
    resolved = target.resolve()
    if resolved == output or not resolved.is_relative_to(output):
        raise SystemExit("Refusing cleanup: a target resolves outside the output root")
for target in targets:
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
PY_CLEANUP

run physical_controls_inputs "$PY" revision/prepare_physical_controls.py --output "$OUT/physical_controls_inputs"
run physical_controls_experiments "$PY" -m revision.run_physical_controls --physical "$OUT/physical_controls_inputs/region_controls_trainfit.npz" \
    --output "$OUT/physical_controls_experiments"
run lag_window_sensitivity "$PY" -m revision.run_lag_window_sensitivity --output "$OUT/lag_window_sensitivity"
run nino_joint_freeze "$PY" -m revision.run_external_nino_physical_joint --freeze --output "$OUT/external_nino_physical_joint"
run nino_joint_run "$PY" -m revision.run_external_nino_physical_joint --run --output "$OUT/external_nino_physical_joint"
run transport_features "$PY" revision/transport_consistency/features.py --execute --plan "$OUT/transport_consistency_v1/design.json" \
    --output "$OUT/transport_consistency_v1/inputs" --data-root "$DATA"
run transport_models "$PY" revision/transport_consistency/experiment.py --input "$OUT/transport_consistency_v1/inputs/input_data.npz" \
    --design "$OUT/transport_consistency_v1/design.json" --output "$OUT/transport_consistency_v1/model"
run transport_external "$PY" revision/transport_consistency/external_check.py --root "$OUT/transport_consistency_v1"
run transport_audit "$PY" revision/transport_consistency/audit.py --output "$OUT/transport_consistency_v1"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "FIXUP OK $(date -u +%FT%TZ)"
else
    echo "FIXUP FINISHED WITH FAILURES: ${FAILED[*]} $(date -u +%FT%TZ)"
fi
