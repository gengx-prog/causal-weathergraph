"""Build the explicitly selected research archives; never scan an entire workspace.

This maintainer utility uses only the Python standard library. Inputs are preserved
byte for byte, ZIP entries use paths relative to revision_outputs, and every entry
and resulting archive is hashed. It does not upload or modify research outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile


RELEASE_TAG = "reproducibility-20261008.1"
BASE_URL = f"https://github.com/gengx-prog/causal-weathergraph/releases/download/{RELEASE_TAG}"
MAX_ASSET_BYTES = 2_000_000_000
MAX_SMALL_ARRAY_BYTES = 10_000_000

CORE_FILES = (
    "inputs/region_trainfit.npz",
    "inputs/trainfit_parameters.npz",
    "inputs/data_manifest.json",
    "graph_nulls/candidates_symmetric_core.csv",
    "holdout_aligned/discovery_edges.csv",
)
EXTENDED_FILES = (
    "inputs/region_trainfit_vectors.npz",
    "inputs/region_series_fullrecord.npz",
    "inputs_diurnal/data_manifest.json",
    "inputs_diurnal/region_trainfit.npz",
    "whec_index/manifest.json",
    "whec_index/whec_parameters.npz",
    "whec_index/whec_regional.npz",
    "whec_index/zonal_mean_eddy_moisture_flux_1979_2018.csv",
    "ceres_cloud_substitution/ceres_cloud_6h_64x32.npz",
    "ceres_cloud_substitution/ceres_preparation_manifest.json",
    "physical_controls_inputs/coarse_surface_pressure_terrain_diagnostic.csv",
    "physical_controls_inputs/coarse_surface_pressure_terrain_diagnostic.json",
    "physical_controls_inputs/data_manifest.json",
    "physical_controls_inputs/preprocessing_plan.json",
    "physical_controls_inputs/region_controls_trainfit.npz",
    "physical_controls_inputs/source_manifest.json",
    "physical_controls_inputs/status.json",
    "physical_controls_inputs/trainfit_parameters.npz",
    "physical_controls_inputs/validation.json",
)
SUPPLEMENTAL_FILES = (
    "ceres_cloud_substitution/era5_cloud_6h_64x32_2017_2025.npz",
    "ceres_cloud_substitution/era5_cloud_reference_manifest.json",
)
EVIDENCE_DIRECTORIES = (
    "three_stage_diurnal",
    "inference",
    "inference_exploratory_cluster",
    "graph_nulls",
    "graph_overlap_null_round2",
    "holdout_aligned",
    "full_var",
    "full_var_lag12",
    "full_var_lag3_aligned12",
    "full_var_history_comparison",
    "regime_contrasts",
    "hemisphere_contrasts",
    "latitude_contrasts",
    "path_diagnostics",
    "joint_window_by_segment",
    "lag_window_sensitivity",
    "long_lag_benchmark",
    "simulations",
    "simulations_post_diagnostic",
    "rpcmci",
    "castle_response_aligned",
    "ceres_cloud_substitution",
    "source_overlap_comparison",
    "source_overlap_native_route",
    "native_route_summary",
    "whec_test",
    "physical_controls_experiments",
    "external_nino_regimes",
    "external_nino_physical_joint",
    "data_shift_diagnostics",
)
ALLOWED_SUFFIXES = {".csv", ".json", ".jsonl", ".npz", ".py", ".yaml", ".yml"}
EXCLUDED_COMPONENTS = {
    "logs", "cache", "models", "__pycache__", ".git", ".venv", "report_preview",
}
TEXT_SUFFIXES = {".json", ".jsonl", ".py", ".yaml", ".yml", ".csv"}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|password|authorization)[\"']?\s*[:=]\s*[\"'](?:Bearer\s+)?[A-Za-z0-9_./+:-]{24,}[\"']"),
    re.compile(r"(?i)https?://[^\s/:]+:[^\s/@]+@"),
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_file(root: Path, relative: str) -> Path:
    key = PurePosixPath(relative)
    if key.is_absolute() or ".." in key.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe archive path: {relative}")
    candidate = root.joinpath(*key.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(f"Required regular input file is missing: {candidate}")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Input escapes its declared source root: {candidate}")
    return candidate


def check_text(path: Path) -> None:
    """Reject recognizable embedded credentials without printing their values."""
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    content = path.read_text(encoding="utf-8-sig")
    for pattern in SECRET_PATTERNS:
        if pattern.search(content):
            raise ValueError(f"Potential embedded credential requires review: {path}")


def selected_directory(root: Path, relative: str) -> dict[str, Path]:
    directory = root / relative
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(f"Required evidence directory is missing: {directory}")
    selected = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(directory).parts
        if any(part in EXCLUDED_COMPONENTS or part.startswith("_archive") for part in parts):
            continue
        if path.name.lower().endswith("_model.npz"):
            continue
        allowed = path.suffix.lower() in ALLOWED_SUFFIXES or path.name == "LICENSE" or path.name.endswith(".csv.gz")
        if not allowed:
            continue
        if path.suffix.lower() == ".npz" and path.stat().st_size > MAX_SMALL_ARRAY_BYTES:
            continue
        key = path.relative_to(root).as_posix()
        selected[key] = checked_file(root, key)
    return selected


def build_selections(source: Path, supplemental: Path) -> dict[str, dict[str, Path]]:
    core = {key: checked_file(source, key) for key in CORE_FILES}
    core.update(selected_directory(source, "three_stage"))
    extended = {key: checked_file(source, key) for key in EXTENDED_FILES}
    extended.update({key: checked_file(supplemental, key) for key in SUPPLEMENTAL_FILES})
    evidence = {}
    for directory in EVIDENCE_DIRECTORIES:
        evidence.update(selected_directory(source, directory))
    # Every path occurs in only one archive; downloading multiple assets never
    # overwrites another asset's archived copy of the same evidence.
    for key in core.keys() | extended.keys():
        evidence.pop(key, None)
    result = {
        "core-inputs.zip": core,
        "extended-inputs.zip": extended,
        "experiment-evidence.zip": evidence,
    }
    for files in result.values():
        for path in files.values():
            check_text(path)
    return result


def write_archive(path: Path, files: dict[str, Path]) -> dict:
    records = []
    # NPZ and gzip files are already compressed. Storing them preserves bytes
    # while avoiding an expensive, ineffective second compression pass.
    with zipfile.ZipFile(path, "x", allowZip64=True, compresslevel=6) as archive:
        for key, source in sorted(files.items()):
            before = source.stat()
            digest = hashlib.sha256()
            info = zipfile.ZipInfo(key, date_time=(2026, 10, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if source.suffix.lower() in {".npz", ".gz"} else zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with source.open("rb") as reader, archive.open(info, "w", force_zip64=True) as writer:
                while chunk := reader.read(4 * 1024 * 1024):
                    digest.update(chunk)
                    writer.write(chunk)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Source changed while archiving: {source}")
            records.append({"path": key, "bytes": before.st_size, "sha256": digest.hexdigest()})
    if path.stat().st_size >= MAX_ASSET_BYTES:
        raise ValueError(f"Archive exceeds the release asset size ceiling: {path}")
    # Read every stored entry back, check CRCs and compare SHA-256, not just names.
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"ZIP CRC validation failed: {path}")
        for record in records:
            with archive.open(record["path"]) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                    raise ValueError(f"ZIP entry hash mismatch: {record['path']}")
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "url": f"{BASE_URL}/{path.name}",
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Recorded revision_outputs directory")
    parser.add_argument("--supplemental-root", type=Path, required=True, help="Staged, traceable ERA5 cloud reference")
    parser.add_argument("--output-dir", type=Path, required=True, help="New local directory for release ZIPs")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/release-manifest.json")
    parser.add_argument("--check-only", action="store_true", help="Validate the selection and report sizes without writing")
    args = parser.parse_args()
    selected = build_selections(args.source_root.resolve(), args.supplemental_root.resolve())
    summary = {name: {"files": len(files), "uncompressed_bytes": sum(p.stat().st_size for p in files.values())} for name, files in selected.items()}
    print(json.dumps(summary, indent=2), flush=True)
    if args.check_only:
        return
    for name in selected:
        if (args.output_dir / name).exists():
            raise FileExistsError(f"Use a new output directory; archive already exists: {args.output_dir / name}")
    if args.manifest.exists():
        raise FileExistsError(f"Use a new manifest path; file already exists: {args.manifest}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    required = sum(p.stat().st_size for files in selected.values() for p in files.values())
    if shutil.disk_usage(args.output_dir).free < required + 100_000_000:
        raise OSError("Insufficient free disk space for the selected archives")
    assets = []
    for name, files in selected.items():
        asset = write_archive(args.output_dir / name, files)
        assets.append(asset)
        print(json.dumps({"created": name, "bytes": asset["bytes"], "sha256": asset["sha256"], "files": len(asset["files"])}), flush=True)
    manifest = {"schema_version": 1, "release_tag": RELEASE_TAG, "assets": assets}
    with args.manifest.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
