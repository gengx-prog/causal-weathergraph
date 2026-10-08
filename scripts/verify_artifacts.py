"""Verify archived publication files and, optionally, downloaded research inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe archive path: {relative}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path leaves destination: {relative}")
    return target


def check_records(root, records):
    checked = 0
    errors = []
    for record in records:
        path = safe_path(root, record["path"])
        if not path.is_file():
            errors.append(f"Missing: {record['path']}")
        elif path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            errors.append(f"Size or SHA256 mismatch: {record['path']}")
        else:
            checked += 1
    return checked, errors


def select_assets(manifest, names=None, datasets=None, *, default_all=False):
    """Validate an explicit selection; large raw collections have no implicit download."""
    assets = manifest["assets"]
    if names and datasets:
        raise ValueError("Choose either --asset or --dataset, not both")
    if datasets:
        known = {asset.get("dataset_id") for asset in assets if asset.get("dataset_id")}
        unknown = set(datasets) - known
        if unknown:
            raise ValueError(f"Unknown dataset {sorted(unknown)}; choose from {sorted(known)}")
        return [asset for asset in assets if asset.get("dataset_id") in datasets]
    known = {asset["name"] for asset in assets}
    if not names:
        if default_all:
            return assets
        if "core-inputs.zip" not in known:
            raise ValueError("This manifest requires --dataset ID or --asset NAME/all; use --list to inspect download sizes")
        names = ["core-inputs.zip"]
    if set(names) == {"all"}:
        return assets
    if not set(names) <= known:
        raise ValueError(f"Unknown asset; choose from {sorted(known)} or all")
    return [asset for asset in assets if asset["name"] in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Extracted release files, e.g. artifacts/data")
    parser.add_argument("--asset", action="append", help="Release asset name; repeat to verify selected assets")
    parser.add_argument("--dataset", action="append", help="Dataset identifier in the selected manifest; repeat as needed")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/release-manifest.json")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "artifacts/repository-manifest.json").read_text(encoding="utf-8"))
    count, errors = check_records(ROOT, manifest["files"])
    zip_count = 0
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if path.suffix == ".zip" and path.is_file():
            try:
                with zipfile.ZipFile(path) as archive:
                    bad = archive.testzip()
                    if bad:
                        errors.append(f"ZIP CRC failure: {record['path']}:{bad}")
                    else:
                        zip_count += 1
            except zipfile.BadZipFile:
                errors.append(f"Invalid ZIP: {record['path']}")
    release_count = 0
    if args.data_root:
        release = json.loads(args.manifest.read_text(encoding="utf-8"))
        try:
            selected = select_assets(release, args.asset, args.dataset, default_all=True)
        except ValueError as exc:
            parser.error(str(exc))
        for asset in selected:
            n, failures = check_records(args.data_root, asset["files"])
            release_count += n
            errors.extend(failures)
    elif args.asset or args.dataset:
        parser.error("--asset/--dataset requires --data-root")
    result = {"status": "passed" if not errors else "failed", "repository_files_verified": count,
              "zip_crc_checks": zip_count, "release_files_verified": release_count, "errors": errors,
              "scope": "File integrity only; this does not rerun scientific experiments."}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
