"""Package an explicit raw-data inventory without changing any source bytes.

Only allowlisted files from the supplied inventories are read. ZIPs are stored
without recompression, remain below 1.8 GB, and are verified member by member.
The private audit inventories may contain local paths; the published manifest
contains only relative archive paths, checksums, and source descriptions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

from verify_artifacts import safe_path, sha256
from download_artifacts import open_temporary, publish_temporary

MAX_ARCHIVE_BYTES = 1_800_000_000
ZIP_RESERVE = 2_000_000
RELEASE_TAG = "raw-data-20261008"
BASE_URL = f"https://github.com/gengx-prog/causal-weathergraph/releases/download/{RELEASE_TAG}"


def load_inventory(paths, workspace):
    files, datasets, seen = [], {}, set()
    for path in paths:
        inventory = json.loads(path.read_text(encoding="utf-8"))
        for dataset in inventory.get("datasets", []):
            if dataset["id"].startswith("ceres-") and "provider_doi" in dataset:
                dataset["netcdf_doi_attribute"] = dataset.pop("provider_doi")
                dataset["citation_doi"] = ["10.5067/TERRA-AQUA-NOAA20/CERES/SYN1DEG-1HOUR_L3.004B"]
                if dataset["id"] == "ceres-global":
                    dataset["citation_doi"].append("10.5067/Terra-Aqua-NOAA20/CERES/SYN1degDay_L3.004B")
                dataset["doi_note"] = "NetCDF attributes are preserved literally; citation_doi uses the current official NASA CMR collection record. No source data bytes were changed."
            datasets[dataset["id"]] = dataset
        for item in inventory.get("selected_files", []) + inventory.get("safe_metadata_files", []):
            original = Path(item["source_absolute"])
            source = original.resolve()
            if not source.is_relative_to(workspace.resolve()) or original.is_symlink():
                raise ValueError(f"Source leaves declared workspace: {source}")
            if not source.is_file() or source.stat().st_size != item["bytes"] or item["bytes"] <= 0:
                raise ValueError(f"Source missing, empty or changed size: {source}")
            archive_path = item["archive_path"]
            safe_path(Path.cwd(), archive_path)
            if archive_path in seen:
                raise ValueError(f"Duplicate archive path: {archive_path}")
            if not re.fullmatch(r"[a-z0-9-]+", item["dataset_id"]):
                raise ValueError(f"Invalid dataset identifier: {item['dataset_id']}")
            if item["bytes"] > MAX_ARCHIVE_BYTES - ZIP_RESERVE:
                raise ValueError(f"A source file needs explicit lossless splitting: {source}")
            seen.add(archive_path)
            files.append({**item, "source_absolute": str(source)})
        for key, stats in inventory.get("summary", {}).items():
            if key not in datasets:
                datasets[key] = {"id": key, "file_count": stats["count"], "bytes": stats["bytes"],
                                 "source_links": inventory.get("provider_sources", {})}
    return sorted(files, key=lambda x: (x["dataset_id"], x["archive_path"])), datasets


def plan_archives(files, limit=MAX_ARCHIVE_BYTES - ZIP_RESERVE):
    by_dataset = defaultdict(list)
    for item in files:
        by_dataset[item["dataset_id"]].append(item)
    plans = []
    for dataset, selected in sorted(by_dataset.items()):
        group, size, number = [], 0, 1
        for item in selected:
            if item["bytes"] > limit:
                raise ValueError("Individual source is larger than the archive payload limit")
            if group and size + item["bytes"] > limit:
                plans.append((f"raw-{dataset}-{number:03d}.zip", dataset, group))
                group, size, number = [], 0, number + 1
            group.append(item)
            size += item["bytes"]
        if group:
            plans.append((f"raw-{dataset}-{number:03d}.zip", dataset, group))
    return plans


def verify_zip(path, records):
    expected = {item["path"]: item for item in records}
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) != len(expected) or set(archive.namelist()) != set(expected):
            raise ValueError(f"Archive inventory differs: {path.name}")
        for member in archive.infolist():
            record = expected[member.filename]
            if member.file_size != record["bytes"]:
                raise ValueError(f"Archive member size differs: {member.filename}")
            digest = hashlib.sha256()
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Archive member SHA256 differs: {member.filename}")


def freeze_plan(plans, state, resume):
    identity = [{"name": name, "dataset_id": dataset,
                 "files": [{"path": item["archive_path"], "bytes": item["bytes"],
                            "recorded_sha256": item.get("recorded_sha256")}
                           for item in items]} for name, dataset, items in plans]
    path = state / "full-plan.json"
    if path.exists():
        if not resume or json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Complete archive plan changed, or --resume was omitted; use a new output directory")
    else:
        if any(state.glob("*.zip.json")):
            raise ValueError("Existing receipts lack a frozen complete plan; inspect them before adopting a new plan")
        path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")


def build_one(name, dataset, items, output, state, resume):
    path, receipt = output / name, state / (name + ".json")
    identity = [{"path": x["archive_path"], "bytes": x["bytes"],
                 "mtime_ns": Path(x["source_absolute"]).stat().st_mtime_ns,
                 "recorded_sha256": x.get("recorded_sha256")} for x in items]
    if path.is_symlink() or receipt.is_symlink():
        raise ValueError(f"Refusing symlink archive/receipt: {path}")
    if path.exists() or receipt.exists():
        if not resume or not (path.exists() and receipt.exists()):
            raise FileExistsError(f"Existing archive/receipt requires a complete matching --resume: {path}")
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if saved["source_identity"] != identity or path.stat().st_size != saved["asset"]["bytes"] or sha256(path) != saved["asset"]["sha256"]:
            raise ValueError(f"Existing archive or source identity changed: {name}")
        return saved["asset"]
    partial = output / (name + ".partial")
    records = []
    with open_temporary(partial, "wb" if resume else "xb") as stream, zipfile.ZipFile(
            stream, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for item in items:
            source = Path(item["source_absolute"])
            before = source.stat()
            digest, total = hashlib.sha256(), 0
            info = zipfile.ZipInfo(item["archive_path"], date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            with source.open("rb") as src, archive.open(info, "w", force_zip64=True) as dst:
                for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                    dst.write(block)
                    digest.update(block)
                    total += len(block)
            after = source.stat()
            if total != item["bytes"] or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Source changed while archiving: {source}")
            actual = digest.hexdigest()
            if item.get("recorded_sha256") and actual != item["recorded_sha256"]:
                raise ValueError(f"Original acquisition checksum mismatch: {item['archive_path']}")
            record = {"path": item["archive_path"], "bytes": total, "sha256": actual}
            for field in ("category", "description", "source_relative"):
                if field in item:
                    record[field] = item[field]
            record["matches_acquisition_checksum"] = True if item.get("recorded_sha256") else None
            records.append(record)
    if partial.stat().st_size >= MAX_ARCHIVE_BYTES:
        raise ValueError(f"Archive exceeds release size ceiling: {name}")
    verify_zip(partial, records)
    asset = {"name": name, "dataset_id": dataset, "bytes": partial.stat().st_size,
             "sha256": sha256(partial), "url": BASE_URL + "/" + name,
             "files": records}
    publish_temporary(partial, path)
    receipt.write_text(json.dumps({"source_identity": identity, "asset": asset}, indent=2) + "\n", encoding="utf-8")
    return asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, action="append", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    files, datasets = load_inventory(args.inventory, args.workspace_root)
    plans = plan_archives(files)
    print(f"Selected {len(files)} files / {sum(x['bytes'] for x in files):,} bytes / {len(plans)} archives", flush=True)
    if args.plan_only:
        return
    if args.manifest.exists() and not args.resume:
        parser.error("Manifest exists; use --resume to verify and reuse completed matching archives")
    args.output.mkdir(parents=True, exist_ok=True)
    state = args.output.parent / "build-state"
    state.mkdir(exist_ok=True)
    freeze_plan(plans, state, args.resume)
    assets = []
    for index, (name, dataset, items) in enumerate(plans, start=1):
        assets.append(build_one(name, dataset, items, args.output, state, args.resume))
        print(f"VERIFIED {index}/{len(plans)} {name} {assets[-1]['bytes']:,} bytes", flush=True)
    for key, value in datasets.items():
        selected = [item for item in files if item["dataset_id"] == key]
        value["archived_files"] = len(selected)
        value["archived_bytes"] = sum(item["bytes"] for item in selected)
    manifest = {"schema_version": 1, "release_tag": RELEASE_TAG,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Actual retained ERA5/WB2 and CERES acquisition subsets and original study inputs, byte-preserved; not all variables or all years of the provider archives.",
                "source_files": len(files), "source_bytes": sum(x["bytes"] for x in files),
                "datasets": list(datasets.values()), "assets": assets}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (args.output.parent / "build-complete.json").write_text(json.dumps({"assets": len(assets), "manifest": str(args.manifest.resolve())}), encoding="utf-8")
    print(f"COMPLETE {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
