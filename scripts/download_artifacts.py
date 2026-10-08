"""Download selected versioned research artifacts, verifying hashes before extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

from verify_artifacts import ROOT, check_records, safe_path, sha256


def extract_verified(archive_path, asset, destination):
    expected = {record["path"]: record for record in asset["files"]}
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if len(members) != len(expected) or {m.filename for m in members} != set(expected):
            raise ValueError(f"Archive inventory mismatch: {asset['name']}")
        for member in members:
            target = safe_path(destination, member.filename)
            record = expected[member.filename]
            if member.file_size != record["bytes"]:
                raise ValueError(f"Unexpected expanded size: {member.filename}")
            if target.exists():
                _, errors = check_records(destination, [record])
                if errors:
                    raise FileExistsError(f"Refusing to overwrite different file: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            with archive.open(member) as src, partial.open("xb") as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            if sha256(partial) != record["sha256"]:
                raise ValueError(f"Expanded file hash mismatch: {member.filename}")
            partial.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", action="append", help="Asset name; default core-inputs.zip; repeat or use all")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/data")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/downloads")
    parser.add_argument("--from-directory", type=Path, help="Use already downloaded ZIPs instead of the network")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "artifacts/release-manifest.json").read_text(encoding="utf-8"))
    selected = set(args.asset or ["core-inputs.zip"])
    known = {asset["name"] for asset in manifest["assets"]}
    if selected != {"all"} and not selected <= known:
        parser.error(f"Unknown asset; choose from {sorted(known)} or all")
    args.cache.mkdir(parents=True, exist_ok=True)
    for asset in manifest["assets"]:
        if selected != {"all"} and asset["name"] not in selected:
            continue
        path = (args.from_directory or args.cache) / asset["name"]
        if not path.exists():
            if args.from_directory:
                raise FileNotFoundError(path)
            partial = path.with_name(path.name + ".part")
            print(f"Downloading {asset['name']} ({asset['bytes']:,} bytes)", flush=True)
            request = urllib.request.Request(asset["url"], headers={"User-Agent": "causal-weathergraph-reproduction"})
            with urllib.request.urlopen(request, timeout=120) as src, partial.open("wb") as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            if partial.stat().st_size != asset["bytes"] or sha256(partial) != asset["sha256"]:
                raise ValueError(f"Downloaded asset failed integrity verification: {asset['name']}")
            partial.rename(path)
        if path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
            raise ValueError(f"Existing asset failed integrity verification: {path}")
        extract_verified(path, asset, args.output)
        checked, errors = check_records(args.output, asset["files"])
        if errors:
            raise ValueError(errors)
        print(f"Verified and extracted {asset['name']}: {checked} files", flush=True)


if __name__ == "__main__":
    main()
