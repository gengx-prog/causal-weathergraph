"""Download selected versioned research artifacts, verifying hashes before extraction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import stat
import urllib.request
import zipfile

from verify_artifacts import ROOT, check_records, safe_path, select_assets, sha256


def temporary_stat(path):
    """Inspect a temporary file without following a symbolic link."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Refusing non-regular or symlink temporary path: {path}")
    return info


def open_temporary(path, mode):
    """Open an ordinary temporary file, checking its identity before truncation."""
    before = temporary_stat(path)
    if before is not None and mode == "xb":
        raise FileExistsError(f"Temporary file exists; use --resume to rebuild it: {path}")
    flags = os.O_WRONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if before is None:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o666)
    try:
        opened = os.fstat(descriptor)
        current = temporary_stat(path)
        if (not stat.S_ISREG(opened.st_mode) or current is None
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
                or (before is not None and (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino))):
            raise ValueError(f"Temporary path changed while opening: {path}")
        if mode == "ab":
            os.lseek(descriptor, 0, os.SEEK_END)
        else:
            os.ftruncate(descriptor, 0)
        return os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        raise


def publish_temporary(partial, target):
    """Promote verified bytes without replacing an existing final path."""
    temporary_stat(partial)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing final file: {target}")
    if os.name == "nt":
        # Windows rename fails if another process created the destination.
        partial.rename(target)
    else:
        # POSIX rename would overwrite it; link creation is atomic and exclusive.
        os.link(partial, target, follow_symlinks=False)
        partial.unlink()


def download_to_cache(asset, path):
    """Retain interrupted ZIPs and resume only a matching byte range."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing final file: {path}")
    partial = path.with_name(path.name + ".part")
    previous = temporary_stat(partial)
    offset = previous.st_size if previous is not None else 0
    if offset > asset["bytes"]:
        raise ValueError(f"Partial download is larger than the selected asset: {partial}. "
                         f"Remove only this temporary file and rerun the same command: {partial}")
    if offset < asset["bytes"]:
        headers = {"User-Agent": "causal-weathergraph-reproduction"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        print(f"Downloading {asset['name']} ({asset['bytes']:,} bytes; cached partial {offset:,})", flush=True)
        request = urllib.request.Request(asset["url"], headers=headers)
        with urllib.request.urlopen(request, timeout=120) as src:
            status = src.status
            if status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", src.headers.get("Content-Range", ""))
                if not match or tuple(map(int, match.groups())) != (offset, asset["bytes"] - 1, asset["bytes"]):
                    raise ValueError("Server returned an unexpected download byte range")
                mode = "ab" if offset else "wb"
            elif status == 200:
                # Some mirrors ignore Range; their full response replaces only the partial ZIP.
                offset, mode = 0, "wb"
            else:
                raise ValueError(f"Unexpected download response status: {status}")
            with open_temporary(partial, mode) as dst:
                total = offset
                for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                    total += len(block)
                    if total > asset["bytes"]:
                        raise ValueError("Download exceeds the manifest byte count")
                    dst.write(block)
    if partial.stat().st_size != asset["bytes"]:
        raise ValueError(f"Downloaded asset failed integrity verification: {asset['name']}. "
                         "The incomplete temporary file was retained; rerun the same command to resume.")
    if sha256(partial) != asset["sha256"]:
        raise ValueError(f"Downloaded asset failed integrity verification: {asset['name']}. "
                         f"Remove only this temporary file and rerun the same command: {partial}")
    publish_temporary(partial, path)


def extract_verified(archive_path, asset, destination):
    expected = {record["path"]: record for record in asset["files"]}
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if len(members) != len(expected) or {m.filename for m in members} != set(expected):
            raise ValueError(f"Archive inventory mismatch: {asset['name']}")
        for member in members:
            if (destination / member.filename).is_symlink():
                raise ValueError(f"Refusing symlink final path: {member.filename}")
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
            with archive.open(member) as src, open_temporary(partial, "wb") as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            if sha256(partial) != record["sha256"]:
                raise ValueError(f"Expanded file hash mismatch: {member.filename}")
            publish_temporary(partial, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", action="append", help="Asset name; default core-inputs.zip; repeat or use all")
    parser.add_argument("--dataset", action="append", help="Dataset ID; repeat to select multiple raw-data groups")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/release-manifest.json")
    parser.add_argument("--list", action="store_true", help="List dataset/asset sizes without downloading")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/data")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/downloads")
    parser.add_argument("--from-directory", type=Path, help="Use already downloaded ZIPs instead of the network")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.list:
        groups = {}
        for asset in manifest["assets"]:
            group = groups.setdefault(asset.get("dataset_id", asset["name"]), {"archives": 0, "bytes": 0, "files": 0})
            group["archives"] += 1
            group["bytes"] += asset["bytes"]
            group["files"] += len(asset["files"])
        print(json.dumps(groups, indent=2))
        return
    try:
        selected = select_assets(manifest, args.asset, args.dataset)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Selected {len(selected)} archives ({sum(a['bytes'] for a in selected):,} download bytes)", flush=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    for asset in selected:
        path = safe_path(args.from_directory or args.cache, asset["name"])
        if not path.exists():
            if args.from_directory:
                raise FileNotFoundError(path)
            download_to_cache(asset, path)
        if path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
            raise ValueError(f"Existing asset failed integrity verification: {path}")
        extract_verified(path, asset, args.output)
        checked, errors = check_records(args.output, asset["files"])
        if errors:
            raise ValueError(errors)
        print(f"Verified and extracted {asset['name']}: {checked} files", flush=True)


if __name__ == "__main__":
    main()
