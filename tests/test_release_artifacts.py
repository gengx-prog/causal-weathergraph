"""Release integrity checks use synthetic archives and never contact the network."""
from __future__ import annotations

import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest


@pytest.fixture
def artifact_tools(monkeypatch):
    # These files are CLI scripts whose sibling imports follow their script path.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return (importlib.import_module("verify_artifacts"),
            importlib.import_module("download_artifacts"))


def make_archive(tmp_path, contents=None):
    if contents is None:
        contents = {"inputs/nested/sample.csv": b"value\n1.25\n",
                    "evidence/result.json": b'{"complete": true}\n'}
    archive_path = tmp_path / "core-inputs.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in contents.items():
            archive.writestr(name, payload)
    asset = {
        "name": archive_path.name,
        "url": "https://example.invalid/core-inputs.zip",
        "bytes": archive_path.stat().st_size,
        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "files": [{"path": name, "bytes": len(payload),
                   "sha256": hashlib.sha256(payload).hexdigest()}
                  for name, payload in contents.items()],
    }
    return archive_path, asset, contents


def test_valid_archive_extracts_and_verifies_every_file(tmp_path, artifact_tools):
    verify, download = artifact_tools
    archive, asset, contents = make_archive(tmp_path)
    destination = tmp_path / "extracted"
    download.extract_verified(archive, asset, destination)
    assert verify.check_records(destination, asset["files"]) == (len(contents), [])
    assert {str(path.relative_to(destination).as_posix()): path.read_bytes()
            for path in destination.rglob("*") if path.is_file()} == contents


@pytest.mark.parametrize("mutation", ["missing", "changed_same_size", "changed_size"])
def test_verifier_reports_missing_or_corrupt_content(tmp_path, artifact_tools, mutation):
    verify, _ = artifact_tools
    payload = b"original"
    record = {"path": "sample.bin", "bytes": len(payload),
              "sha256": hashlib.sha256(payload).hexdigest()}
    if mutation != "missing":
        (tmp_path / "sample.bin").write_bytes(b"modified" if mutation == "changed_same_size" else b"short")
    checked, errors = verify.check_records(tmp_path, [record])
    assert checked == 0
    assert len(errors) == 1
    assert "Missing" in errors[0] if mutation == "missing" else "mismatch" in errors[0]


@pytest.mark.parametrize("unsafe_name", ["../escaped.bin", "/escaped.bin", "C:/escaped.bin",
                                        "folder/../../escaped.bin", "folder\\escaped.bin"])
def test_archive_path_traversal_is_rejected_before_writing(tmp_path, artifact_tools, unsafe_name):
    verify, download = artifact_tools
    archive, asset, _ = make_archive(tmp_path, {unsafe_name: b"untrusted"})
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Unsafe archive path"):
        verify.check_records(destination, asset["files"])
    # Windows ZIP creation can normalize a backslash to '/', in which case the
    # manifest/ZIP inventory check rejects the altered name even earlier.
    with pytest.raises(ValueError, match="Unsafe archive path|Archive inventory mismatch"):
        download.extract_verified(archive, asset, destination)
    assert not destination.exists()
    assert not (tmp_path / "escaped.bin").exists()


def test_extraction_preserves_conflicting_existing_file(tmp_path, artifact_tools):
    _, download = artifact_tools
    archive, asset, contents = make_archive(tmp_path)
    first_name = next(iter(contents))
    destination = tmp_path / "extracted"
    existing = destination / first_name
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"user data that must remain intact")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        download.extract_verified(archive, asset, destination)
    assert existing.read_bytes() == b"user data that must remain intact"
    assert not (destination / "evidence/result.json").exists()


def test_matching_existing_file_is_preserved_and_remaining_files_extract(tmp_path, artifact_tools):
    verify, download = artifact_tools
    archive, asset, contents = make_archive(tmp_path)
    first_name = next(iter(contents))
    destination = tmp_path / "extracted"
    existing = destination / first_name
    existing.parent.mkdir(parents=True)
    existing.write_bytes(contents[first_name])
    original_mtime = existing.stat().st_mtime_ns
    download.extract_verified(archive, asset, destination)
    assert existing.stat().st_mtime_ns == original_mtime
    assert verify.check_records(destination, asset["files"]) == (2, [])


def test_extraction_rejects_content_hash_mismatch(tmp_path, artifact_tools):
    _, download = artifact_tools
    archive, asset, contents = make_archive(tmp_path)
    first_name = next(iter(contents))
    asset["files"][0]["sha256"] = "0" * 64
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Expanded file hash mismatch"):
        download.extract_verified(archive, asset, destination)
    assert not (destination / first_name).exists()


def test_extraction_rejects_wrong_expanded_size_before_writing(tmp_path, artifact_tools):
    _, download = artifact_tools
    archive, asset, _ = make_archive(tmp_path)
    asset["files"][0]["bytes"] += 1
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Unexpected expanded size"):
        download.extract_verified(archive, asset, destination)
    assert not destination.exists()


def test_unlisted_archive_entry_is_rejected_before_extraction(tmp_path, artifact_tools):
    _, download = artifact_tools
    archive, asset, _ = make_archive(tmp_path)
    with zipfile.ZipFile(archive, "a") as stream:
        stream.writestr("unlisted.bin", b"extra")
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Archive inventory mismatch"):
        download.extract_verified(archive, asset, destination)
    assert not destination.exists()


def test_existing_partial_file_is_not_overwritten(tmp_path, artifact_tools):
    _, download = artifact_tools
    archive, asset, contents = make_archive(tmp_path)
    destination = tmp_path / "extracted"
    target = destination / next(iter(contents))
    partial = target.with_name(target.name + ".part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"unfinished prior download")
    with pytest.raises(FileExistsError):
        download.extract_verified(archive, asset, destination)
    assert partial.read_bytes() == b"unfinished prior download"
    assert not target.exists()


def test_downloader_rejects_bad_archive_hash_before_extraction(tmp_path, artifact_tools, monkeypatch):
    _, download = artifact_tools
    archive, asset, _ = make_archive(tmp_path)
    manifest_dir = tmp_path / "artifacts"
    manifest_dir.mkdir()
    asset["sha256"] = "0" * 64
    (manifest_dir / "release-manifest.json").write_text(
        json.dumps({"assets": [asset]}), encoding="utf-8")
    monkeypatch.setattr(download, "ROOT", tmp_path)
    monkeypatch.setattr(download.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: io.BytesIO(archive.read_bytes()))
    monkeypatch.setattr(sys, "argv", ["download_artifacts.py"])
    with pytest.raises(ValueError, match="Downloaded asset failed integrity verification"):
        download.main()
    assert not (manifest_dir / "data").exists()
    assert not (manifest_dir / "downloads" / asset["name"]).exists()


def test_local_archive_download_path_roundtrip(tmp_path, artifact_tools, monkeypatch):
    verify, download = artifact_tools
    _, asset, contents = make_archive(tmp_path)
    manifest_dir = tmp_path / "artifacts"
    manifest_dir.mkdir()
    (manifest_dir / "release-manifest.json").write_text(
        json.dumps({"assets": [asset]}), encoding="utf-8")
    monkeypatch.setattr(download, "ROOT", tmp_path)
    def unexpected_network(*_args, **_kwargs):
        pytest.fail("--from-directory must not contact the network")
    monkeypatch.setattr(download.urllib.request, "urlopen", unexpected_network)
    monkeypatch.setattr(sys, "argv", ["download_artifacts.py", "--from-directory", str(tmp_path)])
    download.main()
    assert verify.check_records(manifest_dir / "data", asset["files"]) == (len(contents), [])
