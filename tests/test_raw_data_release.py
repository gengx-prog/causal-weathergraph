"""Small offline fixtures exercise raw packaging, selection and reproduction."""
from __future__ import annotations

import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import zipfile

import pytest


@pytest.fixture
def raw_tools(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    verify = importlib.import_module("verify_artifacts")
    download = importlib.import_module("download_artifacts")
    builder = importlib.import_module("build_raw_data_release")

    def unexpected_network(*_args, **_kwargs):
        pytest.fail("Raw-release fixture tests must never contact the network")

    monkeypatch.setattr(download.urllib.request, "urlopen", unexpected_network)
    return builder, verify, download


def inventory_fixture(tmp_path):
    workspace = tmp_path / "source"
    workspace.mkdir()
    selected = []
    payloads = {}
    for dataset, name, payload in [
        ("ceres-test", "north.nc", b"CDF\x01\x00\xff\x80\x00original north\n"),
        ("ceres-test", "south.nc", b"CDF\x01\x00\x00original south\r\n"),
        ("era5-test", "sample.npy", b"\x93NUMPY\x00\xfforiginal ERA5\x00"),
    ]:
        path = workspace / dataset / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)
        archive_path = f"supplementary_data/{dataset}/{name}"
        payloads[archive_path] = payload
        selected.append({"source_absolute": str(path), "source_relative": f"{dataset}/{name}",
                         "archive_path": archive_path, "dataset_id": dataset,
                         "bytes": len(payload), "recorded_sha256": hashlib.sha256(payload).hexdigest()})
    inventory = {"datasets": [{"id": "ceres-test", "product": "fixture CERES"},
                              {"id": "era5-test", "product": "fixture ERA5"}],
                 "selected_files": selected}
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    return workspace, path, inventory, payloads


def run_builder(builder, monkeypatch, inventory, workspace, output, manifest, *extra):
    monkeypatch.setattr(sys, "argv", ["build_raw_data_release.py", "--inventory", str(inventory),
                                    "--workspace-root", str(workspace), "--output", str(output),
                                    "--manifest", str(manifest), *extra])
    builder.main()


def built_fixture(tmp_path, raw_tools, monkeypatch):
    builder, _, _ = raw_tools
    workspace, inventory_path, inventory, payloads = inventory_fixture(tmp_path)
    output = tmp_path / "release" / "archives"
    manifest_path = tmp_path / "raw-manifest.json"
    run_builder(builder, monkeypatch, inventory_path, workspace, output, manifest_path)
    return workspace, inventory_path, inventory, payloads, output, manifest_path


def test_archive_planning_separates_datasets_and_enforces_payload_limit(raw_tools):
    builder, _, _ = raw_tools
    files = [{"dataset_id": dataset, "bytes": size, "archive_path": name}
             for dataset, size, name in [("era5", 6, "a"), ("ceres", 4, "b"),
                                         ("ceres", 6, "c"), ("ceres", 1, "d")]]
    plans = builder.plan_archives(files, limit=10)
    assert [(name, dataset, sum(item["bytes"] for item in items))
            for name, dataset, items in plans] == [
        ("raw-ceres-001.zip", "ceres", 10), ("raw-ceres-002.zip", "ceres", 1),
        ("raw-era5-001.zip", "era5", 6)]
    assert all({item["dataset_id"] for item in items} == {dataset}
               for _, dataset, items in plans)
    with pytest.raises(ValueError, match="larger than"):
        builder.plan_archives(files, limit=5)


def test_builder_preserves_source_bytes_and_publishes_only_relative_paths(tmp_path, raw_tools, monkeypatch):
    _, _, inventory, payloads, output, manifest_path = built_fixture(tmp_path, raw_tools, monkeypatch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_files"] == len(payloads)
    assert manifest["source_bytes"] == sum(map(len, payloads.values()))
    assert len(manifest["assets"]) == 2
    extracted = {}
    for asset in manifest["assets"]:
        archive_path = output / asset["name"]
        assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == asset["sha256"]
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                assert member.compress_type == zipfile.ZIP_STORED
                extracted[member.filename] = archive.read(member)
        assert all(record["matches_acquisition_checksum"] for record in asset["files"])
    assert extracted == payloads
    assert all(Path(item["source_absolute"]).read_bytes() == payloads[item["archive_path"]]
               for item in inventory["selected_files"])
    assert "source_absolute" not in manifest_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")


def test_builder_rejects_recorded_acquisition_hash_mismatch(tmp_path, raw_tools, monkeypatch):
    builder, _, _ = raw_tools
    workspace, inventory_path, inventory, _ = inventory_fixture(tmp_path)
    inventory["selected_files"][0]["recorded_sha256"] = "0" * 64
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    output, manifest = tmp_path / "release" / "archives", tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="Original acquisition checksum mismatch"):
        run_builder(builder, monkeypatch, inventory_path, workspace, output, manifest)
    assert not manifest.exists()
    assert not list(output.glob("*.zip"))


def test_matching_resume_keeps_completed_archive_bytes_and_mtime(tmp_path, raw_tools, monkeypatch):
    builder, _, _ = raw_tools
    workspace, inventory_path, _, _, output, manifest = built_fixture(tmp_path, raw_tools, monkeypatch)
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.glob("*.zip")}
    run_builder(builder, monkeypatch, inventory_path, workspace, output, manifest, "--resume")
    assert before == {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.glob("*.zip")}


@pytest.mark.parametrize("change", ["recorded_hash", "archive_path", "source_mtime", "archive_bytes"])
def test_resume_rejects_changed_source_inventory_or_completed_archive(tmp_path, raw_tools, monkeypatch, change):
    builder, _, _ = raw_tools
    workspace, inventory_path, inventory, _, output, manifest = built_fixture(tmp_path, raw_tools, monkeypatch)
    prior_manifest = manifest.read_bytes()
    item = inventory["selected_files"][0]
    if change == "recorded_hash":
        item["recorded_sha256"] = "0" * 64
    elif change == "archive_path":
        item["archive_path"] = "supplementary_data/ceres-test/renamed.nc"
    elif change == "source_mtime":
        source = Path(item["source_absolute"])
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    else:
        archive = output / "raw-ceres-test-001.zip"
        content = bytearray(archive.read_bytes())
        content[10] ^= 1
        archive.write_bytes(content)
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ValueError, match="Existing archive or source identity changed|Complete archive plan changed"):
        run_builder(builder, monkeypatch, inventory_path, workspace, output, manifest, "--resume")
    assert manifest.read_bytes() == prior_manifest


def test_resume_rejects_removing_an_entire_dataset_from_frozen_plan(tmp_path, raw_tools, monkeypatch):
    builder, _, _ = raw_tools
    workspace, inventory_path, inventory, _, output, manifest = built_fixture(tmp_path, raw_tools, monkeypatch)
    prior_manifest = manifest.read_bytes()
    inventory["selected_files"] = [item for item in inventory["selected_files"]
                                   if item["dataset_id"] != "era5-test"]
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ValueError, match="Complete archive plan changed"):
        run_builder(builder, monkeypatch, inventory_path, workspace, output, manifest, "--resume")
    assert manifest.read_bytes() == prior_manifest
    assert (output / "raw-era5-test-001.zip").exists()


@pytest.mark.parametrize("change", ["outside_workspace", "duplicate_archive_path", "unsafe_archive_path", "size"])
def test_inventory_rejects_unsafe_or_changed_sources(tmp_path, raw_tools, change):
    builder, _, _ = raw_tools
    workspace, inventory_path, inventory, _ = inventory_fixture(tmp_path)
    item = inventory["selected_files"][0]
    if change == "outside_workspace":
        outside = tmp_path / "outside.nc"
        outside.write_bytes(Path(item["source_absolute"]).read_bytes())
        item["source_absolute"] = str(outside)
    elif change == "duplicate_archive_path":
        inventory["selected_files"].append(copy.deepcopy(item))
    elif change == "unsafe_archive_path":
        item["archive_path"] = "../outside.nc"
    else:
        item["bytes"] += 1
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ValueError):
        builder.load_inventory([inventory_path], workspace)


def test_raw_manifest_requires_explicit_download_selection_and_preserves_core_default(raw_tools):
    _, verify, _ = raw_tools
    raw = {"assets": [{"name": "raw-ceres-001.zip", "dataset_id": "ceres"},
                      {"name": "raw-era5-001.zip", "dataset_id": "era5"}]}
    with pytest.raises(ValueError, match="requires --dataset"):
        verify.select_assets(raw)
    assert verify.select_assets(raw, datasets=["ceres"]) == [raw["assets"][0]]
    assert verify.select_assets(raw, names=["all"]) == raw["assets"]
    core = {"assets": [{"name": "supplementary-results.zip"}, {"name": "core-inputs.zip"}]}
    assert verify.select_assets(core) == [core["assets"][1]]
    assert verify.select_assets(core, default_all=True) == core["assets"]
    with pytest.raises(ValueError, match="Unknown dataset"):
        verify.select_assets(raw, datasets=["missing"])
    with pytest.raises(ValueError, match="either --asset or --dataset"):
        verify.select_assets(raw, names=["all"], datasets=["ceres"])


def test_dataset_download_and_selected_verification_roundtrip(tmp_path, raw_tools, monkeypatch, capsys):
    _, verify, download = raw_tools
    _, _, _, payloads, output, manifest = built_fixture(tmp_path, raw_tools, monkeypatch)
    destination, cache = tmp_path / "extracted", tmp_path / "cache"
    monkeypatch.setattr(sys, "argv", ["download_artifacts.py", "--manifest", str(manifest),
                                    "--dataset", "ceres-test", "--from-directory", str(output),
                                    "--output", str(destination), "--cache", str(cache)])
    download.main()
    assert {path.relative_to(destination).as_posix(): path.read_bytes()
            for path in destination.rglob("*") if path.is_file()} == {
        path: payload for path, payload in payloads.items() if "ceres-test" in path}
    repository = tmp_path / "repository"
    (repository / "artifacts").mkdir(parents=True)
    (repository / "artifacts" / "repository-manifest.json").write_text(
        json.dumps({"files": []}), encoding="utf-8")
    monkeypatch.setattr(verify, "ROOT", repository)
    monkeypatch.setattr(sys, "argv", ["verify_artifacts.py", "--manifest", str(manifest),
                                    "--dataset", "ceres-test", "--data-root", str(destination)])
    capsys.readouterr()
    assert verify.main() is False
    report = json.loads(capsys.readouterr().out)
    assert report["release_files_verified"] == 2
    assert report["errors"] == []
    assert not (destination / "supplementary_data" / "era5-test").exists()


def test_listing_raw_manifest_makes_no_requests_or_output_directories(tmp_path, raw_tools, monkeypatch, capsys):
    _, _, download = raw_tools
    _, _, _, _, _, manifest = built_fixture(tmp_path, raw_tools, monkeypatch)
    destination, cache = tmp_path / "extracted", tmp_path / "cache"
    monkeypatch.setattr(sys, "argv", ["download_artifacts.py", "--manifest", str(manifest),
                                    "--list", "--output", str(destination), "--cache", str(cache)])
    capsys.readouterr()
    download.main()
    listing = json.loads(capsys.readouterr().out)
    assert set(listing) == {"ceres-test", "era5-test"}
    assert listing["ceres-test"]["files"] == 2
    assert listing["era5-test"]["archives"] == 1
    assert not destination.exists()
    assert not cache.exists()


def test_unselected_raw_cli_fails_before_any_requests_or_cache_creation(tmp_path, raw_tools, monkeypatch):
    _, _, download = raw_tools
    _, _, _, _, _, manifest = built_fixture(tmp_path, raw_tools, monkeypatch)
    cache = tmp_path / "cache"
    monkeypatch.setattr(sys, "argv", ["download_artifacts.py", "--manifest", str(manifest),
                                    "--cache", str(cache)])
    with pytest.raises(SystemExit) as error:
        download.main()
    assert error.value.code == 2
    assert not cache.exists()


class FakeHTTPResponse(io.BytesIO):
    def __init__(self, payload, *, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {}


def cache_fixture(tmp_path):
    payload = b"PK\x03\x04fixture bytes preserved exactly\x00\xff"
    target = tmp_path / "raw-fixture-001.zip"
    asset = {"name": target.name, "bytes": len(payload),
             "sha256": hashlib.sha256(payload).hexdigest(),
             "url": "https://example.invalid/raw-fixture-001.zip"}
    return payload, target, target.with_name(target.name + ".part"), asset


def test_http_206_resumes_correct_range_and_verifies_completed_bytes(tmp_path, raw_tools, monkeypatch):
    _, _, download = raw_tools
    payload, target, partial, asset = cache_fixture(tmp_path)
    partial.write_bytes(payload[:7])
    requests = []

    def fake_open(request, **_kwargs):
        requests.append(request)
        return FakeHTTPResponse(payload[7:], status=206,
                                headers={"Content-Range": f"bytes 7-{len(payload)-1}/{len(payload)}"})

    monkeypatch.setattr(download.urllib.request, "urlopen", fake_open)
    download.download_to_cache(asset, target)
    assert len(requests) == 1
    assert requests[0].get_header("Range") == "bytes=7-"
    assert target.read_bytes() == payload
    assert not partial.exists()


@pytest.mark.parametrize("range_header", ["", "bytes 0-34/35", "bytes 7-99/*", "bytes 7-98/100"])
def test_http_206_rejects_invalid_range_without_touching_existing_partial(
        tmp_path, raw_tools, monkeypatch, range_header):
    _, _, download = raw_tools
    payload, target, partial, asset = cache_fixture(tmp_path)
    partial.write_bytes(payload[:7])
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *_args, **_kwargs:
                        FakeHTTPResponse(payload[7:], status=206,
                                         headers={"Content-Range": range_header}))
    with pytest.raises(ValueError, match="unexpected download byte range"):
        download.download_to_cache(asset, target)
    assert partial.read_bytes() == payload[:7]
    assert not target.exists()


def test_http_200_restarts_when_server_ignores_range(tmp_path, raw_tools, monkeypatch):
    _, _, download = raw_tools
    payload, target, partial, asset = cache_fixture(tmp_path)
    partial.write_bytes(b"stale partial")

    def fake_open(request, **_kwargs):
        assert request.get_header("Range") == "bytes=13-"
        return FakeHTTPResponse(payload, status=200)

    monkeypatch.setattr(download.urllib.request, "urlopen", fake_open)
    download.download_to_cache(asset, target)
    assert target.read_bytes() == payload
    assert not partial.exists()


def test_oversized_http_response_is_rejected_without_publishing_cache(tmp_path, raw_tools, monkeypatch):
    _, _, download = raw_tools
    payload, target, partial, asset = cache_fixture(tmp_path)
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *_args, **_kwargs:
                        FakeHTTPResponse(payload + b"unlisted data", status=200))
    with pytest.raises(ValueError, match="exceeds the manifest byte count"):
        download.download_to_cache(asset, target)
    assert not target.exists()
    assert partial.stat().st_size <= asset["bytes"]


def test_complete_matching_partial_promotes_without_network_request(tmp_path, raw_tools):
    _, _, download = raw_tools
    payload, target, partial, asset = cache_fixture(tmp_path)
    partial.write_bytes(payload)
    download.download_to_cache(asset, target)
    assert target.read_bytes() == payload
    assert not partial.exists()


@pytest.mark.parametrize("corruption", ["wrong_hash", "oversized"])
def test_invalid_complete_partial_is_rejected_without_network_or_publication(tmp_path, raw_tools, corruption):
    _, _, download = raw_tools
    payload, target, partial, asset = cache_fixture(tmp_path)
    partial.write_bytes(b"x" * len(payload) if corruption == "wrong_hash" else payload + b"x")
    before = partial.read_bytes()
    with pytest.raises(ValueError, match="integrity verification|larger than") as error:
        download.download_to_cache(asset, target)
    assert str(partial) in str(error.value)
    assert "Remove only this temporary file and rerun" in str(error.value)
    assert partial.read_bytes() == before
    assert not target.exists()


@pytest.mark.parametrize("resume", [False, True])
def test_builder_rebuilds_interrupted_partial_only_with_resume(tmp_path, raw_tools, resume):
    builder, _, _ = raw_tools
    workspace, inventory_path, _, payloads = inventory_fixture(tmp_path)
    files, _ = builder.load_inventory([inventory_path], workspace)
    name, dataset, items = builder.plan_archives(files)[0]
    output, state = tmp_path / "archives", tmp_path / "state"
    output.mkdir()
    state.mkdir()
    partial = output / (name + ".partial")
    partial.write_bytes(b"interrupted ZIP header")
    if not resume:
        with pytest.raises(FileExistsError, match="use --resume"):
            builder.build_one(name, dataset, items, output, state, resume)
        assert partial.read_bytes() == b"interrupted ZIP header"
        assert not (output / name).exists()
        return
    asset = builder.build_one(name, dataset, items, output, state, resume)
    builder.verify_zip(output / name, asset["files"])
    assert not partial.exists()
    assert (state / (name + ".json")).is_file()
    with zipfile.ZipFile(output / name) as archive:
        assert {member: archive.read(member) for member in archive.namelist()} == {
            path: payload for path, payload in payloads.items() if "ceres-test" in path}


def test_interrupted_extraction_rewrites_partial_and_checks_completed_files(tmp_path, raw_tools, monkeypatch):
    _, verify, download = raw_tools
    _, _, _, payloads, output, manifest_path = built_fixture(tmp_path, raw_tools, monkeypatch)
    asset = json.loads(manifest_path.read_text(encoding="utf-8"))["assets"][0]
    destination = tmp_path / "extracted"
    target = destination / asset["files"][0]["path"]
    partial = target.with_name(target.name + ".part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"incomplete extraction")
    download.extract_verified(output / asset["name"], asset, destination)
    assert not partial.exists()
    assert target.read_bytes() == payloads[asset["files"][0]["path"]]
    assert verify.check_records(destination, asset["files"]) == (2, [])


@pytest.mark.parametrize("stage", ["cache", "extract", "build"])
def test_recovery_refuses_temporary_paths_with_symlink_metadata(tmp_path, raw_tools, monkeypatch, stage):
    builder, _, download = raw_tools
    if stage == "cache":
        _, target, partial, asset = cache_fixture(tmp_path)
        operation = lambda: download.download_to_cache(asset, target)
    else:
        workspace, inventory_path, _, _ = inventory_fixture(tmp_path)
        files, _ = builder.load_inventory([inventory_path], workspace)
        name, dataset, items = builder.plan_archives(files)[0]
        output, state = tmp_path / "archives", tmp_path / "state"
        output.mkdir()
        state.mkdir()
        if stage == "build":
            target = output / name
            partial = output / (name + ".partial")
            operation = lambda: builder.build_one(name, dataset, items, output, state, True)
        else:
            asset = builder.build_one(name, dataset, items, output, state, False)
            destination = tmp_path / "extracted"
            target = destination / asset["files"][0]["path"]
            partial = target.with_name(target.name + ".part")
            partial.parent.mkdir(parents=True)
            operation = lambda: download.extract_verified(output / name, asset, destination)
    partial.write_bytes(b"must not truncate or follow")
    original_lstat = Path.lstat

    def link_metadata(path):
        info = original_lstat(path)
        if path == partial:
            # Exercise link rejection on Windows without requiring symlink privileges.
            return os.stat_result((stat.S_IFLNK | 0o777, *tuple(info)[1:]))
        return info

    monkeypatch.setattr(Path, "lstat", link_metadata)
    with pytest.raises(ValueError, match="symlink temporary"):
        operation()
    assert partial.read_bytes() == b"must not truncate or follow"
    assert not target.exists()


def test_cache_promotion_preserves_final_file_created_during_download(tmp_path, raw_tools, monkeypatch):
    _, _, download = raw_tools
    payload, target, partial, asset = cache_fixture(tmp_path)

    def response(*_args, **_kwargs):
        target.write_bytes(b"another process's existing result")
        return FakeHTTPResponse(payload)

    monkeypatch.setattr(download.urllib.request, "urlopen", response)
    with pytest.raises(FileExistsError, match="existing final file"):
        download.download_to_cache(asset, target)
    assert target.read_bytes() == b"another process's existing result"
    assert partial.read_bytes() == payload


def test_rebuilt_archive_does_not_replace_final_file_created_during_verification(tmp_path, raw_tools, monkeypatch):
    builder, _, _ = raw_tools
    workspace, inventory_path, _, _ = inventory_fixture(tmp_path)
    files, _ = builder.load_inventory([inventory_path], workspace)
    name, dataset, items = builder.plan_archives(files)[0]
    output, state = tmp_path / "archives", tmp_path / "state"
    output.mkdir()
    state.mkdir()
    partial, target = output / (name + ".partial"), output / name
    partial.write_bytes(b"interrupted archive")
    original_verify = builder.verify_zip

    def verify_and_create_final(path, records):
        original_verify(path, records)
        target.write_bytes(b"another process's completed archive")

    monkeypatch.setattr(builder, "verify_zip", verify_and_create_final)
    with pytest.raises(FileExistsError, match="existing final file"):
        builder.build_one(name, dataset, items, output, state, True)
    assert target.read_bytes() == b"another process's completed archive"
    assert partial.is_file()
    assert not (state / (name + ".json")).exists()
