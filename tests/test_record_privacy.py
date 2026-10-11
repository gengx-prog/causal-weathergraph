"""Public software inventories must not query or expose host characteristics."""
from __future__ import annotations

import importlib
from pathlib import Path
import platform
import socket
import subprocess
import os


def test_collector_uses_distribution_metadata_without_host_queries(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    collector = importlib.import_module("collect_environment")

    def forbidden(*args, **kwargs):
        raise AssertionError("Host information must not be queried")

    for name in ("uname", "platform", "system", "release", "version", "machine",
                 "processor", "python_compiler"):
        monkeypatch.setattr(platform, name, forbidden)
    monkeypatch.setattr(socket, "gethostname", forbidden)
    monkeypatch.setattr(os, "cpu_count", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    monkeypatch.setattr(collector.metadata, "version", lambda name: "1.2.3")

    result = collector.collect()
    assert set(result) == {"schema_version", "collected_at_utc", "record_scope",
                           "python", "packages", "limitations"}
    assert set(result["python"]) == {"version", "implementation"}
    assert result["record_scope"] == "python_and_dependency_versions_only"
    assert result["packages"] == dict.fromkeys(collector.PACKAGES, "1.2.3")


def test_export_removes_host_fields_without_changing_scientific_values(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    privacy = importlib.import_module("sanitize_records")
    original = {"cpu_model": "private model", "platform": {"system": "synthetic", "version": "0"}, "device": "cuda",
                "shape": [68668, 66, 4], "effect": -0.0042, "elapsed_seconds": 12.5,
                "python": "3.13.7 (compiler and architecture metadata)",
                "source": "C:/private-work/outputs/model/result.csv"}
    result = privacy.sanitize(original)
    assert result == {"device": "cuda", "shape": [68668, 66, 4], "effect": -0.0042,
                      "elapsed_seconds": 12.5, "python": "3.13.7",
                      "source": "outputs/model/result.csv"}
    junit = tmp_path / "test.xml"
    junit.write_text('<testsuite hostname="private-name" tests="1" failures="0" time="1.25">'
                     '<testcase file="C:/private-work/tests/test_example.py" time="1.25"/>'
                     '</testsuite>', encoding="utf-8")
    privacy.sanitize_file(junit)
    import xml.etree.ElementTree as ET
    suite = ET.fromstring(junit.read_text(encoding="utf-8"))
    assert suite.attrib == {"tests": "1", "failures": "0", "time": "1.25"}
    assert suite[0].attrib == {"file": "tests/test_example.py", "time": "1.25"}


def test_distinct_source_records_survive_path_redaction(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    privacy = importlib.import_module("sanitize_records")
    result = privacy.sanitize({"C:/private-a/data.json": "hash-a",
                               "D:/private-b/data.json": "hash-b"})
    assert len(result) == 2
    assert sorted(result.values()) == ["hash-a", "hash-b"]


def test_scientific_configuration_and_clean_source_bytes_are_preserved(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    privacy = importlib.import_module("sanitize_records")
    science = {"platform": "Terra/Aqua", "architecture": "transformer",
               "device": "cuda", "cpu": "fallback", "gpu": "optional",
               "shape": [66, 4], "threads": 2}
    assert privacy.sanitize(science) == science
    frozen = tmp_path / "frozen.json"
    frozen.write_bytes(b'{"platform":"Terra/Aqua","architecture":"MLP","value":1.20}\r\n')
    original = frozen.read_bytes()
    assert privacy.sanitize_file(frozen) is False
    assert frozen.read_bytes() == original
    xml = tmp_path / "clean.xml"
    xml.write_bytes(b'<?xml version="1.0"?><testsuite tests="1"><testcase /></testsuite>\r\n')
    original_xml = xml.read_bytes()
    assert privacy.sanitize_file(xml) is False
    assert xml.read_bytes() == original_xml
    assert privacy.clean_text('C:/private/Python/python.exe --arg X') == '<local>/python.exe --arg X'
