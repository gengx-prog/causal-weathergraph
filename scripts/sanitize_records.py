"""Remove host metadata and private paths from records before public export.

Run after computation: frozen operational plans can contain paths needed while a
job is running. This export step leaves result values and elapsed times intact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

HOST_FIELDS = frozenset({
    "hostname", "host", "username",
    "os", "computer_product", "cpu_model", "cpu_name", "cpu_count",
    "cpu_and_memory", "logical_processors", "physical_cores", "memory_bytes",
    "ram_bytes", "ram_gb", "nvidia_devices", "gpu_name", "gpu_model",
    "driver_version", "torch_runtime", "compiler", "threadpools",
    "active_threadpools", "peak_working_set_bytes", "process_peak_working_set_bytes",
    "peak_sampled_rss_mb", "rss_mb", "rss_baseline_mib", "sampled_peak_rss_mib",
    "sampled_incremental_rss_mib", "gpu_peak_allocated_bytes",
    "peak_torch_gpu_allocated_bytes", "is_virtual_environment",
    "include_system_site_packages", "thread_environment",
})
_WINDOWS_PATH = re.compile(r"(?<![\w])(?:[A-Za-z]:[\\/])(?:(?!\s+--?\w)[^\r\n\"'<>|])+")
_PRIVATE_POSIX = re.compile(r"(?<![\w:])/(?:home|Users|mnt|tmp|private|opt/hostedtoolcache)/[^\s\"'<>|]+")
_RELATIVE_ANCHORS = ("tests", "revision", "scripts", "visualization", "outputs", "artifacts")


def host_field(key, value):
    """Do not confuse satellite platforms or model architecture with hosts."""
    key = str(key).lower()
    if key in HOST_FIELDS:
        return True
    if key == "platform":
        if isinstance(value, dict):
            return bool({"system", "release", "version", "machine", "architecture"} & value.keys())
        return isinstance(value, str) and bool(re.match(r"^(Windows|Linux|Darwin|macOS)(?:[- /]|$)", value, re.I))
    if key in {"architecture", "machine"}:
        return isinstance(value, str) and value.lower() in {
            "amd64", "x86_64", "x86", "i386", "i686", "aarch64", "arm64", "armv7l"}
    if key in {"cpu", "gpu"}:
        return isinstance(value, str) and bool(re.search(
            r"Intel|GenuineIntel|Xeon|Ryzen|AMD |NVIDIA|GeForce|Radeon|Apple M[0-9]", value, re.I))
    return False


def portable_path(value: str) -> str:
    """Retain a repository-relative suffix, otherwise only the final filename."""
    parts = value.replace("\\", "/").split("/")
    for anchor in _RELATIVE_ANCHORS:
        if anchor in parts:
            return "/".join(parts[parts.index(anchor):])
    return "<local>/" + parts[-1]


def clean_text(value: str) -> str:
    value = _WINDOWS_PATH.sub(lambda m: portable_path(m.group()), value)
    return _PRIVATE_POSIX.sub(lambda m: portable_path(m.group()), value)


def sanitize(value, key=""):
    """Remove only explicit host/resource fields; preserve scientific numbers."""
    if isinstance(value, dict):
        result = {}
        for original_key, item in value.items():
            if host_field(original_key, item):
                continue
            public_key = clean_text(str(original_key))
            candidate, suffix = public_key, 2
            while candidate in result:
                candidate = f"{public_key} [{suffix}]"
                suffix += 1
            result[candidate] = sanitize(item, str(original_key))
        if result.get("record_scope") == "current_interpreter_and_machine_at_collection_time":
            result["record_scope"] = "python_and_dependency_versions_only"
            result.pop("privacy", None)
            result["limitations"] = ["Software metadata retained from the recorded run."]
        return result
    if isinstance(value, list):
        return [sanitize(item, key) for item in value]
    if isinstance(value, str):
        if key == "python" and re.match(r"^\d+\.\d+\.\d+", value):
            return re.match(r"^\d+\.\d+\.\d+", value).group()
        return clean_text(value)
    return value


def sanitize_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        parsed = json.loads(original)
        public = sanitize(parsed)
        if public == parsed:
            return False
        payload = json.dumps(public, indent=2, ensure_ascii=False) + "\n"
    elif path.suffix.lower() == ".xml":
        tree = ET.fromstring(original)
        changed = False
        for node in tree.iter():
            attributes = {key: clean_text(value) for key, value in node.attrib.items() if key != "hostname"}
            changed |= attributes != node.attrib
            node.attrib = attributes
            if node.text:
                changed |= clean_text(node.text) != node.text
                node.text = clean_text(node.text)
            if node.tail:
                changed |= clean_text(node.tail) != node.tail
                node.tail = clean_text(node.tail)
        if not changed:
            return False
        payload = ET.tostring(tree, encoding="unicode") + "\n"
    else:
        payload = clean_text(original)
    if payload == original:
        return False
    path.write_text(payload, encoding="utf-8", newline="\n")
    return True


def sanitize_tree(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(sanitize_file(path) for path in root.rglob("*")
               if path.is_file() and path.suffix.lower() in {".json", ".xml", ".log", ".txt"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Export directory after computation completes")
    args = parser.parse_args()
    print(json.dumps({"records_sanitized": sanitize_tree(args.root)}))


if __name__ == "__main__":
    main()
