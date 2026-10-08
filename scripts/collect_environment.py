"""Write a portable, allowlisted environment record without machine identifiers.

This script uses the standard library. Optional libraries are inspected only when
already installed; it never installs packages or downloads data.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "networkx",
    "matplotlib", "PyYAML", "xarray", "netCDF4", "zarr", "tigramite",
    "psutil", "requests", "threadpoolctl", "ortools", "pytest", "joblib",
    "tqdm", "torch", "cdsapi", "h5netcdf", "pyarrow", "python-docx",
    "bibtexparser", "pylatexenc", "pip", "setuptools",
)


def command_output(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            arguments, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=20, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def windows_inventory() -> dict:
    # Select specific properties: do not collect names, accounts or serial numbers.
    script = (
        "$osInfo=Get-CimInstance Win32_OperatingSystem;"
        "$cpuInfo=Get-CimInstance Win32_Processor;"
        "$memoryInfo=Get-CimInstance Win32_ComputerSystem;"
        "[pscustomobject]@{"
        "os_caption=$osInfo.Caption;os_build=$osInfo.BuildNumber;"
        "cpu_model=($cpuInfo.Name -join ', ');"
        "physical_cores=($cpuInfo.NumberOfCores | Measure-Object -Sum).Sum;"
        "logical_processors=($cpuInfo.NumberOfLogicalProcessors | Measure-Object -Sum).Sum;"
        "ram_bytes=$memoryInfo.TotalPhysicalMemory} | ConvertTo-Json -Compress"
    )
    raw = command_output(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script])
    try:
        return json.loads(raw) if raw else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def cpu_and_memory() -> dict:
    result = {
        "model": platform.processor() or None,
        "physical_cores": None,
        "logical_processors": os.cpu_count(),
        "ram_bytes": None,
    }
    if sys.platform.startswith("linux"):
        try:
            cpu_text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            model = next((line.split(":", 1)[1].strip() for line in cpu_text.splitlines()
                          if line.startswith("model name")), None)
            result["model"] = model or result["model"]
            memory = Path("/proc/meminfo").read_text(encoding="utf-8")
            total = next(line for line in memory.splitlines() if line.startswith("MemTotal:"))
            result["ram_bytes"] = int(total.split()[1]) * 1024
            core_ids = set()
            for block in cpu_text.split("\n\n"):
                fields = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
                fields = {key.strip(): value.strip() for key, value in fields.items()}
                if "physical id" in fields and "core id" in fields:
                    core_ids.add((fields["physical id"], fields["core id"]))
            result["physical_cores"] = len(core_ids) or None
        except (OSError, StopIteration, ValueError):
            pass
    elif sys.platform == "darwin":
        for key, field in (("machdep.cpu.brand_string", "model"),
                           ("hw.physicalcpu", "physical_cores"),
                           ("hw.memsize", "ram_bytes")):
            value = command_output(["sysctl", "-n", key])
            if value:
                result[field] = value if field == "model" else int(value)
    return result


def nvidia_devices() -> list[dict]:
    raw = command_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not raw:
        return []
    devices = []
    for row in csv.reader(io.StringIO(raw)):
        if len(row) != 3:
            continue
        model, driver, memory = (part.strip() for part in row)
        devices.append({"model": model, "driver_version": driver,
                        "memory_mib": int(memory) if memory.isdigit() else None})
    return devices


def collect() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    hardware = cpu_and_memory()
    os_info = {"system": platform.system(), "release": platform.release(),
               "version": platform.version(), "architecture": platform.machine()}
    if os.name == "nt":
        details = windows_inventory()
        hardware.update({"model": details.get("cpu_model", hardware["model"]),
                         "physical_cores": details.get("physical_cores"),
                         "logical_processors": details.get("logical_processors", os.cpu_count()),
                         "ram_bytes": details.get("ram_bytes")})
        os_info.update({"caption": details.get("os_caption"), "build": details.get("os_build")})
    if hardware["ram_bytes"] is not None:
        hardware["ram_gib"] = round(hardware["ram_bytes"] / 1024 ** 3, 2)
    include_system = None
    config_path = Path(sys.prefix) / "pyvenv.cfg"
    if config_path.is_file():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("include-system-site-packages"):
                include_system = line.partition("=")[2].strip().lower() == "true"
    torch_info = {"installed": packages["torch"] is not None}
    if torch_info["installed"]:
        try:
            import torch
            torch_info.update({"cuda_build": torch.version.cuda,
                               "cuda_available": torch.cuda.is_available(),
                               "cudnn_version": torch.backends.cudnn.version()})
        except Exception as exc:
            # The exception text can contain local paths. Record only its class.
            torch_info["inspection_error_type"] = type(exc).__name__
    thread_settings = {}
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED", "PYTHONUTF8"):
        value = os.environ.get(name)
        thread_settings[name] = value if value is None or value.isdecimal() or value == "random" else "other"
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_scope": "current_interpreter_and_machine_at_collection_time",
        "privacy": "Allowlisted fields only; excludes hostname, username, paths, serials, network identifiers and credentials.",
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                   "compiler": platform.python_compiler(), "is_virtual_environment": sys.prefix != sys.base_prefix,
                   "include_system_site_packages": include_system},
        "os": os_info,
        "cpu_and_memory": hardware,
        "nvidia_devices": nvidia_devices(),
        "torch_runtime": torch_info,
        "packages": packages,
        "thread_environment": thread_settings,
        "limitations": ["Missing optional tools or packages are recorded as null or an empty list.",
                        "Hardware inventory is best effort; GPU inventory lists NVIDIA devices exposed by nvidia-smi.",
                        "This is an observation of the current environment, not evidence of a historical run."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="JSON output file; default: print JSON to stdout")
    args = parser.parse_args()
    payload = json.dumps(collect(), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
