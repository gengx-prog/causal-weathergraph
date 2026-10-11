"""Record Python and dependency versions without querying the host."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import sys

PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "networkx",
    "matplotlib", "PyYAML", "xarray", "netCDF4", "zarr", "tigramite",
    "psutil", "requests", "threadpoolctl", "ortools", "pytest", "joblib",
    "tqdm", "torch", "cdsapi", "h5netcdf", "pyarrow",
    "PyMuPDF", "Pillow", "pip", "setuptools",
)


def collect() -> dict:
    """Read distribution metadata only; optional packages are not imported."""
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": 3,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_scope": "python_and_dependency_versions_only",
        "python": {"version": ".".join(map(str, sys.version_info[:3])),
                   "implementation": sys.implementation.name},
        "packages": packages,
        "limitations": ["Uninstalled optional dependencies are recorded as null.",
                        "This records software visible to the current interpreter, not a historical run."],
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
