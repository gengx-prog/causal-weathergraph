#!/usr/bin/env python
"""Audit time coverage and 6-hour completeness of local data files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_weathergraph.data_io import SUPPORTED_EXTENSIONS, scan_data_files
from causal_weathergraph.utils import ensure_dir, load_config, resolve_path, setup_logging

TIME_NAMES = ("time", "valid_time", "date", "datetime")
SKIP_NPZ_KEYS = {"lat", "latitude", "lon", "longitude", "timestamps", "time", "times", "dates", "node_ids"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(resolve_path(args.config, REPO_ROOT))
    data_root = Path(args.data_root or config.get("data", {}).get("root", "/home/vipuser/Data")).expanduser()
    records: list[dict[str, Any]] = []
    for path in scan_data_files(data_root):
        records.extend(audit_file(path))
    out_dir = ensure_dir(Path(args.output_root) / "reports")
    out = pd.DataFrame.from_records(
        records,
        columns=[
            "file",
            "variable",
            "time_min",
            "time_max",
            "n_time",
            "expected_steps",
            "missing_steps",
            "duplicated_steps",
        ],
    )
    out.to_csv(out_dir / "data_time_audit.csv", index=False)
    print(f"Saved {len(out)} audit rows to {out_dir / 'data_time_audit.csv'}")


def audit_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return []
    try:
        if suffix in {".nc", ".nc4", ".cdf", ".zarr"}:
            return audit_xarray(path)
        if suffix == ".npz":
            return audit_npz(path)
        if suffix == ".npy":
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            return [empty_time_record(path, path.stem, int(arr.shape[0]) if arr.ndim else 0)]
        if suffix in {".csv", ".parquet"}:
            return audit_table(path)
    except Exception as exc:
        return [
            {
                "file": str(path),
                "variable": f"ERROR: {exc!r}",
                "time_min": "",
                "time_max": "",
                "n_time": np.nan,
                "expected_steps": np.nan,
                "missing_steps": np.nan,
                "duplicated_steps": np.nan,
            }
        ]
    return []


def audit_xarray(path: Path) -> list[dict[str, Any]]:
    import xarray as xr

    opener = xr.open_zarr if path.suffix.lower() == ".zarr" else xr.open_dataset
    ds = opener(path)
    try:
        time_name = detect_name(list(ds.dims) + list(ds.coords), TIME_NAMES)
        if time_name is None or time_name not in ds:
            return [empty_time_record(path, "dataset", 0)]
        times = pd.to_datetime(ds[time_name].values)
        records = []
        for name, da in ds.data_vars.items():
            if time_name not in da.dims:
                continue
            records.append(time_record(path, str(name), times[: int(da.sizes[time_name])]))
        if not records:
            records.append(time_record(path, "dataset", times))
        return records
    finally:
        ds.close()


def audit_npz(path: Path) -> list[dict[str, Any]]:
    records = []
    with np.load(path, allow_pickle=True) as npz:
        time_key = next((key for key in ("timestamps", "time", "times", "dates") if key in npz.files), None)
        times = pd.to_datetime(npz[time_key]) if time_key else None
        variable_names = [str(v) for v in npz["variable_names"].tolist()] if "variable_names" in npz.files else []
        data_key = next((key for key in ("data", "X", "x", "array", "values") if key in npz.files), None)
        if data_key and variable_names:
            arr = npz[data_key]
            n_time = int(arr.shape[0]) if arr.ndim else 0
            for name in variable_names:
                records.append(time_record(path, name, times[:n_time] if times is not None else None, n_time))
            return records
        for key in npz.files:
            if key in SKIP_NPZ_KEYS or key == "variable_names":
                continue
            arr = npz[key]
            n_time = int(arr.shape[0]) if arr.ndim else 0
            records.append(time_record(path, key, times[:n_time] if times is not None else None, n_time))
    return records or [empty_time_record(path, "dataset", 0)]


def audit_table(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    time_name = detect_name(df.columns, TIME_NAMES)
    if time_name is None:
        return [empty_time_record(path, "table", len(df))]
    times = pd.to_datetime(df[time_name])
    variables = [c for c in df.columns if c != time_name]
    if not variables:
        variables = ["table"]
    return [time_record(path, str(variable), times) for variable in variables]


def time_record(path: Path, variable: str, times: pd.DatetimeIndex | pd.Series | None, n_time: int | None = None) -> dict[str, Any]:
    if times is None:
        return empty_time_record(path, variable, int(n_time or 0))
    dt = pd.DatetimeIndex(pd.to_datetime(times)).dropna()
    n = int(len(dt) if n_time is None else n_time)
    if len(dt) == 0:
        return empty_time_record(path, variable, n)
    unique = pd.DatetimeIndex(dt.unique()).sort_values()
    expected = len(pd.date_range(unique.min(), unique.max(), freq="6h"))
    duplicated = int(len(dt) - len(unique))
    missing = int(max(0, expected - len(unique)))
    return {
        "file": str(path),
        "variable": variable,
        "time_min": str(unique.min()),
        "time_max": str(unique.max()),
        "n_time": n,
        "expected_steps": int(expected),
        "missing_steps": missing,
        "duplicated_steps": duplicated,
    }


def empty_time_record(path: Path, variable: str, n_time: int) -> dict[str, Any]:
    return {
        "file": str(path),
        "variable": variable,
        "time_min": "",
        "time_max": "",
        "n_time": int(n_time),
        "expected_steps": np.nan,
        "missing_steps": np.nan,
        "duplicated_steps": np.nan,
    }


def detect_name(names: Any, candidates: tuple[str, ...]) -> str | None:
    normalized = {normalize_name(name): str(name) for name in names}
    for candidate in candidates:
        found = normalized.get(normalize_name(candidate))
        if found is not None:
            return found
    return None


def normalize_name(name: Any) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


if __name__ == "__main__":
    main()
