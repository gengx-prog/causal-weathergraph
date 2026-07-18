"""Flexible data discovery and loading utilities.

The loader intentionally supports several common layouts instead of assuming a
single reanalysis file convention. It returns an in-memory array with shape
``[time, node, variable]``; region aggregation is then used to keep causal
discovery computationally modest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .utils import canonicalize_variable_request, get_logger

logger = get_logger("data_io")

SUPPORTED_EXTENSIONS = {
    ".nc",
    ".nc4",
    ".cdf",
    ".zarr",
    ".npy",
    ".npz",
    ".csv",
    ".parquet",
}

VARIABLE_ALIASES: dict[str, tuple[str, ...]] = {
    "temperature": (
        "temperature",
        "temp",
        "t",
        "t2m",
        "t850",
        "air_temperature",
        "2m_temperature",
    ),
    "humidity": (
        "specific_humidity",
        "humidity",
        "q",
        "q850",
        "rh",
        "relative_humidity",
    ),
    "wind": ("wind", "wind_speed", "windspeed", "speed", "wspd"),
    "cloud_cover": (
        "cloud_cover",
        "total_cloud_cover",
        "tcc",
        "cloud",
        "cc",
        "cloud_fraction",
    ),
}

WIND_SPEED_ALIASES = ("wind_speed", "windspeed", "wspd", "speed")
U_WIND_ALIASES = ("u", "u10", "u850", "u_component", "u_wind", "eastward_wind")
V_WIND_ALIASES = ("v", "v10", "v850", "v_component", "v_wind", "northward_wind")

TIME_NAMES = ("time", "valid_time", "date", "datetime")
LAT_NAMES = ("lat", "latitude", "y", "nav_lat")
LON_NAMES = ("lon", "longitude", "x", "nav_lon")
NODE_NAMES = ("node", "nodes", "point", "points", "gridpoint", "location", "station")


@dataclass
class AtmosphericDataset:
    """Unified atmospheric data container.

    Attributes:
        data: Float array with shape ``[time, node, variable]``.
        variable_names: Canonical variable names aligned with ``data`` axis 2.
        lat: Node latitudes in degrees, shape ``[node]``. May contain NaNs.
        lon: Node longitudes in degrees, shape ``[node]``. May contain NaNs.
        timestamps: Optional datetime index aligned with the time axis.
        node_ids: Optional stable node identifiers.
        metadata: Source metadata useful for reproducibility.
    """

    data: np.ndarray
    variable_names: list[str]
    lat: np.ndarray
    lon: np.ndarray
    timestamps: pd.DatetimeIndex | None = None
    node_ids: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 3:
            raise ValueError("AtmosphericDataset.data must have shape [time, node, variable].")
        self.lat = np.asarray(self.lat, dtype=float).reshape(-1)
        self.lon = np.asarray(self.lon, dtype=float).reshape(-1)
        if self.lat.size != self.data.shape[1] or self.lon.size != self.data.shape[1]:
            raise ValueError("Latitude/longitude arrays must match the node dimension.")
        if len(self.variable_names) != self.data.shape[2]:
            raise ValueError("variable_names must match the variable dimension.")
        if self.node_ids is None:
            self.node_ids = np.arange(self.data.shape[1])

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(time, node, variable)`` shape."""
        return self.data.shape

    def variable_index(self, name: str) -> int | None:
        """Return the index for a canonical variable name, or None."""
        canonical = canonicalize_variable_request(name)
        try:
            return self.variable_names.index(canonical)
        except ValueError:
            return None

    def subset_variables(self, names: Iterable[str]) -> "AtmosphericDataset":
        """Return a dataset containing only requested variables that are present."""
        wanted = [canonicalize_variable_request(v) for v in names]
        indices = [i for i, v in enumerate(self.variable_names) if v in wanted]
        if not indices:
            raise ValueError(f"None of the requested variables are present: {wanted}")
        return AtmosphericDataset(
            data=self.data[:, :, indices],
            variable_names=[self.variable_names[i] for i in indices],
            lat=self.lat,
            lon=self.lon,
            timestamps=self.timestamps,
            node_ids=self.node_ids,
            metadata=dict(self.metadata),
        )


def normalize_name(name: str) -> str:
    """Normalize variable and dimension names for alias matching."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def scan_data_files(data_root: str | Path) -> list[Path]:
    """Recursively find supported data files/directories under ``data_root``."""
    root = Path(data_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    found: list[Path] = []
    for path in root.rglob("*"):
        if path.name.startswith("."):
            continue
        if any(parent.suffix == ".zarr" for parent in path.parents):
            continue
        if path.is_dir() and path.suffix.lower() == ".zarr":
            found.append(path)
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(path)
    return sorted(found)


def inspect_data_root(data_root: str | Path) -> dict[str, Any]:
    """Inspect supported files and return JSON-serializable metadata."""
    files = scan_data_files(data_root)
    summary = {
        "data_root": str(Path(data_root).expanduser()),
        "n_supported_files": len(files),
        "files": [],
    }
    for path in files:
        summary["files"].append(inspect_file(path))
    return summary


def inspect_file(path: str | Path) -> dict[str, Any]:
    """Inspect one supported data file without loading more than necessary."""
    path = Path(path)
    record: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "format": path.suffix.lower().lstrip("."),
        "size_bytes": None if path.is_dir() else path.stat().st_size,
    }
    try:
        suffix = path.suffix.lower()
        if suffix in {".nc", ".nc4", ".cdf", ".zarr"}:
            ds = _open_xarray(path)
            record.update(_inspect_xarray_dataset(ds))
            ds.close()
        elif suffix in {".npy", ".npz"}:
            record.update(_inspect_numpy_file(path))
        elif suffix in {".csv", ".parquet"}:
            record.update(_inspect_table_file(path))
    except Exception as exc:  # pragma: no cover - intentionally defensive
        record["error"] = repr(exc)
    return record


def load_atmospheric_data(
    data_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> AtmosphericDataset:
    """Load local atmospheric data into ``[time, node, variable]`` format.

    The function scans the local data root, tries every supported file, and
    chooses the richest compatible combination of variables. It never downloads
    data.
    """
    config = config or {}
    if data_root is None:
        data_root = config.get("root", "/home/vipuser/Data")
    target_variables = [
        canonicalize_variable_request(v)
        for v in config.get("variables", ["temperature", "humidity", "wind", "cloud_cover"])
    ]
    files = scan_data_files(data_root)
    if not files:
        raise FileNotFoundError(f"No supported data files found under {data_root}")

    partials: list[AtmosphericDataset] = []
    for path in files:
        loaded = _try_load_file(path, target_variables, config)
        if loaded is not None and loaded.data.shape[2] > 0:
            partials.append(loaded)

    if not partials:
        raise RuntimeError(
            "Supported files were found, but no requested atmospheric variables could be loaded. "
            f"Requested variables: {target_variables}"
        )

    merged = _merge_partial_datasets(partials, target_variables, config)
    missing = [v for v in target_variables if v not in merged.variable_names]
    if missing:
        logger.warning("Requested variables not found or not compatible: %s", missing)
    logger.info(
        "Loaded data with shape [time=%d, node=%d, variable=%d] and variables=%s",
        merged.data.shape[0],
        merged.data.shape[1],
        merged.data.shape[2],
        merged.variable_names,
    )
    return merged


def save_dataset_npz(dataset: AtmosphericDataset, path: str | Path) -> None:
    """Save a unified dataset to a compressed NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = (
        dataset.timestamps.astype("datetime64[ns]").astype(str).to_numpy()
        if dataset.timestamps is not None
        else np.array([], dtype=str)
    )
    np.savez_compressed(
        path,
        data=dataset.data,
        variable_names=np.array(dataset.variable_names, dtype=str),
        lat=dataset.lat,
        lon=dataset.lon,
        timestamps=timestamps,
        node_ids=np.asarray(dataset.node_ids),
    )


def load_dataset_npz(path: str | Path) -> AtmosphericDataset:
    """Load a dataset saved by :func:`save_dataset_npz`."""
    with np.load(path, allow_pickle=True) as npz:
        timestamps_raw = npz.get("timestamps", np.array([], dtype=str))
        timestamps = None
        if timestamps_raw.size:
            timestamps = pd.to_datetime(timestamps_raw)
        return AtmosphericDataset(
            data=npz["data"],
            variable_names=[str(v) for v in npz["variable_names"].tolist()],
            lat=npz["lat"],
            lon=npz["lon"],
            timestamps=timestamps,
            node_ids=npz.get("node_ids", None),
            metadata={"source": str(path)},
        )


def _open_xarray(path: Path):
    import xarray as xr

    if path.suffix.lower() == ".zarr":
        return xr.open_zarr(path, consolidated=None)
    return xr.open_dataset(path)


def _inspect_xarray_dataset(ds: Any) -> dict[str, Any]:
    dims = {str(k): int(v) for k, v in ds.sizes.items()}
    data_vars = {}
    for name, da in ds.data_vars.items():
        data_vars[str(name)] = {
            "dims": [str(d) for d in da.dims],
            "shape": [int(s) for s in da.shape],
            "dtype": str(da.dtype),
        }
    out = {
        "dims": dims,
        "coords": [str(c) for c in ds.coords],
        "variables": data_vars,
    }
    time_name = _detect_name(list(ds.dims) + list(ds.coords), TIME_NAMES)
    if time_name and time_name in ds.coords:
        values = ds[time_name].values
        if len(values):
            try:
                times = pd.to_datetime(values)
                out["time_range"] = [str(times.min()), str(times.max())]
            except Exception:
                out["time_range"] = [str(values[0]), str(values[-1])]
    return out


def _inspect_numpy_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return {"arrays": {"data": {"shape": list(arr.shape), "dtype": str(arr.dtype)}}}
    arrays: dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as npz:
        for key in npz.files:
            arr = npz[key]
            arrays[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return {"arrays": arrays}


def _inspect_table_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=1000)
        return {"columns": list(df.columns), "preview_rows": int(len(df))}
    df = pd.read_parquet(path)
    return {"columns": list(df.columns), "rows": int(len(df))}


def _try_load_file(
    path: Path,
    target_variables: list[str],
    config: dict[str, Any],
) -> AtmosphericDataset | None:
    try:
        suffix = path.suffix.lower()
        if suffix in {".nc", ".nc4", ".cdf", ".zarr"}:
            ds = _open_xarray(path)
            try:
                return _dataset_from_xarray(ds, path, target_variables, config)
            finally:
                ds.close()
        if suffix in {".npy", ".npz"}:
            return _dataset_from_numpy(path, target_variables, config)
        if suffix in {".csv", ".parquet"}:
            return _dataset_from_table(path, target_variables, config)
    except Exception as exc:
        logger.debug("Skipping %s because loading failed: %r", path, exc)
    return None


def _dataset_from_xarray(
    ds: Any,
    path: Path,
    target_variables: list[str],
    config: dict[str, Any],
) -> AtmosphericDataset | None:
    time_dim = _detect_name(list(ds.dims) + list(ds.coords), TIME_NAMES)
    if time_dim is None:
        logger.debug("No recognizable time dimension in %s", path)
        return None

    arrays: list[np.ndarray] = []
    names: list[str] = []
    lat_ref: np.ndarray | None = None
    lon_ref: np.ndarray | None = None
    timestamps = _extract_timestamps(ds, time_dim, config)
    wind_component_found: str | None = None

    for canonical in target_variables:
        if canonical == "wind":
            wind_array = _extract_wind_from_xarray(ds, time_dim, config)
            if wind_array is None:
                continue
            arr, lat, lon, wind_component = wind_array
            wind_component_found = wind_component
        else:
            var_name = _find_xarray_var(ds, VARIABLE_ALIASES.get(canonical, (canonical,)))
            if var_name is None:
                continue
            arr, lat, lon = _dataarray_to_time_node(ds[var_name], ds, time_dim, config)
        arrays.append(arr)
        names.append(canonical)
        lat_ref = lat if lat_ref is None else lat_ref
        lon_ref = lon if lon_ref is None else lon_ref

    if not arrays:
        arrays_from_var_dim = _extract_combined_variable_array(ds, time_dim, target_variables, config)
        if arrays_from_var_dim is None:
            return None
        data, names, lat_ref, lon_ref = arrays_from_var_dim
        return AtmosphericDataset(
            data=data,
            variable_names=names,
            lat=lat_ref,
            lon=lon_ref,
            timestamps=timestamps[: data.shape[0]] if timestamps is not None else None,
            metadata={"source_file": str(path), "loader": "xarray_combined"},
        )

    data, lat_ref, lon_ref, timestamps = _align_arrays(arrays, lat_ref, lon_ref, timestamps)
    metadata = {"source_file": str(path), "loader": "xarray"}
    if wind_component_found is not None:
        metadata["wind_component"] = wind_component_found
    return AtmosphericDataset(
        data=data,
        variable_names=names,
        lat=lat_ref,
        lon=lon_ref,
        timestamps=timestamps,
        metadata=metadata,
    )


def _extract_wind_from_xarray(
    ds: Any,
    time_dim: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
    wind_name = _find_xarray_var(ds, WIND_SPEED_ALIASES) or _find_xarray_var_exact(ds, ("wind",))
    if wind_name is not None:
        arr, lat, lon = _dataarray_to_time_node(ds[wind_name], ds, time_dim, config)
        return arr, lat, lon, "speed"
    u_name = _find_xarray_var(ds, U_WIND_ALIASES)
    v_name = _find_xarray_var(ds, V_WIND_ALIASES)
    if u_name is not None and v_name is not None:
        u, lat, lon = _dataarray_to_time_node(ds[u_name], ds, time_dim, config)
        v, _, _ = _dataarray_to_time_node(ds[v_name], ds, time_dim, config)
        n_t = min(u.shape[0], v.shape[0])
        n_n = min(u.shape[1], v.shape[1])
        return np.sqrt(u[:n_t, :n_n] ** 2 + v[:n_t, :n_n] ** 2), lat[:n_n], lon[:n_n], "speed"
    single_component = u_name or v_name
    if single_component is not None:
        arr, lat, lon = _dataarray_to_time_node(ds[single_component], ds, time_dim, config)
        component = "u" if single_component == u_name else "v"
        return arr, lat, lon, component
    return None


def _extract_combined_variable_array(
    ds: Any,
    time_dim: str,
    target_variables: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray] | None:
    variable_dim_names = ("variable", "variables", "var", "channel", "feature")
    for da in ds.data_vars.values():
        var_dim = _detect_name(da.dims, variable_dim_names)
        if var_dim is None or time_dim not in da.dims:
            continue
        coord_values = (
            [str(v) for v in da[var_dim].values.tolist()]
            if var_dim in da.coords
            else [f"var_{i}" for i in range(da.sizes[var_dim])]
        )
        canonical_by_index = [canonicalize_variable_request(v) for v in coord_values]
        keep = [i for i, v in enumerate(canonical_by_index) if v in target_variables]
        if not keep:
            continue
        arrays = []
        names = []
        lat_ref = lon_ref = None
        for idx in keep:
            sub = da.isel({var_dim: idx})
            arr, lat, lon = _dataarray_to_time_node(sub, ds, time_dim, config)
            arrays.append(arr)
            names.append(canonical_by_index[idx])
            lat_ref = lat if lat_ref is None else lat_ref
            lon_ref = lon if lon_ref is None else lon_ref
        data, lat_ref, lon_ref, _ = _align_arrays(arrays, lat_ref, lon_ref, None)
        return data, names, lat_ref, lon_ref
    return None


def _dataarray_to_time_node(
    da: Any,
    ds: Any,
    time_dim: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stride = int(config.get("time_stride") or 1)
    max_t = config.get("max_time_steps")
    indexer: dict[str, Any] = {time_dim: slice(None, None, stride)}
    spatial_subsample = config.get("spatial_subsample")

    da = _select_first_extra_dims(da, time_dim)
    spatial_dims = [d for d in da.dims if d != time_dim]
    if spatial_subsample is not None:
        step = int(spatial_subsample)
        if step > 1:
            for dim in spatial_dims:
                indexer[dim] = slice(None, None, step)
    da = da.isel(indexer)
    if max_t is not None:
        da = da.isel({time_dim: slice(0, int(max_t))})

    spatial_dims = [d for d in da.dims if d != time_dim]
    if not spatial_dims:
        arr = np.asarray(da.transpose(time_dim).values, dtype=float).reshape(-1, 1)
        return arr, np.array([np.nan]), np.array([np.nan])

    transposed = da.transpose(time_dim, *spatial_dims)
    values = np.asarray(transposed.values, dtype=float)
    arr = values.reshape(values.shape[0], -1)
    lat, lon = _lat_lon_for_spatial(ds, transposed, spatial_dims)
    if lat.size != arr.shape[1] or lon.size != arr.shape[1]:
        lat = np.full(arr.shape[1], np.nan)
        lon = np.full(arr.shape[1], np.nan)
    return arr, lat, lon


def _select_first_extra_dims(da: Any, time_dim: str) -> Any:
    """Select the first level from non-spatial high-rank dimensions."""
    if len(da.dims) <= 3:
        return da
    protected = {time_dim}
    protected.update(d for d in da.dims if normalize_name(d) in {normalize_name(n) for n in LAT_NAMES + LON_NAMES + NODE_NAMES})
    for dim in list(da.dims):
        if dim in protected:
            continue
        if len([d for d in da.dims if d != time_dim]) <= 2:
            break
        da = da.isel({dim: 0})
    return da


def _lat_lon_for_spatial(ds: Any, da: Any, spatial_dims: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lat_name = _detect_name(list(ds.coords) + list(ds.variables), LAT_NAMES)
    lon_name = _detect_name(list(ds.coords) + list(ds.variables), LON_NAMES)
    node_dim = spatial_dims[0] if len(spatial_dims) == 1 else None
    spatial_sizes = {d: int(da.sizes[d]) for d in spatial_dims}

    if lat_name and lon_name and lat_name in ds and lon_name in ds:
        try:
            lat_coord = da.coords[lat_name] if lat_name in da.coords else ds[lat_name]
            lon_coord = da.coords[lon_name] if lon_name in da.coords else ds[lon_name]
            lat_values = _coordinate_to_flat(lat_coord, spatial_dims, spatial_sizes)
            lon_values = _coordinate_to_flat(lon_coord, spatial_dims, spatial_sizes)
            if lat_values.size == lon_values.size:
                return lat_values, _normalize_longitudes(lon_values)
        except Exception:
            pass

    if node_dim and node_dim in da.coords:
        n = int(da.sizes[node_dim])
        return np.full(n, np.nan), np.full(n, np.nan)

    n = int(np.prod([da.sizes[d] for d in spatial_dims]))
    return np.full(n, np.nan), np.full(n, np.nan)


def _coordinate_to_flat(coord: Any, spatial_dims: list[str], spatial_sizes: dict[str, int]) -> np.ndarray:
    coord_dims = list(coord.dims)
    if not coord_dims:
        return np.array([float(coord.values)])
    if len(coord_dims) == 1 and len(spatial_dims) >= 2:
        dim = coord_dims[0]
        if dim in spatial_dims:
            arrays = []
            for spatial_dim in spatial_dims:
                if spatial_dim == dim:
                    arrays.append(np.asarray(coord.values))
                else:
                    arrays.append(np.arange(spatial_sizes[spatial_dim]))
            mesh = np.meshgrid(*arrays, indexing="ij")
            idx = spatial_dims.index(dim)
            return np.asarray(mesh[idx], dtype=float).reshape(-1)
    if set(coord_dims).issubset(set(spatial_dims)):
        ordered = coord.transpose(*[d for d in spatial_dims if d in coord_dims])
        values = np.asarray(ordered.values, dtype=float)
        if values.ndim < len(spatial_dims):
            shape = [coord.sizes[d] if d in coord_dims else 1 for d in spatial_dims]
            values = values.reshape(shape)
            values = np.broadcast_to(values, [spatial_sizes[d] for d in spatial_dims])
        return values.reshape(-1)
    return np.asarray(coord.values, dtype=float).reshape(-1)


def _extract_timestamps(ds: Any, time_dim: str, config: dict[str, Any]) -> pd.DatetimeIndex | None:
    if time_dim not in ds.coords:
        return None
    stride = int(config.get("time_stride") or 1)
    max_t = config.get("max_time_steps")
    values = ds[time_dim].isel({time_dim: slice(None, None, stride)}).values
    if max_t is not None:
        values = values[: int(max_t)]
    try:
        return pd.to_datetime(values)
    except Exception:
        return None


def _dataset_from_numpy(
    path: Path,
    target_variables: list[str],
    config: dict[str, Any],
) -> AtmosphericDataset | None:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as npz:
            return _dataset_from_npz(npz, path, target_variables, config)

    arr = np.load(path, allow_pickle=False)
    inferred = _infer_variable_from_filename(path)
    if inferred is None:
        if arr.ndim == 3 and arr.shape[-1] <= 16:
            names = [f"var_{i}" for i in range(arr.shape[-1])]
            data = _apply_time_options(arr, config)
            data = data.reshape(data.shape[0], -1, data.shape[-1])
            n = data.shape[1]
            return AtmosphericDataset(data, names, np.full(n, np.nan), np.full(n, np.nan), metadata={"source_file": str(path)})
        return None
    if inferred not in target_variables:
        return None
    data_2d = _array_to_time_node(arr, config)
    n = data_2d.shape[1]
    metadata = {"source_file": str(path), "loader": "npy"}
    if inferred == "wind":
        metadata["wind_component"] = _infer_wind_component_from_name(path.name) or "speed"
    return AtmosphericDataset(
        data=data_2d[:, :, None],
        variable_names=[inferred],
        lat=np.full(n, np.nan),
        lon=np.full(n, np.nan),
        metadata=metadata,
    )


def _dataset_from_npz(
    npz: Any,
    path: Path,
    target_variables: list[str],
    config: dict[str, Any],
) -> AtmosphericDataset | None:
    keys = list(npz.files)
    lat = _optional_npz_array(npz, ("lat", "latitude"))
    lon = _optional_npz_array(npz, ("lon", "longitude"))
    timestamps = _optional_timestamps_npz(npz)

    variable_names = _optional_variable_names_npz(npz)
    data_key = next((k for k in ("data", "X", "x", "array", "values") if k in keys), None)
    if data_key is not None:
        arr = np.asarray(npz[data_key], dtype=float)
        if arr.ndim == 3 and variable_names and len(variable_names) in arr.shape:
            data, names = _combined_numpy_array(arr, variable_names, target_variables, config)
            if data is not None:
                n = data.shape[1]
                lat, lon = _lat_lon_from_optional(lat, lon, n)
                return AtmosphericDataset(data, names, lat, lon, timestamps=timestamps, metadata={"source_file": str(path), "loader": "npz_combined"})

    arrays: list[np.ndarray] = []
    names: list[str] = []
    used_keys: set[str] = set()
    wind_component_found: str | None = None
    if "wind" in target_variables:
        wind_key = _find_key_by_alias(keys, WIND_SPEED_ALIASES) or _find_key_by_alias_exact(keys, ("wind",))
        u_key = _find_key_by_alias(keys, U_WIND_ALIASES)
        v_key = _find_key_by_alias(keys, V_WIND_ALIASES)
        if wind_key is not None:
            arrays.append(_array_to_time_node(np.asarray(npz[wind_key], dtype=float), config))
            names.append("wind")
            used_keys.add(wind_key)
            wind_component_found = "speed"
        elif u_key is not None and v_key is not None:
            u = _array_to_time_node(np.asarray(npz[u_key], dtype=float), config)
            v = _array_to_time_node(np.asarray(npz[v_key], dtype=float), config)
            n_t = min(u.shape[0], v.shape[0])
            n_n = min(u.shape[1], v.shape[1])
            arrays.append(np.sqrt(u[:n_t, :n_n] ** 2 + v[:n_t, :n_n] ** 2))
            names.append("wind")
            used_keys.update({u_key, v_key})
            wind_component_found = "speed"
        elif u_key is not None or v_key is not None:
            single = u_key or v_key
            logger.warning("Only one wind component found in %s (%s); using it as wind proxy.", path, single)
            arrays.append(_array_to_time_node(np.asarray(npz[single], dtype=float), config))
            names.append("wind")
            used_keys.add(single)
            wind_component_found = "u" if single == u_key else "v"

    for key in keys:
        if key in used_keys:
            continue
        canonical = canonicalize_variable_request(key)
        if canonical not in target_variables or canonical == "wind":
            continue
        arr = _array_to_time_node(np.asarray(npz[key], dtype=float), config)
        arrays.append(arr)
        names.append(canonical)
    if not arrays:
        inferred = _infer_variable_from_filename(path)
        if inferred and inferred in target_variables and data_key is not None:
            arr = _array_to_time_node(np.asarray(npz[data_key], dtype=float), config)
            arrays.append(arr)
            names.append(inferred)
    if not arrays:
        return None
    data, lat_ref, lon_ref, timestamps = _align_arrays(
        arrays,
        lat.reshape(-1) if lat is not None else None,
        _normalize_longitudes(lon.reshape(-1)) if lon is not None else None,
        timestamps,
    )
    if lat_ref is None or lat_ref.size != data.shape[1]:
        lat_ref = np.full(data.shape[1], np.nan)
    if lon_ref is None or lon_ref.size != data.shape[1]:
        lon_ref = np.full(data.shape[1], np.nan)
    metadata = {"source_file": str(path), "loader": "npz"}
    if wind_component_found is not None:
        metadata["wind_component"] = wind_component_found
    return AtmosphericDataset(
        data=data,
        variable_names=names,
        lat=lat_ref,
        lon=lon_ref,
        timestamps=timestamps,
        metadata=metadata,
    )


def _dataset_from_table(
    path: Path,
    target_variables: list[str],
    config: dict[str, Any],
) -> AtmosphericDataset | None:
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    time_col = _detect_name(df.columns, TIME_NAMES)
    if time_col is None:
        return None
    var_cols = [c for c in df.columns if canonicalize_variable_request(c) in target_variables]
    value_specs: list[tuple[str, str]] = []
    if "wind" in target_variables:
        wind_col = _find_table_col(df.columns, WIND_SPEED_ALIASES) or _find_key_by_alias_exact(
            [str(c) for c in df.columns], ("wind",)
        )
        u_col = _find_table_col(df.columns, U_WIND_ALIASES)
        v_col = _find_table_col(df.columns, V_WIND_ALIASES)
        if wind_col is not None:
            value_specs.append(("wind", wind_col))
        elif u_col is not None and v_col is not None:
            df = df.copy()
            df["__wind_speed"] = np.sqrt(df[u_col].astype(float) ** 2 + df[v_col].astype(float) ** 2)
            value_specs.append(("wind", "__wind_speed"))
        elif u_col is not None or v_col is not None:
            single = u_col or v_col
            logger.warning("Only one wind component found in %s (%s); using it as wind proxy.", path, single)
            value_specs.append(("wind", single))
    for col in var_cols:
        canonical = canonicalize_variable_request(col)
        if canonical != "wind":
            value_specs.append((canonical, col))
    if not value_specs:
        return None
    lat_col = _detect_name(df.columns, LAT_NAMES)
    lon_col = _detect_name(df.columns, LON_NAMES)
    node_col = _detect_name(df.columns, NODE_NAMES)
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    if lat_col and lon_col:
        node_keys = [lat_col, lon_col]
    elif node_col:
        node_keys = [node_col]
    else:
        node_keys = []

    if node_keys:
        node_df = df[node_keys].drop_duplicates().reset_index(drop=True)
        node_df["_node"] = np.arange(len(node_df))
        df = df.merge(node_df, on=node_keys, how="left")
        node_col_use = "_node"
    else:
        df["_node"] = 0
        node_df = pd.DataFrame({"_node": [0]})
        node_col_use = "_node"

    times = pd.Index(sorted(df[time_col].unique()))
    stride = int(config.get("time_stride") or 1)
    times = times[::stride]
    max_t = config.get("max_time_steps")
    if max_t is not None:
        times = times[: int(max_t)]
    df = df[df[time_col].isin(times)]

    arrays = []
    names = []
    for canonical, col in value_specs:
        pivot = df.pivot_table(index=time_col, columns=node_col_use, values=col, aggfunc="mean")
        pivot = pivot.reindex(index=times, columns=node_df["_node"].values)
        arrays.append(pivot.to_numpy(dtype=float))
        names.append(canonical)
    data, _, _, timestamps = _align_arrays(arrays, None, None, pd.to_datetime(times))
    if lat_col and lon_col:
        lat = node_df[lat_col].to_numpy(dtype=float)
        lon = _normalize_longitudes(node_df[lon_col].to_numpy(dtype=float))
    else:
        lat = np.full(data.shape[1], np.nan)
        lon = np.full(data.shape[1], np.nan)
    return AtmosphericDataset(
        data=data,
        variable_names=names,
        lat=lat[: data.shape[1]],
        lon=lon[: data.shape[1]],
        timestamps=timestamps,
        metadata={"source_file": str(path), "loader": "table"},
    )


def _merge_partial_datasets(
    partials: list[AtmosphericDataset],
    target_variables: list[str],
    config: dict[str, Any] | None = None,
) -> AtmosphericDataset:
    """Merge compatible partial files into a single time-continuous dataset.

    Many reanalysis exports are stored as separate files per variable and/or
    time period. The first project version originally chose the largest single
    compatible period; this merger instead stitches compatible time segments
    together by timestamp, dropping overlapping duplicate timestamps. The
    resulting directed dependencies are still observational lagged
    dependencies, not interventional effects.
    """
    config = config or {}
    reference = _choose_reference_dataset(partials)
    series_by_var: dict[str, tuple[np.ndarray, pd.DatetimeIndex | None, list[str]]] = {}

    for var in target_variables:
        if var == "wind":
            wind_choice = _choose_wind_dataset(partials, reference, config)
            if wind_choice is None:
                continue
            series_by_var["wind"] = wind_choice
            continue

        segments = _collect_variable_segments(partials, reference, var)
        if not segments:
            continue
        series_by_var[var] = _concatenate_time_segments(segments, config)

    if not series_by_var:
        return reference

    names = [v for v in target_variables if v in series_by_var]
    aligned_arrays, timestamps = _align_variable_series([series_by_var[v] for v in names])
    n_n = min(arr.shape[1] for arr in aligned_arrays)
    data = np.stack([arr[:, :n_n] for arr in aligned_arrays], axis=2)
    sources = {var: series_by_var[var][2] for var in names}

    return AtmosphericDataset(
        data=data,
        variable_names=names,
        lat=reference.lat[:n_n],
        lon=reference.lon[:n_n],
        timestamps=timestamps,
        node_ids=reference.node_ids[:n_n] if reference.node_ids is not None else None,
        metadata={"sources_by_variable": sources, "loader": "merged_time_segments"},
    )


def _choose_reference_dataset(partials: list[AtmosphericDataset]) -> AtmosphericDataset:
    """Choose the spatial grid used as the reference for merging."""
    return max(
        partials,
        key=lambda d: (
            int(np.isfinite(d.lat).sum()) + int(np.isfinite(d.lon).sum()),
            d.data.shape[1],
            d.data.shape[0],
            len(set(d.variable_names)),
        ),
    )


def _collect_variable_segments(
    partials: list[AtmosphericDataset],
    reference: AtmosphericDataset,
    var: str,
) -> list[tuple[np.ndarray, pd.DatetimeIndex | None, str]]:
    """Collect compatible time segments for one canonical variable."""
    segments: list[tuple[np.ndarray, pd.DatetimeIndex | None, str]] = []
    for ds in partials:
        idx = ds.variable_index(var)
        if idx is None or not _compatible_spatial_grid(reference, ds):
            continue
        source = str(ds.metadata.get("source_file", "unknown"))
        segments.append((ds.data[:, :, idx], ds.timestamps, source))
    return segments


def _choose_wind_dataset(
    partials: list[AtmosphericDataset],
    reference: AtmosphericDataset,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex | None, list[str]] | None:
    """Choose or synthesize wind speed from compatible partial time segments."""
    config = config or {}
    speed_segments: list[tuple[np.ndarray, pd.DatetimeIndex | None, str]] = []
    u_segments: list[tuple[np.ndarray, pd.DatetimeIndex | None, str]] = []
    v_segments: list[tuple[np.ndarray, pd.DatetimeIndex | None, str]] = []
    proxy_segments: list[tuple[np.ndarray, pd.DatetimeIndex | None, str]] = []

    for ds in partials:
        idx = ds.variable_index("wind")
        if idx is None or not _compatible_spatial_grid(reference, ds):
            continue
        component = str(ds.metadata.get("wind_component", "speed"))
        source = str(ds.metadata.get("source_file", "unknown"))
        entry = (ds.data[:, :, idx], ds.timestamps, source)
        if component == "speed":
            speed_segments.append(entry)
        elif component == "u":
            u_segments.append(entry)
        elif component == "v":
            v_segments.append(entry)
        else:
            proxy_segments.append(entry)

    wind_segments: list[tuple[np.ndarray, pd.DatetimeIndex | None, str]] = list(speed_segments)
    if speed_segments:
        logger.info("Using wind-speed segments and de-duplicating overlaps where present.")
    if u_segments and v_segments:
        wind_data, wind_times, sources = _synthesize_wind_from_components(u_segments, v_segments, config)
        wind_segments.append((wind_data, wind_times, " + ".join(sources)))
    elif u_segments or v_segments:
        logger.warning("Only one compatible wind component was found; using it as wind proxy.")
        wind_segments.extend(u_segments or v_segments)
    elif proxy_segments:
        wind_segments.extend(proxy_segments)

    if not wind_segments:
        return None
    # Speed files and u/v-derived wind may overlap. Concatenate so duplicate
    # timestamps are removed and the broadest time span is retained.
    return _concatenate_time_segments(wind_segments, config)


def _compatible_grid(reference: AtmosphericDataset, other: AtmosphericDataset) -> bool:
    if not _compatible_spatial_grid(reference, other):
        return False
    if reference.data.shape[0] != other.data.shape[0]:
        return False
    return True


def _compatible_spatial_grid(reference: AtmosphericDataset, other: AtmosphericDataset) -> bool:
    if reference.data.shape[1] != other.data.shape[1]:
        return False
    if np.isfinite(reference.lat).any() and np.isfinite(other.lat).any():
        # Some local ERA5/CDS exports round 64x32 grid coordinates to three
        # decimals while WeatherBench-style files retain half-grid precision.
        # Treat those as the same spatial grid, but still reject reversed or
        # materially different node orderings.
        lat_close = np.nanmean(np.abs(reference.lat - other.lat)) < 1e-2
        if np.isfinite(reference.lon).any() and np.isfinite(other.lon).any():
            lon_close = np.nanmean(np.abs(reference.lon - other.lon)) < 1e-2
        else:
            lon_close = True
        return bool(lat_close and lon_close)
    return True


def _synthesize_wind_from_components(
    u_segments: list[tuple[np.ndarray, pd.DatetimeIndex | None, str]],
    v_segments: list[tuple[np.ndarray, pd.DatetimeIndex | None, str]],
    config: dict[str, Any],
) -> tuple[np.ndarray, pd.DatetimeIndex | None, list[str]]:
    """Align u/v components and return wind speed segments."""
    u_data, u_times, u_sources = _concatenate_time_segments(u_segments, config)
    v_data, v_times, v_sources = _concatenate_time_segments(v_segments, config)
    arrays, timestamps = _align_variable_series(
        [(u_data, u_times, u_sources), (v_data, v_times, v_sources)]
    )
    n_n = min(arr.shape[1] for arr in arrays)
    wind = np.sqrt(arrays[0][:, :n_n] ** 2 + arrays[1][:, :n_n] ** 2)
    return wind, timestamps, [f"u:{src}" for src in u_sources] + [f"v:{src}" for src in v_sources]


def _concatenate_time_segments(
    segments: list[tuple[np.ndarray, pd.DatetimeIndex | None, str]],
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex | None, list[str]]:
    """Concatenate time segments, sorting and de-duplicating timestamps."""
    config = config or {}
    if not segments:
        raise ValueError("At least one segment is required.")

    prepared = []
    for data, timestamps, source in segments:
        arr = np.asarray(data, dtype=float)
        ts = timestamps
        if ts is not None and len(ts) != arr.shape[0]:
            ts = ts[: arr.shape[0]]
        prepared.append((arr, ts, source))

    if all(ts is not None for _, ts, _ in prepared):
        arrays = [arr[: len(ts)] for arr, ts, _ in prepared if ts is not None]
        times = pd.DatetimeIndex(
            np.concatenate([np.asarray(ts.values) for _, ts, _ in prepared if ts is not None])
        )
        data = np.concatenate(arrays, axis=0)
        order = np.argsort(times.values, kind="mergesort")
        times = pd.DatetimeIndex(times.values[order])
        data = data[order]
        keep = ~times.duplicated(keep="first")
        times = pd.DatetimeIndex(times.values[keep])
        data = data[keep]
        max_t = config.get("max_time_steps")
        if max_t is not None:
            data = data[: int(max_t)]
            times = times[: int(max_t)]
        return data, times, _unique_sources(source for _, _, source in prepared)

    # Fallback for arrays without timestamps: preserve source-name order and
    # concatenate by file order. This is less scientifically ideal, so timestamp
    # metadata is preferred whenever available.
    prepared = sorted(prepared, key=lambda item: item[2])
    data = np.concatenate([arr for arr, _, _ in prepared], axis=0)
    max_t = config.get("max_time_steps")
    if max_t is not None:
        data = data[: int(max_t)]
    return data, None, _unique_sources(source for _, _, source in prepared)


def _align_variable_series(
    series: list[tuple[np.ndarray, pd.DatetimeIndex | None, list[str]]],
) -> tuple[list[np.ndarray], pd.DatetimeIndex | None]:
    """Align variables on common timestamps, or by minimum length if absent."""
    if not series:
        raise ValueError("At least one variable series is required.")
    if all(timestamps is not None for _, timestamps, _ in series):
        common = pd.DatetimeIndex(series[0][1])
        for _, timestamps, _ in series[1:]:
            common = common.intersection(pd.DatetimeIndex(timestamps))
        common = common.sort_values()
        if common.empty:
            raise ValueError("No common timestamps across requested variables.")
        aligned = []
        for data, timestamps, _ in series:
            idx = pd.DatetimeIndex(timestamps).get_indexer(common)
            if np.any(idx < 0):
                raise ValueError("Failed to align timestamps across variables.")
            aligned.append(np.asarray(data, dtype=float)[idx])
        return aligned, common

    n_t = min(data.shape[0] for data, _, _ in series)
    return [np.asarray(data, dtype=float)[:n_t] for data, _, _ in series], None


def _unique_sources(sources: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for source in sources:
        if source in seen:
            continue
        seen.add(source)
        unique.append(source)
    return unique


def _align_arrays(
    arrays: list[np.ndarray],
    lat: np.ndarray | None,
    lon: np.ndarray | None,
    timestamps: pd.DatetimeIndex | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex | None]:
    n_t = min(a.shape[0] for a in arrays)
    n_n = min(a.shape[1] for a in arrays)
    data = np.stack([a[:n_t, :n_n] for a in arrays], axis=2)
    if lat is None or lat.size < n_n:
        lat = np.full(n_n, np.nan)
    if lon is None or lon.size < n_n:
        lon = np.full(n_n, np.nan)
    if timestamps is not None:
        timestamps = timestamps[:n_t]
    return data, lat[:n_n], lon[:n_n], timestamps


def _find_xarray_var(ds: Any, aliases: Iterable[str]) -> str | None:
    names = list(ds.data_vars)
    normalized_to_name = {normalize_name(name): name for name in names}
    for alias in aliases:
        normalized = normalize_name(alias)
        if normalized in normalized_to_name:
            return normalized_to_name[normalized]
    for alias in aliases:
        normalized = normalize_name(alias)
        if len(normalized) < 3:
            continue
        candidates = [name for name in names if normalized in normalize_name(name)]
        if candidates:
            return candidates[0]
    return None


def _find_xarray_var_exact(ds: Any, aliases: Iterable[str]) -> str | None:
    names = list(ds.data_vars)
    normalized_to_name = {normalize_name(name): name for name in names}
    for alias in aliases:
        normalized = normalize_name(alias)
        if normalized in normalized_to_name:
            return normalized_to_name[normalized]
    return None


def _find_key_by_alias(keys: Iterable[str], aliases: Iterable[str]) -> str | None:
    normalized_to_key = {normalize_name(k): k for k in keys}
    for alias in aliases:
        normalized = normalize_name(alias)
        if normalized in normalized_to_key:
            return normalized_to_key[normalized]
    for alias in aliases:
        normalized = normalize_name(alias)
        if len(normalized) < 3:
            continue
        for key in keys:
            if normalized in normalize_name(key):
                return key
    return None


def _find_key_by_alias_exact(keys: Iterable[str], aliases: Iterable[str]) -> str | None:
    normalized_to_key = {normalize_name(k): k for k in keys}
    for alias in aliases:
        normalized = normalize_name(alias)
        if normalized in normalized_to_key:
            return normalized_to_key[normalized]
    return None


def _find_table_col(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    return _find_key_by_alias([str(c) for c in columns], aliases)


def _detect_name(names: Iterable[Any], aliases: Iterable[str]) -> str | None:
    name_list = [str(n) for n in names]
    normalized_to_name = {normalize_name(n): n for n in name_list}
    for alias in aliases:
        normalized = normalize_name(alias)
        if normalized in normalized_to_name:
            return normalized_to_name[normalized]
    for alias in aliases:
        normalized = normalize_name(alias)
        for name in name_list:
            if normalized == normalize_name(name):
                return name
    return None


def _normalize_longitudes(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180
    return lon


def _infer_variable_from_filename(path: Path) -> str | None:
    stem = normalize_name(path.stem)
    for canonical, aliases in VARIABLE_ALIASES.items():
        if any(normalize_name(alias) in stem for alias in aliases):
            return canonical
    if any(normalize_name(alias) in stem for alias in U_WIND_ALIASES + V_WIND_ALIASES):
        return "wind"
    return None


def _infer_wind_component_from_name(name: str) -> str | None:
    normalized = normalize_name(name)
    if any(normalize_name(alias) in normalized for alias in U_WIND_ALIASES):
        return "u"
    if any(normalize_name(alias) in normalized for alias in V_WIND_ALIASES):
        return "v"
    if any(normalize_name(alias) in normalized for alias in VARIABLE_ALIASES["wind"]):
        return "speed"
    return None


def _array_to_time_node(arr: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    arr = _apply_time_options(np.asarray(arr, dtype=float), config)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def _apply_time_options(arr: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    stride = int(config.get("time_stride") or 1)
    arr = arr[::stride]
    max_t = config.get("max_time_steps")
    if max_t is not None:
        arr = arr[: int(max_t)]
    return arr


def _optional_npz_array(npz: Any, names: Iterable[str]) -> np.ndarray | None:
    for name in names:
        if name in npz.files:
            return np.asarray(npz[name])
    return None


def _optional_variable_names_npz(npz: Any) -> list[str] | None:
    for name in ("variable_names", "variables", "vars", "channels", "features"):
        if name in npz.files:
            return [str(v) for v in npz[name].tolist()]
    return None


def _optional_timestamps_npz(npz: Any) -> pd.DatetimeIndex | None:
    for name in ("timestamps", "time", "times", "dates"):
        if name in npz.files:
            try:
                return pd.to_datetime(npz[name])
            except Exception:
                return None
    return None


def _combined_numpy_array(
    arr: np.ndarray,
    variable_names: list[str],
    target_variables: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray | None, list[str]]:
    if arr.shape[-1] == len(variable_names):
        axis = arr.ndim - 1
    else:
        axis = next((i for i, s in enumerate(arr.shape) if i != 0 and s == len(variable_names)), None)
    if axis is None:
        return None, []
    arr = np.moveaxis(arr, axis, -1)
    arr = _apply_time_options(arr, config)
    arrays: list[np.ndarray] = []
    names: list[str] = []
    for target in target_variables:
        if target == "wind":
            speed_idx = _find_name_index_by_alias(variable_names, WIND_SPEED_ALIASES)
            if speed_idx is None:
                speed_idx = _find_name_index_by_alias(variable_names, ("wind",), exact_only=True)
            u_idx = _find_name_index_by_alias(variable_names, U_WIND_ALIASES)
            v_idx = _find_name_index_by_alias(variable_names, V_WIND_ALIASES)
            if speed_idx is not None:
                arrays.append(arr[..., speed_idx])
                names.append("wind")
            elif u_idx is not None and v_idx is not None:
                arrays.append(np.sqrt(arr[..., u_idx] ** 2 + arr[..., v_idx] ** 2))
                names.append("wind")
            elif u_idx is not None or v_idx is not None:
                idx = u_idx if u_idx is not None else v_idx
                arrays.append(arr[..., idx])
                names.append("wind")
            continue
        idx = _find_name_index_for_canonical(variable_names, target)
        if idx is not None:
            arrays.append(arr[..., idx])
            names.append(target)
    if not arrays:
        return None, []
    data = np.stack(arrays, axis=-1)
    if data.ndim == 2:
        data = data[:, None, :]
    elif data.ndim > 3:
        data = data.reshape(data.shape[0], -1, data.shape[-1])
    return np.asarray(data, dtype=float), names


def _find_name_index_for_canonical(names: list[str], target: str) -> int | None:
    for idx, name in enumerate(names):
        if canonicalize_variable_request(name) == target:
            return idx
    aliases = VARIABLE_ALIASES.get(target, (target,))
    return _find_name_index_by_alias(names, aliases)


def _find_name_index_by_alias(
    names: list[str],
    aliases: Iterable[str],
    exact_only: bool = False,
) -> int | None:
    normalized_names = [normalize_name(name) for name in names]
    for alias in aliases:
        normalized = normalize_name(alias)
        for idx, name in enumerate(normalized_names):
            if name == normalized:
                return idx
    if exact_only:
        return None
    for alias in aliases:
        normalized = normalize_name(alias)
        if len(normalized) < 3:
            continue
        for idx, name in enumerate(normalized_names):
            if normalized in name:
                return idx
    return None


def _lat_lon_from_optional(
    lat: np.ndarray | None,
    lon: np.ndarray | None,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lat is None or np.asarray(lat).size != n:
        lat = np.full(n, np.nan)
    if lon is None or np.asarray(lon).size != n:
        lon = np.full(n, np.nan)
    return np.asarray(lat, dtype=float).reshape(-1), _normalize_longitudes(np.asarray(lon, dtype=float).reshape(-1))
