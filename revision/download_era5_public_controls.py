"""Anonymous, resumable acquisition of selected WeatherBench2 ERA5 controls.

HTTP and decoding use at most four workers. NetCDF4 is only used by the main
thread. Each pressure-level object is fetched once and its required levels are
extracted together. This does not acquire the unavailable 2023-01-11--2025 tail.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

import numpy as np
from numcodecs import get_codec
import xarray as xr


SOURCE = (
    "https://storage.googleapis.com/weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr/"
)
FIELDS = {
    "vertical_velocity": [("omega_500", 500), ("omega_700", 700)],
    "geopotential": [("geopotential_500", 500)],
    "temperature": [("temperature_700", 700)],
    "surface_pressure": [("surface_pressure", None)],
    "mean_sea_level_pressure": [("mean_sea_level_pressure", None)],
}
REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO.parent / "supplementary_data" / "era5_controls"
DEFAULT_REFERENCE = (
    REPO.parent.parent / "vipuser" / "Data"
    / "weatherbench2_era5_6h_64x32_850hPa_1979_2018" / "temperature_850.nc"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class PublicReader:
    def __init__(self, retries=3, timeout=60):
        self.retries = retries
        self.timeout = timeout
        self.lock = threading.Lock()
        self.successful_response_bytes = 0
        self.successful_requests = 0
        self.retry_events = 0

    def fetch(self, key):
        url = SOURCE + key
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "ERA5-controls-audit/1.0"})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    expected = response.headers.get("Content-Length")
                    if expected is not None and len(raw) != int(expected):
                        raise OSError("Truncated HTTP object: " + key)
                    record = {
                        "key": key, "url": url, "bytes": len(raw),
                        "sha256_compressed_object": sha256(raw),
                        "generation": response.headers.get("x-goog-generation"),
                        "etag": response.headers.get("ETag"),
                        "google_hash": response.headers.get("x-goog-hash"),
                    }
                with self.lock:
                    self.successful_response_bytes += len(raw)
                    self.successful_requests += 1
                return raw, record
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt + 1 == self.retries:
                    raise
                with self.lock:
                    self.retry_events += 1
                time.sleep(min(2 ** attempt, 8))
        raise AssertionError("unreachable")

    def counters(self):
        with self.lock:
            return {
                "successful_http_response_bytes_this_run": self.successful_response_bytes,
                "successful_http_requests_this_run": self.successful_requests,
                "retry_events_this_run": self.retry_events,
            }


def decode_chunk(raw, array_meta, chunk_index):
    compressor = array_meta.get("compressor")
    decoded = get_codec(compressor).decode(raw) if compressor else raw
    if array_meta.get("filters"):
        raise ValueError("Unexpected Zarr filters; this reader intentionally supports the inspected store only")
    values = np.frombuffer(decoded, dtype=np.dtype(array_meta["dtype"]))
    shape = tuple(array_meta["chunks"])
    edge_shape = tuple(min(c, n - i * c) for c, n, i in zip(shape, array_meta["shape"], chunk_index))
    if values.size == int(np.prod(shape)):
        shape_to_use = shape
    elif values.size == int(np.prod(edge_shape)):
        shape_to_use = edge_shape
    else:
        raise ValueError(f"Decoded chunk has {values.size} values; expected {shape} or {edge_shape}")
    return values.reshape(shape_to_use, order=array_meta.get("order", "C"))


def read_coordinate(name, metadata, reader):
    spec = metadata[name + "/.zarray"]
    records, arrays = [], []
    for index in range((spec["shape"][0] + spec["chunks"][0] - 1) // spec["chunks"][0]):
        raw, record = reader.fetch(f"{name}/{index}")
        records.append(record)
        arrays.append(decode_chunk(raw, spec, (index,)))
    return np.concatenate(arrays)[:spec["shape"][0]], records


def fetch_selected_chunk(index, metadata, reader, levels, times, latitude, longitude, start, stop):
    chunk_start = index * 100
    response_lo, response_hi = max(start, chunk_start), min(stop, chunk_start + 100)
    local_slice = slice(response_lo - chunk_start, response_hi - chunk_start)
    arrays, attributes, objects = {}, {}, []
    for variable, selections in FIELDS.items():
        spec = metadata[variable + "/.zarray"]
        dims = metadata[variable + "/.zattrs"]["_ARRAY_DIMENSIONS"]
        expected_dims = ["time", "level", "longitude", "latitude"] if selections[0][1] is not None else ["time", "longitude", "latitude"]
        if dims != expected_dims or spec["chunks"][0] != 100:
            raise ValueError(f"Unexpected source chunk geometry for {variable}: {dims}, {spec['chunks']}")
        chunk_index = (index,) + (0,) * (len(dims) - 1)
        key = variable + "/" + ".".join(map(str, chunk_index))
        raw, record = reader.fetch(key)
        objects.append(record)
        block = decode_chunk(raw, spec, chunk_index)
        for output_name, pressure_level in selections:
            if pressure_level is None:
                selected = block[local_slice]
            else:
                positions = np.flatnonzero(levels == pressure_level)
                if len(positions) != 1:
                    raise ValueError(f"Missing/nonunique level: {pressure_level}")
                selected = block[local_slice, int(positions[0])]
            values = np.ascontiguousarray(selected.transpose(0, 2, 1), dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite values in {output_name}, source chunk {index}; no imputation performed")
            arrays[output_name] = values
            attrs = {k: v for k, v in metadata[variable + "/.zattrs"].items() if k != "_ARRAY_DIMENSIONS"}
            attrs.update({"source_variable": variable, "source_chunk_key": key})
            if pressure_level is not None:
                attrs["pressure_level_hpa"] = int(pressure_level)
            attributes[output_name] = attrs
    return {
        "index": index, "arrays": arrays, "attributes": attributes, "objects": objects,
        "times": times[response_lo:response_hi], "latitude": latitude, "longitude": longitude,
        "source_index_start": response_lo, "source_index_stop_exclusive": response_hi,
    }


def shard_paths(output, index):
    file = output / "shards" / f"controls_chunk_{index:06d}.nc"
    return file, file.with_suffix(".json")


def existing_shard(output, index, request_hash):
    file, receipt = shard_paths(output, index)
    if not file.exists() or not receipt.exists():
        return None
    try:
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if record["request_sha256"] != request_hash or record["file_sha256"] != file_sha256(file):
            return None
        return record
    except (OSError, ValueError, KeyError):
        return None


def save_shard(output, result, request_hash):
    file, receipt_path = shard_paths(output, result["index"])
    partial = file.with_suffix(".partial.nc")
    dataset = xr.Dataset(
        {name: (("time", "latitude", "longitude"), values, result["attributes"][name]) for name, values in result["arrays"].items()},
        coords={"time": result["times"], "latitude": result["latitude"], "longitude": result["longitude"]},
        attrs={"source": SOURCE, "request_sha256": request_hash, "source_time_chunk": result["index"], "retrieved_utc": utc_now(), "grid_method": "Existing WeatherBench2 conservative 64x32 grid; no regridding in this acquisition"},
    )
    encoding = {name: {"dtype": "float32", "zlib": True, "complevel": 1, "shuffle": True} for name in result["arrays"]}
    dataset.to_netcdf(partial, engine="netcdf4", encoding=encoding)
    dataset.close()
    with xr.open_dataset(partial, engine="netcdf4") as verified:
        if set(verified.data_vars) != set(result["arrays"]):
            raise ValueError("Written fields mismatch")
        np.testing.assert_array_equal(verified.time.values, result["times"])
        np.testing.assert_array_equal(verified.latitude.values, result["latitude"])
        np.testing.assert_array_equal(verified.longitude.values, result["longitude"])
        for name, values in result["arrays"].items():
            np.testing.assert_array_equal(verified[name].values, values)
    os.replace(partial, file)
    record = {
        "source_time_chunk": result["index"], "request_sha256": request_hash,
        "file": str(file.resolve()), "file_bytes": file.stat().st_size, "file_sha256": file_sha256(file),
        "n_timestamps": len(result["times"]), "first": str(result["times"][0]), "last": str(result["times"][-1]),
        "source_index_start": result["source_index_start"], "source_index_stop_exclusive": result["source_index_stop_exclusive"],
        "source_objects": result["objects"], "compressed_source_bytes": sum(x["bytes"] for x in result["objects"]),
        "fields": {name: {"shape": list(values.shape), "units": result["attributes"][name].get("units"), "minimum": float(values.min()), "maximum": float(values.max()), "nonfinite": 0} for name, values in result["arrays"].items()},
        "completed_utc": utc_now(), "netcdf_roundtrip": "all coordinates and all values identical",
    }
    atomic_json(receipt_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--reference-grid", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    if args.limit_chunks is not None and args.limit_chunks < 1:
        parser.error("--limit-chunks must be positive")
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)
    reader = PublicReader(args.retries, args.timeout)
    metadata_raw, metadata_record = reader.fetch(".zmetadata")
    metadata = json.loads(metadata_raw)["metadata"]
    coordinates, coordinate_records = {}, {}
    for name in ["time", "level", "latitude", "longitude"]:
        coordinates[name], coordinate_records[name] = read_coordinate(name, metadata, reader)
    if metadata["time/.zattrs"]["units"] != "hours since 1959-01-01":
        raise ValueError("Unexpected time units")
    times = np.datetime64("1959-01-01", "h") + coordinates["time"].astype("timedelta64[h]")
    if not np.all(np.diff(times) == np.timedelta64(6, "h")):
        raise ValueError("Source time is not continuous at six-hour intervals")
    start = int(np.searchsorted(times, np.datetime64("1979-01-01T00", "h")))
    stop = int(np.searchsorted(times, np.datetime64("2023-01-10T18", "h"), side="right"))
    if times[start] != np.datetime64("1979-01-01T00", "h") or times[stop-1] != np.datetime64("2023-01-10T18", "h"):
        raise ValueError("Expected source date range not available")
    grid_check = {"reference_file": str(args.reference_grid), "checked": False}
    if args.reference_grid.exists():
        with xr.open_dataset(args.reference_grid) as reference:
            for coord in ["latitude", "longitude"]:
                np.testing.assert_allclose(coordinates[coord], reference[coord].values, rtol=0, atol=1e-10)
            grid_check.update({"checked": True, "latitude_max_abs_difference": float(np.max(abs(coordinates["latitude"] - reference.latitude.values))), "longitude_max_abs_difference": float(np.max(abs(coordinates["longitude"] - reference.longitude.values)))})
    all_indices = list(range(start // 100, (stop - 1) // 100 + 1))
    request = {"source": SOURCE, "metadata_sha256": sha256(metadata_raw), "fields": FIELDS, "first": str(times[start]), "last": str(times[stop-1]), "missing_tail": ["2023-01-11T00:00:00", "2025-12-31T18:00:00"], "source_index_start": start, "source_index_stop_exclusive": stop, "n_timestamps": stop-start, "total_shards": len(all_indices), "output_dimensions": ["time", "latitude", "longitude"]}
    request_hash = sha256(json.dumps(request, sort_keys=True).encode())
    manifest_path = output / "manifest_public.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["request_sha256"] != request_hash:
            raise ValueError("Existing output belongs to a different request or source metadata version")
    (output / "source_zmetadata.json").write_bytes(metadata_raw)
    np.savez(output / "source_coordinates.npz", **coordinates)
    manifest = {"request": request, "request_sha256": request_hash, "source_metadata_object": metadata_record, "coordinate_objects": coordinate_records, "source_time_first": str(times[0]), "source_time_last": str(times[-1]), "levels_hpa": coordinates["level"].tolist(), "latitude": coordinates["latitude"].tolist(), "longitude": coordinates["longitude"].tolist(), "grid_check": grid_check, "runner": str(Path(__file__).resolve()), "runner_sha256": file_sha256(Path(__file__).resolve()), "updated_utc": utc_now(), "network_policy": "anonymous public HTTPS, no CDS credentials, at most four concurrent HTTP reads", "missing_policy": "fail on nonfinite; no imputation", "write_policy": "main thread only; atomic per-shard NetCDF and SHA256 receipt", "full_period_compressed_object_bytes_estimate": 17292461869, "estimate_source": "GCS object metadata sizes for chunks292 through935, five variables; excludes retry traffic"}
    atomic_json(manifest_path, manifest)
    selected = all_indices[:args.limit_chunks] if args.limit_chunks else all_indices
    records = {}
    for index in all_indices:
        existing = existing_shard(output, index, request_hash)
        if existing:
            records[index] = existing
    pending = [index for index in selected if index not in records]
    started = time.perf_counter()
    status = {"status": "running", "started_utc": utc_now(), "request_sha256": request_hash, "pid": os.getpid(), "planned_shards": len(all_indices), "selected_shards_this_run": len(selected), "cached_shards_at_start": len(records), "pending_shards_this_run": len(pending), "workers": args.workers, "limit_chunks": args.limit_chunks, "unavailable_tail": request["missing_tail"]}
    status_lock = threading.Lock()
    stop_monitor = threading.Event()

    def refresh():
        with status_lock:
            copy = dict(status)
            copy.update(reader.counters())
            copy.update({"completed_shards": len(records), "completed_timestamps": sum(r["n_timestamps"] for r in records.values()), "completed_source_compressed_bytes": sum(r["compressed_source_bytes"] for r in records.values()), "completed_output_bytes": sum(r["file_bytes"] for r in records.values()), "last_success_utc": max((r["completed_utc"] for r in records.values()), default=None), "elapsed_seconds_this_run": time.perf_counter()-started, "updated_utc": utc_now()})
            atomic_json(output / "status_public.json", copy)

    def monitor():
        while not stop_monitor.wait(5):
            refresh()

    refresh()
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            queue = deque()
            iterator = iter(pending)
            def submit_one():
                index = next(iterator, None)
                if index is not None:
                    queue.append(pool.submit(fetch_selected_chunk, index, metadata, reader, coordinates["level"], times, coordinates["latitude"], coordinates["longitude"], start, stop))
            for _ in range(args.workers):
                submit_one()
            while queue:
                result = queue.popleft().result()
                record = save_shard(output, result, request_hash)
                with status_lock:
                    records[result["index"]] = record
                print(json.dumps({"chunk": result["index"], "completed_shards": len(records), "total_shards": len(all_indices), "first": record["first"], "last": record["last"], "compressed_bytes": record["compressed_source_bytes"], "file_bytes": record["file_bytes"]}), flush=True)
                refresh()
                submit_one()
        with status_lock:
            status["status"] = "completed_public_subset" if len(records) == len(all_indices) else "pilot_completed"
            status["finished_utc"] = utc_now()
        atomic_json(output / "index_public.json", [records[k] for k in sorted(records)])
    except BaseException as error:
        with status_lock:
            status["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            status["error"] = repr(error)
        raise
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=6)
        refresh()


if __name__ == "__main__":
    main()
