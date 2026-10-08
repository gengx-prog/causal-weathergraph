"""Bounded public metadata inventory and one-day ISCCP download probe.

Does not print credentials, authenticate to NASA, download full-period science
data, or start a persistent worker. Earthdata configuration is inspected only in
standard netrc locations and named environment variables. Scientific transfers
are restricted to four specified ISCCP snapshots, <=100,000,000 bytes in total.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import netrc
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
import xarray as xr


CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
NCEI = ("https://www.ncei.noaa.gov/data/"
        "international-satellite-cloud-climate-project-isccp-h-series-data/"
        "access/isccp-basic/hgg/201001/")
COLLECTIONS = {
    "ceres_hourly": ("C3181056140-LARC_CLOUD", "2000-03-01", "2025-12-31"),
    "merra2_aerosol": ("C1276812830-GES_DISC", "1980-01-01", "2025-12-31"),
}
BYTE_LIMIT = 100_000_000


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def auth_presence():
    """Never return username, password, token, host list, or file contents."""
    output = {}
    for name in (".netrc", "_netrc"):
        path = Path.home() / name
        record = {"exists": path.is_file(), "earthdata_entry_present": False}
        if path.is_file():
            try:
                entry = netrc.netrc(str(path)).authenticators("urs.earthdata.nasa.gov")
                record["earthdata_entry_present"] = bool(entry and entry[0] and entry[2])
                record["parse_ok"] = True
            except Exception:
                record["parse_ok"] = False
        output[name] = record
    output["environment_present"] = {
        key: bool(os.environ.get(key)) for key in
        ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN", "EARTHDATA_ACCESS_TOKEN")
    }
    env = output["environment_present"]
    output["standard_configuration_present"] = bool(
        any(output[n]["earthdata_entry_present"] for n in (".netrc", "_netrc"))
        or (env["EARTHDATA_USERNAME"] and env["EARTHDATA_PASSWORD"])
        or env["EARTHDATA_TOKEN"] or env["EARTHDATA_ACCESS_TOKEN"])
    output["authentication_usability"] = (
        "not_tested_configuration_present" if output["standard_configuration_present"]
        else "not_available_no_standard_configuration")
    output["scope"] = "Standard local netrc files and named Earthdata environment variables only. No credential values recorded."
    return output


def get_json(url, params=None, attempts=3):
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=(10, 45))
            response.raise_for_status()
            return response.json(), response.headers
        except (requests.RequestException, ValueError):
            if attempt == attempts - 1:
                raise


def data_links(entry):
    return [x["href"] for x in entry.get("links", [])
            if x.get("rel", "").endswith("/data#") and "href" in x]


def inventory_one(task, output):
    collection, start, end = COLLECTIONS[task]
    params = {"collection_concept_id": collection,
              "temporal": f"{start}T00:00:00Z,{end}T23:59:59Z",
              "page_size": 2000, "sort_key": "start_date"}
    records = []
    hits = None
    page = 1
    while True:
        params["page_num"] = page
        raw, headers = get_json(CMR, params)
        hits = int(headers.get("CMR-Hits", hits or 0))
        entries = raw["feed"]["entry"]
        for entry in entries:
            links = data_links(entry)
            direct = next((url for url in links if "protected" in url or url.endswith(".nc4")), "")
            records.append({"collection_id": collection, "granule_id": entry["id"],
                            "title": entry["title"], "start": entry.get("time_start", ""),
                            "end": entry.get("time_end", ""),
                            "cmr_reported_size_mb": entry.get("granule_size", ""),
                            "direct_download_url": direct,
                            "other_data_links": " | ".join(url for url in links if url != direct)})
        print(f"{task}: metadata page {page}, {len(records)}/{hits}", flush=True)
        if len(records) >= hits or not entries:
            break
        page += 1
    keys = ["collection_id", "granule_id", "title", "start", "end", "cmr_reported_size_mb",
            "direct_download_url", "other_data_links"]
    path = output / f"{task}_granule_inventory.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    sizes = [float(row["cmr_reported_size_mb"]) for row in records if row["cmr_reported_size_mb"]]
    dates = [row["start"][:10] for row in records]
    expected = {(date.fromisoformat(start).toordinal() + i) for i in
                range((date.fromisoformat(end) - date.fromisoformat(start)).days + 1)}
    observed = {date.fromisoformat(v).toordinal() for v in dates}
    result = {"collection_id": collection, "requested_start": start, "requested_end": end,
              "cmr_hits": hits, "listed_records": len(records),
              "unique_granule_ids": len({row["granule_id"] for row in records}),
              "unique_start_dates": len(observed), "expected_daily_dates": len(expected),
              "missing_start_dates": [date.fromordinal(i).isoformat() for i in sorted(expected - observed)],
              "duplicate_start_date_records": len(dates) - len(set(dates)),
              "first_start": min(row["start"] for row in records),
              "last_end": max(row["end"] for row in records),
              "sizes_available": len(sizes), "sum_cmr_reported_mb": sum(sizes),
              "median_cmr_reported_mb": float(np.median(sizes)),
              "min_cmr_reported_mb": min(sizes), "max_cmr_reported_mb": max(sizes),
              "manifest": str(path),
              "coverage_scope": "Public CMR metadata only; one record per start date does not verify within-file variables, missing values, or download accessibility.",
              "size_scope": "Sum of CMR granule_size reported as MB, not bytes transferred. Stored units retained to avoid assuming decimal/binary conversion."}
    years = []
    for year in sorted({d[:4] for d in dates}):
        year_rows = [r for r in records if r["start"].startswith(year)]
        years.append({"year": year, "granules": len(year_rows),
                      "sum_cmr_reported_mb": sum(float(r["cmr_reported_size_mb"] or 0) for r in year_rows)})
    with (output / f"{task}_annual_inventory.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["year", "granules", "sum_cmr_reported_mb"])
        writer.writeheader(); writer.writerows(years)
    write_json(output / f"{task}_inventory_summary.json", result)
    return result


def probe_thredds():
    dataset = "cdr/isccp_hgg_agg/ISCCP-H_Aggregation_Basic_Gridded_Global_(HGG)_best.ncd"
    urls = ["https://www.ncei.noaa.gov/thredds/catalog/satellite/cdr-isccp-h.xml",
            "https://www.ncei.noaa.gov/thredds/dodsC/" + quote(dataset, safe="/()_") + ".dds",
            "https://www.ncei.noaa.gov/thredds/ncss/" + quote(dataset, safe="/()_") + "/dataset.xml"]
    def request(url):
        result = {"url": url}
        try:
            response = requests.get(url, timeout=(8, 12))
            result.update(http_status=response.status_code, bytes=len(response.content),
                          content_type=response.headers.get("Content-Type", ""))
            result["metadata_available"] = response.ok and ("Dataset" in response.text or "catalog" in response.text)
        except requests.RequestException as exc:
            result.update(metadata_available=False, error_type=type(exc).__name__)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(request, urls))
    return {"checks": results,
            "conclusion": "No server-side variable subset has been successfully downloaded in this probe; failures do not prove the service lacks subsetting capability."}


def safe_download(url, path, budget_remaining):
    if path.exists():
        with xr.open_dataset(path) as dataset:
            assert "cldamt" in dataset
        return {"url": url, "path": str(path), "bytes": path.stat().st_size,
                "transferred_this_run": 0, "cached": True,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    temp = path.with_suffix(path.suffix + ".part")
    transferred = 0
    with requests.get(url, stream=True, timeout=(10, 30)) as response:
        response.raise_for_status()
        content_length = int(response.headers.get("Content-Length", 0))
        if content_length > budget_remaining:
            raise RuntimeError("Download would exceed the fixed scientific-data byte budget")
        with temp.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                transferred += len(chunk)
                if transferred > budget_remaining:
                    raise RuntimeError("Transfer exceeded fixed byte budget; partial file retained for audit")
                handle.write(chunk)
    with xr.open_dataset(temp) as dataset:
        assert "cldamt" in dataset
    temp.replace(path)
    return {"url": url, "path": str(path), "bytes": path.stat().st_size,
            "transferred_this_run": transferred, "cached": False,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def decode_isccp_cloud(path):
    """Apply packed valid range BEFORE scale/offset; do not clip missing cells.

    HGG v01r00 can store int16 32767 without an explicit _FillValue.
    xarray's default decoder then gives 3326.7 percent, so the packed valid
    range is authoritative. Time decoding remains enabled.
    """
    with xr.open_dataset(path, mask_and_scale=False) as dataset:
        raw = dataset["cldamt"].load()
        assert raw.attrs["units"] == "percent"
        lower, upper = float(raw.attrs["valid_min"]), float(raw.attrs["valid_max"])
        scale = float(raw.attrs.get("scale_factor", 1))
        offset = float(raw.attrs.get("add_offset", 0))
        valid = np.isfinite(raw.values) & (raw.values >= lower) & (raw.values <= upper)
        for key in ("_FillValue", "missing_value"):
            if key in raw.attrs:
                for value in np.atleast_1d(raw.attrs[key]):
                    valid &= raw.values != value
        # Follow the packing attributes' native precision, as the CF decoder
        # does, rather than promote float32 .1 to float64 and create tiny
        # negative roundoff at the exact0-percent endpoint.
        decode_dtype = np.result_type(raw.attrs.get("scale_factor", np.float32(1)),
                                      raw.attrs.get("add_offset", np.float32(0)))
        decoded = (raw.values.astype(decode_dtype) * np.asarray(scale, dtype=decode_dtype)
                   + np.asarray(offset, dtype=decode_dtype))
        values = np.where(valid, decoded, np.nan)
        cloud = xr.DataArray(values, dims=raw.dims, coords=raw.coords, name="cldamt",
                             attrs={"units": "percent", "long_name": "Cloud amount"})
        info = {"packed_dtype": str(raw.dtype), "packed_valid_min": lower,
                "packed_valid_max": upper, "scale_factor": scale, "add_offset": offset,
                "invalid_packed_values": [int(x) for x in np.unique(raw.values[~valid])],
                "invalid_cells": int((~valid).sum()), "decoded_dtype": str(decode_dtype),
                "explicit_fill_value_present": "_FillValue" in raw.attrs,
                "rule": "Mask outside packed valid_min/valid_max and any explicit fill value, then scale+offset. No clipping or interpolation."}
    assert np.isfinite(values).any()
    assert float(np.nanmin(values)) >= -1e-5 and float(np.nanmax(values)) <= 100 + 1e-5
    return cloud, info


def isccp_pilot(output):
    raw_dir = output / "pilot_isccp_20100101_raw"
    raw_dir.mkdir(exist_ok=True)
    records = []
    arrays = []
    budget = BYTE_LIMIT
    for hour in (0, 6, 12, 18):
        name = f"ISCCP-Basic.HGG.v01r00.GLOBAL.2010.01.01.{hour:02d}00.GPC.10KM.CS00.EA1.00.nc"
        record = safe_download(NCEI + name, raw_dir / name, budget)
        budget -= record["transferred_this_run"]
        records.append(record)
        cloud, decoding = decode_isccp_cloud(raw_dir / name)
        record["packed_decoding"] = decoding
        arrays.append(cloud)
    cloud = xr.concat(arrays, dim="time")
    expected = np.array([f"2010-01-01T{hour:02d}:00:00" for hour in (0, 6, 12, 18)], dtype="datetime64[ns]")
    assert np.array_equal(cloud.time.values, expected)
    assert cloud.shape == (4, 180, 360)
    assert np.allclose(np.diff(cloud.lat), 1) and np.allclose(np.diff(cloud.lon), 1)
    assert float(cloud.min()) >= -1e-5 and float(cloud.max()) <= 100 + 1e-5
    fraction = (cloud / 100.0).astype("float32")
    fraction.attrs = {"long_name": "ISCCP HGG total cloud area fraction", "units": "1",
                      "source_variable": "cldamt", "conversion": "packed valid-range mask, then scale+offset, then percent / 100"}
    derived = xr.Dataset({"cloud_fraction": fraction})
    derived.attrs = {"source": "NOAA ISCCP HGG Basic v01r00", "pilot_date": "2010-01-01",
                     "sampling": "Original instantaneous snapshots at 00/06/12/18 UTC; no temporal interpolation",
                     "processing": "Local variable extraction from four small public source files; not a THREDDS subset download",
                     "created_utc": utc_now()}
    derived_path = output / "isccp_hgg_total_cloud_20100101_6hourly_global.nc"
    derived.to_netcdf(derived_path, encoding={"cloud_fraction": {"zlib": True, "complevel": 4}})
    missing = []
    for i, timestamp in enumerate(expected):
        a = fraction.values[i]
        missing.append({"time": str(timestamp), "finite_cells": int(np.isfinite(a).sum()),
                        "missing_cells": int(np.isnan(a).sum()), "missing_fraction": float(np.isnan(a).mean()),
                        "min_fraction": float(np.nanmin(a)), "max_fraction": float(np.nanmax(a))})
    result = {"status": "downloaded_and_validated", "source_files": records,
              "source_total_bytes": sum(x["bytes"] for x in records),
              "scientific_transfer_limit_bytes": BYTE_LIMIT,
              "transferred_this_run_bytes": sum(x["transferred_this_run"] for x in records),
              "derived_path": str(derived_path), "derived_bytes": derived_path.stat().st_size,
              "derived_sha256": hashlib.sha256(derived_path.read_bytes()).hexdigest(),
              "dimensions": dict(derived.sizes), "time_validation": "four exact six-hour UTC snapshots",
              "lat_range": [float(cloud.lat.min()), float(cloud.lat.max())],
              "lon_range": [float(cloud.lon.min()), float(cloud.lon.max())],
              "validation_by_time": missing,
              "subset_route": "Public direct HTTPS, then local cloud-variable extraction. Four source files total well below100 MB.",
              "not_done": "No full-period download, model fitting, regional weighting, interpolation, or background worker."}
    write_json(output / "isccp_pilot_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=r"D:\Paper2\Major Revision\supplementary_data\cloud_validation")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {"started_utc": utc_now(), "authentication": auth_presence(), "inventories": {}}
    write_json(output / "earthdata_auth_status.json", result["authentication"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {key: pool.submit(inventory_one, key, output) for key in COLLECTIONS}
        for key, future in futures.items():
            try:
                result["inventories"][key] = future.result()
            except Exception as exc:
                result["inventories"][key] = {"status": "failed", "error_type": type(exc).__name__}
    result["thredds"] = probe_thredds()
    write_json(output / "isccp_thredds_probe.json", result["thredds"])
    result["ceres_science_download"] = {"status": "not_attempted", "reason":
        "No usable standard Earthdata authentication was available during this task. Public metadata inventory completed separately."}
    if not args.inventory_only:
        result["isccp_pilot"] = isccp_pilot(output)
    result["completed_utc"] = utc_now()
    write_json(output / "external_download_probe_summary.json", result)
    print(json.dumps({"completed": True, "output": str(output),
                      "inventory_status": {key: value.get("listed_records", value.get("status"))
                                           for key, value in result["inventories"].items()},
                      "pilot_status": result.get("isccp_pilot", {}).get("status")}), flush=True)


if __name__ == "__main__":
    main()
