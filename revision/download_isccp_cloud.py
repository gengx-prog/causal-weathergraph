"""Resumable ISCCP HGG Basic worker; default is PLAN, never full download.

Modes: --mode plan | pilot | full. Pilot is 2010-01-01 only. The revision's
acquisition_scope.json controls whether full mode is in scope. NetCDF operations
are locked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision.probe_external_downloads import decode_isccp_cloud

NETCDF_LOCK = threading.Lock()
BASE = ("https://www.ncei.noaa.gov/data/"
        "international-satellite-cloud-climate-project-isccp-h-series-data/"
        "access/isccp-basic/hgg/")
FULL_START, FULL_END = date(1983, 7, 1), date(2017, 6, 30)
DEFAULT_OUTPUT = Path(r"D:\Paper2\Major Revision\supplementary_data\cloud_validation\isccp_hgg")
ACQUISITION_SCOPE = Path(__file__).with_name("acquisition_scope.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def tasks(start, end):
    day = start
    while day <= end:
        for hour in (0, 6, 12, 18):
            name = f"ISCCP-Basic.HGG.v01r00.GLOBAL.{day:%Y.%m.%d}.{hour:02d}00.GPC.10KM.CS00.EA1.00.nc"
            yield {"name": name, "date": day.isoformat(), "hour": hour,
                   "relative_path": f"{day:%Y/%m}/{name}",
                   "url": BASE + f"{day:%Y%m}/" + name}
        day += timedelta(days=1)


def validate(path, task):
    with NETCDF_LOCK:
        cloud, info = decode_isccp_cloud(path)
        expected = np.datetime64(f"{task['date']}T{task['hour']:02d}:00:00", "ns")
        assert cloud.shape == (1, 180, 360)
        assert cloud.time.values[0] == expected
        assert np.allclose(np.diff(cloud.lat), 1) and np.allclose(np.diff(cloud.lon), 1)
        info.update(shape=list(cloud.shape), time=str(expected),
                    min_percent=float(cloud.min()), max_percent=float(cloud.max()),
                    missing_fraction=float(np.isnan(cloud.values).mean()))
    return info


def process(task, output, previous, attempts, seed_dir):
    path = output / task["relative_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = previous.get(task["relative_path"])
    if path.exists() and prior and prior.get("status") == "validated":
        st = path.stat()
        if st.st_size == prior.get("bytes") and st.st_mtime_ns == prior.get("mtime_ns"):
            return {**prior, "cached": True, "transferred_this_run": 0, "checked_utc": now()}
    transferred = 0
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            from_seed = False
            if not path.exists() and (seed_dir / task["name"]).is_file():
                shutil.copy2(seed_dir / task["name"], path)
                from_seed = True
            if not path.exists():
                partial = path.with_suffix(".nc.part")
                # Restart a partial file rather than append without verifying HTTP Range.
                with requests.get(task["url"], stream=True, timeout=(10, 30)) as response:
                    response.raise_for_status()
                    length = int(response.headers.get("Content-Length", 0))
                    if length > 100_000_000:
                        raise ValueError("Unexpectedly large single snapshot")
                    file_bytes = 0
                    with partial.open("wb") as handle:
                        for block in response.iter_content(1024 * 1024):
                            file_bytes += len(block)
                            transferred += len(block)
                            if file_bytes > 100_000_000:
                                raise ValueError("Single snapshot exceeded100MB")
                            handle.write(block)
                validate(partial, task)
                partial.replace(path)
            info = validate(path, task)
            st = path.stat()
            return {**task, "status": "validated", "path": str(path),
                    "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "validation": info, "attempts": attempt, "cached": transferred == 0,
                    "copied_from_pilot": from_seed, "transferred_this_run": transferred,
                    "checked_utc": now()}
        except Exception as exc:
            errors.append(type(exc).__name__)
            if path.exists():
                # Preserve an invalid file for inspection rather than silently delete.
                quarantine = path.with_suffix(f".nc.invalid.{int(time.time())}")
                path.replace(quarantine)
            if attempt < attempts:
                time.sleep(min(2 ** attempt, 4))
    return {**task, "status": "failed", "attempts": attempts, "error_types": errors,
            "transferred_this_run": transferred, "checked_utc": now()}


def load_journal(path):
    previous = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                previous[item["relative_path"]] = item
            except (ValueError, KeyError):
                continue
    return previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plan", "pilot", "full"), default="plan")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    assert 1 <= args.workers <= 4 and 1 <= args.attempts <= 3
    if args.mode == "full":
        policy = json.loads(ACQUISITION_SCOPE.read_text(encoding="utf-8"))
        if not policy["datasets"]["isccp_hgg"]["full_download_allowed"]:
            parser.error(
                "Full ISCCP acquisition is excluded by acquisition_scope.json: "
                "the October 20 revision uses a limited data budget and this "
                "archive does not cover the 2019-2025 evaluation."
            )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    seed_dir = output.parent / "pilot_isccp_20100101_raw"
    pilot_sizes = [p.stat().st_size for p in seed_dir.glob("*.nc")]
    estimated_bytes_each = float(np.mean(pilot_sizes)) if pilot_sizes else 3_000_000.
    count = ((FULL_END - FULL_START).days + 1) * 4
    plan = {"created_utc": now(), "mode": args.mode, "product": "ISCCP HGG Basic v01r00",
            "start": FULL_START.isoformat(), "end": FULL_END.isoformat(), "utc_hours": [0, 6, 12, 18],
            "expected_snapshots": count, "estimated_bytes_per_snapshot": estimated_bytes_each,
            "estimated_full_bytes": int(estimated_bytes_each * count),
            "estimate_scope": "From four2010-01-01 pilot snapshots, not full inventory; historical sizes and missing files may differ.",
            "available_disk_bytes": shutil.disk_usage(output).free,
            "concurrency": args.workers, "attempts_max": args.attempts,
            "missing_policy": "Log failures; no temporal or spatial interpolation.",
            "resume_policy": "Reuse validated unchanged files from journal; retry missing/failed files. Partial transfers restart at file boundary.",
            "netcdf_thread_safety": "All NetCDF reads/decodes protected by a shared lock.",
            "full_download_started": args.mode == "full"}
    atomic_json(output / "download_plan.json", plan)
    if args.mode == "plan":
        print(json.dumps(plan, ensure_ascii=False)); return
    start, end = (date(2010, 1, 1), date(2010, 1, 1)) if args.mode == "pilot" else (FULL_START, FULL_END)
    target_count = ((end - start).days + 1) * 4
    journal = output / "download_manifest.jsonl"
    previous = load_journal(journal)
    state_path = output / ("pilot_state.json" if args.mode == "pilot" else "download_state.json")
    state = {"started_utc": now(), "mode": args.mode, "status": "running", "expected": target_count,
             "completed": 0, "validated": 0, "failed": 0, "cached": 0, "bytes_transferred": 0,
             "full_download_started": args.mode == "full"}
    atomic_json(state_path, state)
    iterator = iter(tasks(start, end))
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, journal.open("a", encoding="utf-8") as log:
        pending = {}
        def refill():
            while len(pending) < args.workers * 2:
                task = next(iterator, None)
                if task is None:
                    break
                future = pool.submit(process, task, output, previous, args.attempts, seed_dir)
                pending[future] = task
        refill()
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                task = pending.pop(future)
                try:
                    record = future.result()
                except Exception as exc:
                    record = {**task, "status": "failed", "error_types": [type(exc).__name__], "transferred_this_run": 0}
                log.write(json.dumps(record, ensure_ascii=False) + "\n"); log.flush()
                state["completed"] += 1
                state["validated"] += record["status"] == "validated"
                state["failed"] += record["status"] == "failed"
                state["cached"] += record.get("cached", False)
                state["bytes_transferred"] += record.get("transferred_this_run", 0)
                state["last_file"] = task["name"]
                state["updated_utc"] = now()
                if record["status"] == "failed": failed.append(record)
            atomic_json(state_path, state)
            refill()
    state.update(status="complete" if not failed else "complete_with_gaps", completed_utc=now())
    atomic_json(state_path, state)
    atomic_json(output / ("pilot_gaps.json" if args.mode == "pilot" else "download_gaps.json"), failed)
    print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()
