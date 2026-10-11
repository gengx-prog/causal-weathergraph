"""Resumable CDS acquisition of the primary fields on the native 0.25-degree grid.

The 2023-01-11--2025 continuation of the primary record (850-hPa temperature,
specific humidity, u and v wind, and total cloud cover) was previously requested
on the CDS 5.625-degree grid, which interpolates instead of area-averaging. The
same-date comparison (run_source_overlap_comparison.py) showed that this route
departs from the WeatherBench 2 conservative product, e.g. +0.69 m/s in 850-hPa
wind speed. Here the same five fields are retrieved on the native grid so that
they can be remapped with the WB2 conservative method
(regrid_era5_primary_native.py). The five pre-splice overlap windows (152
timestamps) come first, so that the remapped route can be checked against WB2
before the long batch is used.

With --route gba the same requests ask the CDS (MARS) to perform the area-weighted
grid-box-average remapping onto the WB2 64x32 grid server-side; on the overlap
windows this reproduces WB2 to about 1e-5 SD while returning ~1/300 of the bytes.

Several requests are kept queued at the CDS at once and finished results are
downloaded in parallel with byte-range resumption. Request ids are persisted, so
a restart resumes the queued requests instead of resubmitting them. Credentials
are read by cdsapi from the local configuration and are never logged; signed
result URLs are kept in memory only.
"""
from __future__ import annotations

import argparse
import calendar
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psutil
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
ROUTES = {
    'native': dict(output='era5_primary_native', grid=[0.25, 0.25], area=[90, 0, -90, 359.75], extra={},
                   lat=np.arange(-90, 90.001, .25), lon=np.arange(0, 360, .25), hard_cap=50_000_000_000,
                   description='Native CDS regular 0.25 degrees; remapped later with the WB2 conservative method.'),
    'gba': dict(output='era5_primary_gba', grid=[5.625, 5.625], area=[87.1875, 0, -87.1875, 354.375],
                extra={'interpolation': 'grid-box-average'},
                lat=np.linspace(-87.1875, 87.1875, 32), lon=np.arange(0, 360, 5.625), hard_cap=2_000_000_000,
                description='CDS/MARS grid-box-average (area-weighted) remapping to the WB2 64x32 grid, server-side.'),
}
ACTIVE = dict(ROUTES['native'], name='native')
FREE_DISK_MARGIN = 20_000_000_000
TIMES = ['00:00', '06:00', '12:00', '18:00']
GROUPS = [
    ('primary_850', 'reanalysis-era5-pressure-levels',
     ['temperature', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind'], ['850'], ['t', 'q', 'u', 'v']),
    ('total_cloud_cover', 'reanalysis-era5-single-levels', ['total_cloud_cover'], None, ['tcc']),
]
# Same windows as the WB2/CDS same-date comparison of the coarse route.
OVERLAP_BLOCKS = [(2022, m, 1, 7) for m in (1, 4, 7, 10)] + [(2023, 1, 1, 10)]
UNITS = {'t': {'K'}, 'q': {'kg kg**-1', 'kg kg-1'}, 'u': {'m s**-1', 'm s-1'}, 'v': {'m s**-1', 'm s-1'},
         'tcc': {'(0 - 1)', '1', '0-1', 'dimensionless'}}
TERMINAL_FAILURES = {'failed', 'dismissed', 'deleted', 'rejected'}
MAX_SUBMISSIONS = 3
MAX_DOWNLOAD_FAILURES = 6
# Interrupted transfers that still made progress are resumed without limit; only
# consecutive attempts without any new bytes count, with exponential backoff.
MAX_STALLS = 12
RETRYABLE_REASONS = {'repeated download failures', 'no download progress', 'status unreadable', 'result unavailable'}
# netCDF4/HDF5 reads are not thread-safe; serialise the ingestion check.
NETCDF_LOCK = threading.Lock()


class ReviewRequired(RuntimeError):
    pass


class TransientDownloadError(RuntimeError):
    """A transfer interrupted by the network; `gained` new bytes were kept for resumption."""

    def __init__(self, gained, cause):
        super().__init__(f'{type(cause).__name__} after {gained / 1e6:.1f} MB')
        self.gained, self.cause = gained, cause


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f'{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z {message}', flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for _ in range(40):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            # On Windows a concurrent reader can briefly block the replacement.
            time.sleep(0.25)
    tmp.replace(path)


def occupied_bytes(out):
    return sum(p.stat().st_size for p in out.rglob('*') if p.is_file())


def make_jobs(route='native'):
    cfg = ROUTES[route]

    def request(variables, levels, year, month, first, last):
        req = dict(product_type=['reanalysis'], variable=variables, year=[str(year)], month=[f'{month:02d}'],
                   day=[f'{d:02d}' for d in range(first, last + 1)], time=TIMES,
                   data_format='netcdf', download_format='unarchived', grid=list(cfg['grid']), area=list(cfg['area']),
                   **cfg['extra'])
        if levels:
            req['pressure_level'] = levels
        return req

    jobs = []

    def add(kind, tag, year, month, first, last):
        for group, dataset, variables, levels, short in GROUPS:
            jobs.append(dict(id=f'{group}_{tag}', kind=kind, dataset=dataset,
                             request=request(variables, levels, year, month, first, last), expected_variables=short,
                             first_time=f'{year}-{month:02d}-{first:02d}T00:00:00',
                             last_time=f'{year}-{month:02d}-{last:02d}T18:00:00'))

    for year, month, first, last in OVERLAP_BLOCKS:
        add('overlap', f'{year}{month:02d}_{first:02d}_{last:02d}', year, month, first, last)
    for year in (2023, 2024, 2025):
        for month in range(1, 13):
            first = 11 if (year, month) == (2023, 1) else 1
            add('extension', f'{year}{month:02d}', year, month, first, calendar.monthrange(year, month)[1])

    def times(kind):
        return sorted({t for j in jobs if j['kind'] == kind for t in pd.date_range(j['first_time'], j['last_time'], freq='6h')})

    overlap, extension = times('overlap'), times('extension')
    assert len(jobs) == 82 and len({j['id'] for j in jobs}) == 82
    assert len(overlap) == 152 and overlap[-1] < pd.Timestamp('2023-01-11')
    assert len(extension) == 4344 and extension[0] == pd.Timestamp('2023-01-11') and extension[-1] == pd.Timestamp('2025-12-31 18:00')
    return jobs


def inspect_file(path, job):
    with NETCDF_LOCK, xr.open_dataset(path, engine='netcdf4') as ds:
        tn = 'valid_time' if 'valid_time' in ds.coords else 'time'
        expected = pd.date_range(job['first_time'], job['last_time'], freq='6h')
        if not pd.DatetimeIndex(ds[tn].values).equals(expected):
            raise ReviewRequired('Returned times differ from the frozen request.')
        lat, lon = ds.latitude.values, ds.longitude.values
        if len(lat) != len(ACTIVE['lat']) or len(lon) != len(ACTIVE['lon']):
            raise ReviewRequired('Unexpected grid dimensions.')
        lat_error = float(np.max(np.abs(np.sort(lat) - ACTIVE['lat'])))
        lon_error = float(np.max(np.abs(np.sort(lon % 360) - ACTIVE['lon'])))
        # GRIB1 single-level fields carry coordinates in millidegrees (87.1875 -> 87.188).
        if lat_error > 1e-3 or lon_error > 1e-3:
            raise ReviewRequired('Unexpected grid coordinates.')
        if job['request'].get('pressure_level'):
            ln = next((k for k in ['pressure_level', 'level', 'isobaricInhPa'] if k in ds.coords), None)
            if ln is None or not np.array_equal(np.atleast_1d(ds[ln].values).astype(float), [850.]):
                raise ReviewRequired('Pressure level differs from 850 hPa.')
        info = {}
        for name in job['expected_variables']:
            if name not in ds or ds[name].attrs.get('units') not in UNITS[name]:
                raise ReviewRequired(f'Missing field or unexpected units: {name}')
            planes = ds[name].isel({tn: [0, len(expected) - 1]}).values
            if not np.isfinite(planes).all():
                raise ReviewRequired(f'Nonfinite values in the first/last plane of {name}.')
            # GRIB packing leaves cloud fractions up to ~1e-5 outside [0, 1].
            if name == 'tcc' and (planes.min() < -1e-4 or planes.max() > 1 + 1e-4):
                raise ReviewRequired('Cloud fraction outside 0-1.')
            info[name] = dict(shape=list(ds[name].shape), units=ds[name].attrs['units'],
                              checked_min=float(planes.min()), checked_max=float(planes.max()))
        expver = sorted({str(v) for v in np.atleast_1d(ds['expver'].values)}) if 'expver' in ds.variables else None
    return dict(time_count=len(expected), first_time=str(expected[0]), last_time=str(expected[-1]),
                latitude_max_error_degrees=lat_error, longitude_max_error_degrees=lon_error, expver=expver,
                variables=info, validation_scope='Coordinates, times, level, units; first/last planes finite. '
                                                 'Full values are decoded and checked by the remapping stage.')


def safe_text(exc, secrets):
    text = str(exc)
    for secret in secrets:
        if secret:
            text = text.replace(secret, '[REDACTED]')
    return re.sub(r'\?[^\s\'"]+', '?[REDACTED]', text)[:400]


def credentials():
    config = Path(os.environ.get('CDSAPI_RC', str(Path.home() / '.cdsapirc')))
    values = {}
    if config.is_file():
        import yaml
        parsed = yaml.safe_load(config.read_text(encoding='utf-8-sig')) or {}
        if isinstance(parsed, dict):
            values = parsed
    url = str(os.getenv('CDSAPI_URL') or values.get('url') or '').strip()
    key = str(os.getenv('CDSAPI_KEY') or values.get('key') or '').strip()
    if not (url and key and '<' not in key):
        raise ReviewRequired('No usable local CDS configuration; nothing submitted.')
    return [key, *[part for part in key.split(':') if len(part) > 10]]


def download(location, expected, part, state, out, update):
    """Stream one result to `part`, resuming an earlier partial file when the server allows it."""
    existing = part.stat().st_size if part.exists() else 0
    if expected <= 0 or existing > expected:
        raise ReviewRequired('Unexpected provider size or overlong partial file.')
    if state.get('expected_bytes') not in {None, expected}:
        # A resubmitted request may legitimately produce a different encoding.
        part.unlink(missing_ok=True)
        existing = 0
    if occupied_bytes(out) + expected - existing > ACTIVE['hard_cap']:
        raise ReviewRequired(f"The {ACTIVE['hard_cap'] / 1e9:.0f} GB acquisition cap would be exceeded.")
    if shutil.disk_usage(out).free < expected - existing + FREE_DISK_MARGIN:
        raise ReviewRequired('Free disk space below the 20 GB margin.')
    update(status='downloading', expected_bytes=expected, bytes_downloaded=existing)
    if existing == expected:
        return existing
    headers = {'Accept-Encoding': 'identity'}
    if existing:
        headers['Range'] = f'bytes={existing}-'
        if state.get('etag'):
            headers['If-Range'] = state['etag']
    with requests.get(location, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        if existing and response.status_code != 206:
            existing = 0  # range ignored: restart this file from zero
        update(etag=response.headers.get('ETag'))
        written, last = existing, time.monotonic()
        with part.open('ab' if existing else 'wb') as f:
            for chunk in response.iter_content(1024 * 1024):
                if not chunk:
                    continue
                if written + len(chunk) > expected:
                    raise ReviewRequired('Provider sent more bytes than advertised.')
                f.write(chunk)
                written += len(chunk)
                if time.monotonic() - last >= 15:
                    update(bytes_downloaded=written)
                    last = time.monotonic()
        update(bytes_downloaded=written)
        if written != expected:
            raise requests.ConnectionError('Incomplete download; range resume required.')
    return existing


def fetch(job, state, location, expected, out, update):
    """Download, check and commit one finished request (runs in a worker thread)."""
    target = out / job['kind'] / (job['id'] + '.nc')
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    before = part.stat().st_size if part.exists() else 0
    t0 = time.monotonic()
    try:
        start_offset = download(location, expected, part, state, out, update)
    except ReviewRequired:
        raise
    except Exception as exc:
        after = part.stat().st_size if part.exists() else 0
        raise TransientDownloadError(max(after - before, 0), exc) from exc
    seconds = time.monotonic() - t0
    size = part.stat().st_size
    quality = inspect_file(part, job)
    checksum = digest(part)
    update(status='verified_pending_rename', sha256=checksum, bytes_downloaded=size, quality=quality)
    part.replace(target)
    return dict(file=str(target.relative_to(out)), size=size, new_bytes=size - start_offset,
                rate=(size - start_offset) / max(seconds, 1e-3) / 1e6)


def run(out, max_inflight, max_downloads, poll_seconds, max_hours, retry_failed=False, route='native'):
    ACTIVE.update(ROUTES[route], name=route)
    secrets = credentials()
    jobs = make_jobs(route)
    plan_path = out / 'request_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if json_digest(plan['jobs']) != json_digest(jobs):
            raise ReviewRequired('Requests differ from the frozen plan.')
    else:
        out.mkdir(parents=True, exist_ok=True)
        plan = dict(frozen_utc=utcnow(), job_count=len(jobs), overlap_timestamps=152, extension_timestamps=4344,
                    hard_cap_bytes=ACTIVE['hard_cap'], route=route, grid=ACTIVE['description'],
                    purpose='Replace the CDS 5.625-degree interpolation route of the 2023-01-11--2025 primary fields by '
                            'native fields remapped with the WeatherBench 2 conservative method; overlap windows verify the route.',
                    code_sha256=digest(__file__), jobs=jobs)
        write_json(plan_path, plan)
    code_sha = digest(__file__)
    snapshot = out / f'code_snapshot_{code_sha[:12]}.py'
    if not snapshot.exists():
        shutil.copyfile(__file__, snapshot)
    lock = out / 'worker.lock'
    if lock.exists():
        pid = int(lock.read_text(encoding='ascii').strip() or 0)
        if pid and psutil.pid_exists(pid) and pid != os.getpid():
            raise ReviewRequired(f'Another worker (pid {pid}) holds the lock.')
        lock.unlink()
    lock.write_text(str(os.getpid()), encoding='ascii')

    states = {}
    for job in jobs:
        p = out / 'jobs' / (job['id'] + '.json')
        states[job['id']] = (json.loads(p.read_text(encoding='utf-8')) if p.exists() else
                             dict(job_id=job['id'], status='not_submitted', request_sha256=json_digest(job),
                                  request_id=None, submissions=0, download_failures=0))
    for job in jobs:
        s = states[job['id']]
        if s['request_sha256'] != json_digest(job):
            raise ReviewRequired('Stored request fingerprint mismatch: ' + job['id'])
        if s['status'] == 'completed':
            target = out / s['file']
            if not target.exists() or target.stat().st_size != s['bytes_downloaded']:
                raise ReviewRequired('A completed file is missing or changed: ' + job['id'])
        if retry_failed and s['status'] == 'failed_review_required' and s.get('review_reason') in RETRYABLE_REASONS:
            # Network failures, not data problems: keep the request id and any partial file.
            s.update(status='submitted' if s.get('request_id') else 'not_submitted', download_failures=0, stalls=0,
                     review_reason=None, retried_utc=utcnow())
            write_json(out / 'jobs' / (job['id'] + '.json'), s)
            log(f'reset for retry: {job["id"]}')

    status = dict(status='starting', pid=os.getpid(), started_utc=utcnow(), max_inflight=max_inflight,
                  max_downloads=max_downloads, poll_seconds=poll_seconds, total_jobs=len(jobs), code_sha256=code_sha)
    started = time.monotonic()
    session = {'bytes': 0}
    guard = threading.RLock()
    inflight = []

    def persist():
        with guard:
            done = [s for s in states.values() if s['status'] == 'completed']
            elapsed = time.monotonic() - started
            status.update(updated_utc=utcnow(), completed_jobs=len(done),
                          completed_bytes=sum(s['bytes_downloaded'] for s in done),
                          session_bytes=session['bytes'], elapsed_hours=round(elapsed / 3600, 3),
                          session_rate_mb_s=round(session['bytes'] / max(elapsed, 1) / 1e6, 2),
                          inflight=[dict(job_id=j['id'], provider_status=states[j['id']].get('provider_status'),
                                         bytes_downloaded=states[j['id']].get('bytes_downloaded', 0)) for j, _ in inflight],
                          failed=[s['job_id'] for s in states.values() if s['status'] == 'failed_review_required'])
            write_json(out / 'status.json', status)

    def update(state, **values):
        with guard:
            state.update(values)
            state['updated_utc'] = utcnow()
            write_json(out / 'jobs' / (state['job_id'] + '.json'), state)
        persist()

    logging.disable(logging.CRITICAL)
    import cdsapi
    silent = lambda *a, **k: None
    client = cdsapi.Client(quiet=True, debug=False, progress=False, timeout=120, retry_max=5, sleep_max=60,
                           wait_until_complete=False, info_callback=silent, warning_callback=silent,
                           error_callback=silent, debug_callback=silent)
    queue = deque(j for j in jobs if states[j['id']]['status'] not in {'completed', 'failed_review_required'})
    pool = ThreadPoolExecutor(max_workers=max_downloads)
    downloading, retry_after = {}, {}
    log(f'start: {len(queue)} of {len(jobs)} jobs to do, max_inflight={max_inflight}, max_downloads={max_downloads}')
    try:
        while queue or inflight:
            if time.monotonic() - started > max_hours * 3600 and not downloading:
                status['status'] = 'paused_time_limit_resume_with_same_command'
                persist()
                log('time limit reached; queued request ids are kept for resumption')
                return 2
            while queue and len(inflight) < max_inflight:
                job = queue.popleft()
                state = states[job['id']]
                try:
                    if state.get('request_id'):
                        remote = client.client.get_remote(state['request_id'])
                        log(f'resumed {job["id"]} ({state["request_id"][:8]})')
                    else:
                        remote = client.retrieve(job['dataset'], job['request'])
                        update(state, request_id=remote.request_id, submitted_utc=utcnow(), status='submitted',
                               submissions=state.get('submissions', 0) + 1)
                        log(f'submitted {job["id"]} ({remote.request_id[:8]})')
                except Exception as exc:
                    errors = state.get('submit_errors', 0) + 1
                    log(f'submit error {job["id"]} ({errors}): {type(exc).__name__}: {safe_text(exc, secrets)}')
                    if errors >= 10:
                        update(state, request_id=None, submit_errors=errors, status='failed_review_required',
                               review_reason='repeated submission errors')
                    else:
                        update(state, request_id=None, submit_errors=errors)
                        queue.appendleft(job)
                    time.sleep(60)
                    break
                inflight.append((job, remote))
            progressed = False
            for item in list(inflight):
                job, remote = item
                jid, state = job['id'], states[job['id']]
                if jid in downloading:
                    future = downloading[jid]
                    if not future.done():
                        continue
                    del downloading[jid]
                    try:
                        info = future.result()
                    except ReviewRequired as exc:
                        if 'cap' in str(exc) or 'disk' in str(exc):
                            raise
                        log(f'REJECTED {jid}: {exc}')
                        part = out / job['kind'] / (jid + '.nc.part')
                        if part.exists():
                            part.replace(part.with_name(part.name + '.rejected'))
                        inflight.remove(item)
                        update(state, status='failed_review_required', review_reason=str(exc))
                        continue
                    except TransientDownloadError as exc:
                        stalls = 0 if exc.gained > 0 else state.get('stalls', 0) + 1
                        wait = 15 if exc.gained > 0 else min(60 * 2 ** (stalls - 1), 1800)
                        retry_after[jid] = time.monotonic() + wait
                        http_status = getattr(getattr(exc.cause, 'response', None), 'status_code', None)
                        log(f'interrupted {jid}: +{exc.gained / 1e6:.1f} MB kept, stalls={stalls}, retry in {wait}s '
                            f'({type(exc.cause).__name__}{f" HTTP {http_status}" if http_status else ""})')
                        if http_status in (403, 404, 410):
                            # The provider no longer serves this result: resubmit from scratch.
                            inflight.remove(item)
                            (out / job['kind'] / (jid + '.nc.part')).unlink(missing_ok=True)
                            if state.get('submissions', 0) < MAX_SUBMISSIONS:
                                update(state, request_id=None, status='result_unavailable_resubmit', stalls=0,
                                       expected_bytes=None, etag=None)
                                queue.append(job)
                            else:
                                update(state, status='failed_review_required', review_reason='result unavailable')
                        elif stalls >= MAX_STALLS:
                            inflight.remove(item)
                            update(state, stalls=stalls, status='failed_review_required', review_reason='no download progress')
                        else:
                            update(state, stalls=stalls)
                        continue
                    except Exception as exc:
                        failures = state.get('download_failures', 0) + 1
                        log(f'download error {jid} ({failures}): {type(exc).__name__}: {safe_text(exc, secrets)}')
                        if failures >= MAX_DOWNLOAD_FAILURES:
                            inflight.remove(item)
                            update(state, download_failures=failures, status='failed_review_required',
                                   review_reason='repeated download failures')
                        else:
                            update(state, download_failures=failures)
                        continue
                    with guard:
                        session['bytes'] += info['new_bytes']
                    inflight.remove(item)
                    update(state, status='completed', completed_utc=utcnow(), file=info['file'])
                    progressed = True
                    log(f'completed {jid}: {info["size"] / 1e6:.1f} MB at {info["rate"]:.1f} MB/s; '
                        f'{status["completed_jobs"]}/{len(jobs)} done, {status["completed_bytes"] / 1e9:.2f} GB, '
                        f'session {status["session_rate_mb_s"]:.1f} MB/s')
                    continue
                try:
                    provider = remote.status
                except Exception as exc:
                    errors = state.get('poll_errors', 0) + 1
                    log(f'poll error {jid} ({errors}): {type(exc).__name__}: {safe_text(exc, secrets)}')
                    if errors % 40 == 0:
                        # About 20 minutes without a readable status: drop this request id.
                        inflight.remove(item)
                        if state.get('submissions', 0) < MAX_SUBMISSIONS:
                            update(state, poll_errors=errors, request_id=None, status='poll_failed_resubmit')
                            queue.append(job)
                        else:
                            update(state, poll_errors=errors, request_id=None, status='failed_review_required',
                                   review_reason='status unreadable')
                    else:
                        with guard:
                            state['poll_errors'] = errors
                    continue
                if provider != state.get('provider_status'):
                    log(f'{jid}: {provider}')
                    update(state, provider_status=provider, provider_checked_utc=utcnow())
                if provider in TERMINAL_FAILURES:
                    inflight.remove(item)
                    if state.get('submissions', 0) < MAX_SUBMISSIONS:
                        update(state, request_id=None, status='provider_' + provider)
                        queue.append(job)
                    else:
                        update(state, request_id=None, status='failed_review_required', review_reason='provider ' + provider)
                    continue
                if provider != 'successful' or len(downloading) >= max_downloads or time.monotonic() < retry_after.get(jid, 0):
                    continue
                try:
                    result = remote.get_results()
                    location, expected = result.location, int(result.content_length)
                except Exception as exc:
                    log(f'result lookup error {jid}: {type(exc).__name__}: {safe_text(exc, secrets)}')
                    continue
                downloading[jid] = pool.submit(fetch, job, state, location, expected, out,
                                               lambda s=state, **kw: update(s, **kw))
            status['status'] = 'running'
            persist()
            if not progressed:
                time.sleep(poll_seconds if not downloading else min(poll_seconds, 10))
        failed = [s['job_id'] for s in states.values() if s['status'] == 'failed_review_required']
        status['status'] = 'completed' if not failed else 'completed_with_failures_review_required'
        write_json(out / 'download_manifest.json', dict(
            completed_utc=utcnow(), plan_sha256=digest(plan_path), code_sha256=code_sha,
            total_bytes=sum(s.get('bytes_downloaded', 0) for s in states.values() if s['status'] == 'completed'),
            files=[{k: v for k, v in s.items() if k not in {'etag'}} for s in states.values()]))
        persist()
        log(f'finished: {status["status"]}')
        return 0 if not failed else 1
    except Exception as exc:
        status.update(status='stopped_review_required', error_type=type(exc).__name__, error=safe_text(exc, secrets))
        persist()
        log(f'stopped: {type(exc).__name__}: {safe_text(exc, secrets)}')
        return 1
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        if lock.exists() and lock.read_text(encoding='ascii') == str(os.getpid()):
            lock.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--route', choices=sorted(ROUTES), default='native')
    ap.add_argument('--output', type=Path, help='Defaults to supplementary_data/<route output>.')
    ap.add_argument('--plan-only', action='store_true', help='Validate the request plan; submit nothing.')
    ap.add_argument('--max-inflight', type=int, default=8)
    ap.add_argument('--max-downloads', type=int, default=3)
    ap.add_argument('--poll-seconds', type=int, default=30)
    ap.add_argument('--max-hours', type=float, default=36)
    ap.add_argument('--retry-failed', action='store_true',
                    help='Requeue jobs that failed for network reasons, keeping request ids and partial files.')
    args = ap.parse_args()
    args.output = args.output or ROOT.parent / 'supplementary_data' / ROUTES[args.route]['output']
    if args.plan_only:
        jobs = make_jobs(args.route)
        print(json.dumps({'jobs': len(jobs), 'overlap_jobs': sum(j['kind'] == 'overlap' for j in jobs),
                          'extension_jobs': sum(j['kind'] == 'extension' for j in jobs),
                          'first': jobs[0]['id'], 'last': jobs[-1]['id'],
                          'free_disk_gb': round(shutil.disk_usage(args.output.parent).free / 1e9, 1)}, indent=1))
        return 0
    if not 1 <= args.max_inflight <= 12 or not 1 <= args.max_downloads <= args.max_inflight or args.poll_seconds < 10:
        ap.error('need 1 <= max-downloads <= max-inflight <= 12 and poll-seconds >= 10')
    return run(args.output, args.max_inflight, args.max_downloads, args.poll_seconds, args.max_hours, args.retry_failed, args.route)


if __name__ == '__main__':
    sys.exit(main())
