"""Bounded, resumable CDS acquisition for same-date source/processing comparisons.

Uses the existing local CDS credentials without logging them.  Does not modify
the primary ERA5 inputs or the earlier 144-job control-field acquisition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT.parent / 'supplementary_data/era5_source_overlap_2022_2023'
HARD_CAP = 4_000_000_000
BOOKKEEPING_RESERVE = 10_000_000
BLOCKS = [(2022, m, 7) for m in [1, 4, 7, 10]] + [(2023, 1, 10)]
# Coarse primary fields go first so that their much smaller responses can be
# checked while the larger native control requests await processing.
GROUPS = [
    ('primary_850_coarse', 'pressure', ['u_component_of_wind', 'v_component_of_wind', 'specific_humidity', 'temperature'], ['850'], ['u', 'v', 'q', 't'], 'coarse'),
    ('cloud_cover_coarse', 'single', ['total_cloud_cover'], None, ['tcc'], 'coarse'),
    ('omega_500_700_native', 'pressure', ['vertical_velocity'], ['500', '700'], ['w'], 'native'),
    ('geopotential_500_native', 'pressure', ['geopotential'], ['500'], ['z'], 'native'),
    ('temperature_700_native', 'pressure', ['temperature'], ['700'], ['t'], 'native'),
    ('surface_pressure_msl_native', 'single', ['surface_pressure', 'mean_sea_level_pressure'], None, ['sp', 'msl'], 'native'),
]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def occupied_bytes(out):
    # Count ALL retained bytes in this acquisition directory, including partials,
    # receipts, snapshots and logs.  Never scan or remove unrelated directories.
    return sum(p.stat().st_size for p in out.rglob('*') if p.is_file())


class BudgetExceeded(RuntimeError):
    pass


class ReviewRequired(RuntimeError):
    pass


def check_capacity(current, additional):
    if additional < 0 or current + additional + BOOKKEEPING_RESERVE > HARD_CAP:
        raise BudgetExceeded('Frozen 4 GB acquisition cap, including partial files, would be exceeded.')


def make_jobs():
    jobs = []
    for year, month, last_day in BLOCKS:
        for group, kind, variables, levels, short, grid in GROUPS:
            req = dict(product_type=['reanalysis'], variable=variables, year=[str(year)],
                       month=[f'{month:02d}'], day=[f'{d:02d}' for d in range(1, last_day + 1)],
                       time=['00:00', '06:00', '12:00', '18:00'],
                       data_format='netcdf', download_format='unarchived')
            if grid == 'native':
                req.update(grid=[.25, .25], area=[90, 0, -90, 359.75])
            else:
                req.update(grid=[5.625, 5.625], area=[87.1875, 0, -87.1875, 354.375])
            if levels:
                req['pressure_level'] = levels
            jobs.append(dict(id=f'{group}_{year}{month:02d}_01_{last_day:02d}',
                             dataset=f'reanalysis-era5-{kind}-levels', request=req,
                             expected_variables=short, grid_route=grid,
                             first_time=f'{year}-{month:02d}-01T00:00:00',
                             last_time=f'{year}-{month:02d}-{last_day:02d}T18:00:00'))
    assert len(jobs) == 30 and len({j['id'] for j in jobs}) == 30
    times = {str(t) for j in jobs for t in pd.date_range(j['first_time'], j['last_time'], freq='6h')}
    assert len(times) == 152 and max(times) < '2023-01-11'
    return jobs


def freeze(out):
    if (out / 'request_plan.json').exists() or (out / 'status.json').exists():
        raise ReviewRequired('A frozen plan/status already exists; it will not be overwritten.')
    out.mkdir(parents=True, exist_ok=True)
    if any(p.suffix in {'.nc', '.part'} for p in out.rglob('*') if p.is_file()):
        raise ReviewRequired('Unregistered scientific files found in new output directory.')
    jobs = make_jobs()
    plan = dict(frozen_utc=utcnow(), status='frozen_before_any_submission', jobs=jobs,
                job_count=30, unique_six_hour_times=152, acquisition_hard_cap_bytes=HARD_CAP,
                cap_scope='All files in this package, including .nc.part; 10 MB reserved for bookkeeping.',
                code_sha256=digest(__file__),
                purpose='Same-date WeatherBench2/CDS source and processing-route comparison before the 2023-01-11 seam; not adjacent-date differencing, causal validation, or independent ERA5 observations.',
                selection='2022 Jan/Apr/Jul/Oct days 1-7 and 2023 Jan days 1-10; all four six-hour UTC times; fixed without selecting on comparison outcomes.',
                estimate=dict(native_controls_compressed_bytes_empirical=1_430_603_856,
                              native_controls_float32_bytes=3_787_499_520,
                              five_coarse_primary_float32_bytes=6_225_920,
                              method='Native estimate scales existing 40,885,152,308-byte/4,344-time batch; not a guaranteed server quote.'),
                source_reference_paths=[str(ROOT.parent / 'supplementary_data/era5_controls/shards'),
                                        str(ROOT.parent.parent / 'vipuser/Data/weatherbench2_era5_6h_64x32_850hPa_2019_2023-01-10')],
                validation='All coordinates, times, pressure levels, variables and units; finite/range checks on first and last planes; full-value scientific audit and conservative regridding are later stages.',
                limitations=['All inputs are derived from ERA5; cross-provider agreement is not independent physical validation.',
                             'Native controls and directly requested coarse primary fields follow different existing processing routes.',
                             'The chosen windows do not establish all-year or all-period source homogeneity.',
                             'Original scientific inputs and earlier acquisition directories are preserved.'])
    shutil.copyfile(__file__, out / 'code_snapshot.py')
    write_json(out / 'request_plan.json', plan)
    write_json(out / 'status.json', dict(status='planned_not_submitted', updated_utc=utcnow(),
                                       plan_sha256=digest(out / 'request_plan.json'), total_jobs=30,
                                       completed_jobs=0, active_job=None, occupied_bytes=occupied_bytes(out)))
    print('Frozen 30 requests / 152 distinct times; no requests submitted.', flush=True)


def inspect_file(path, job):
    accepted_units = {'u': {'m s**-1', 'm s-1', 'm/s'}, 'v': {'m s**-1', 'm s-1', 'm/s'},
                      'q': {'kg kg**-1', 'kg kg-1'}, 't': {'K'}, 'w': {'Pa s**-1', 'Pa s-1', 'Pa/s'},
                      'z': {'m**2 s**-2', 'm2 s-2'}, 'sp': {'Pa'}, 'msl': {'Pa'},
                      'tcc': {'(0 - 1)', '1', '0-1', 'dimensionless'}}
    with xr.open_dataset(path, engine='netcdf4') as ds:
        tn = 'valid_time' if 'valid_time' in ds.coords else 'time'
        expected = pd.date_range(job['first_time'], job['last_time'], freq='6h')
        if not pd.DatetimeIndex(ds[tn].values).equals(expected):
            raise ReviewRequired('Returned time coordinates do not match the frozen dates.')
        lat, lon = ds.latitude.values, ds.longitude.values
        native = job['grid_route'] == 'native'
        expected_lat = np.arange(-90, 90.001, .25) if native else np.linspace(-87.1875, 87.1875, 32)
        expected_lon = np.arange(0, 360, .25 if native else 5.625)
        if len(lat) != len(expected_lat) or len(lon) != len(expected_lon):
            raise ReviewRequired('Grid dimensions do not match request route.')
        lat_error = float(np.max(np.abs(np.sort(lat) - expected_lat)))
        lon_error = float(np.max(np.abs(np.sort(lon % 360) - expected_lon)))
        if lat_error > .001 or lon_error > .001:
            raise ReviewRequired('Unexpected grid coordinates.')
        levels = job['request'].get('pressure_level')
        if levels:
            ln = next((k for k in ['pressure_level', 'level', 'isobaricInhPa'] if k in ds.coords), None)
            if ln is None or not np.array_equal(np.sort(np.atleast_1d(ds[ln].values)), sorted(map(int, levels))):
                raise ReviewRequired('Pressure levels do not match the selected values.')
        info = {}
        for name in job['expected_variables']:
            if name not in ds or ds[name].attrs.get('units') not in accepted_units[name]:
                raise ReviewRequired('Returned variable names or units differ from the frozen design.')
            da = ds[name]
            planes = da.isel({tn: [0, len(expected) - 1]}).values
            if not np.isfinite(planes).all():
                raise ReviewRequired('Nonfinite values in boundary-plane ingestion check.')
            if name == 'tcc' and (planes.min() < -1e-5 or planes.max() > 1.00001):
                raise ReviewRequired('Cloud fraction is outside the expected 0-1 scale.')
            info[name] = dict(shape=list(da.shape), units=da.attrs['units'],
                              checked_min=float(planes.min()), checked_max=float(planes.max()))
        return dict(time_count=len(expected), first_time=str(expected[0]), last_time=str(expected[-1]),
                    latitude_max_error_degrees=lat_error, longitude_max_error_degrees=lon_error,
                    variables=info, validation_scope='All coordinate/time/level/unit metadata; first/last time planes only, not a full-value audit.')


def bounded_download(result, part, state, out, save):
    expected = int(result.content_length)
    existing = part.stat().st_size if part.exists() else 0
    if expected <= 0 or existing > expected:
        raise ReviewRequired('Unexpected provider size or overlong partial file.')
    check_capacity(occupied_bytes(out), expected - existing)
    if state.get('expected_bytes') not in {None, expected}:
        raise ReviewRequired('Provider size changed for the recorded request; partial retained.')
    state.update(status='downloading', expected_bytes=expected, bytes_downloaded=existing)
    save()
    if existing == expected:
        return
    headers = {'Accept-Encoding': 'identity'}
    if existing:
        headers['Range'] = f'bytes={existing}-'
        if state.get('etag'):
            headers['If-Range'] = state['etag']
    # Results URLs may be signed. Keep them only in memory; never serialize or
    # log them. No CDS token/header is forwarded to the result host.
    with requests.get(result.location, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if existing:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if response.status_code != 206 or not match or int(match[1]) != existing or int(match[3]) != expected:
                raise ReviewRequired('Server did not confirm exact byte-range continuation; partial retained.')
            if state.get('etag') and response.headers.get('ETag') != state['etag']:
                raise ReviewRequired('ETag changed; partial retained for review.')
        elif response.status_code != 200:
            raise ReviewRequired('Unexpected initial download status.')
        if response.headers.get('Content-Length') and int(response.headers['Content-Length']) != expected - existing:
            raise ReviewRequired('HTTP content length disagrees with the result metadata.')
        state['etag'] = response.headers.get('ETag')
        save()
        written, last_report = existing, time.monotonic()
        mode = 'ab' if existing else 'xb'
        with part.open(mode) as f:
            for chunk in response.iter_content(512 * 1024):
                if not chunk:
                    continue
                if written + len(chunk) > expected:
                    raise ReviewRequired('Provider sent more bytes than advertised.')
                check_capacity(occupied_bytes(out), len(chunk))
                f.write(chunk)
                f.flush()
                written += len(chunk)
                if time.monotonic() - last_report >= 5:
                    state['bytes_downloaded'] = written
                    save()
                    last_report = time.monotonic()
        state['bytes_downloaded'] = written
        save()
        if written != expected:
            raise requests.ConnectionError('Incomplete download; exact-range resume required.')


def run(out, poll_seconds, max_hours):
    plan_path = out / 'request_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['code_sha256'] != digest(__file__) or plan['acquisition_hard_cap_bytes'] != HARD_CAP:
        raise ReviewRequired('Executable or hard cap differs from the frozen protocol.')
    if json_digest(plan['jobs']) != json_digest(make_jobs()):
        raise ReviewRequired('Requests differ from the frozen protocol.')
    if shutil.disk_usage(out).free < HARD_CAP + BOOKKEEPING_RESERVE:
        raise BudgetExceeded('Less free disk space than the full frozen package allowance.')
    lock = out / 'worker.lock'
    if lock.exists():
        raise ReviewRequired('Worker lock exists; inspect existing process before a restart.')
    lock.write_text(str(os.getpid()), encoding='ascii')
    status = json.loads((out / 'status.json').read_text(encoding='utf-8'))
    status.update(status='starting', pid=os.getpid(), started_utc=utcnow(),
                  plan_sha256=digest(plan_path), poll_seconds=poll_seconds, max_hours=max_hours)
    states = []
    for job in plan['jobs']:
        p = out / 'jobs' / (job['id'] + '.json')
        states.append(json.loads(p.read_text(encoding='utf-8')) if p.exists() else
                      dict(job_id=job['id'], status='not_submitted', request_sha256=json_digest(job), request_id=None))
    started = time.monotonic()

    def save():
        status.update(updated_utc=utcnow(), completed_jobs=sum(s['status'] == 'completed' for s in states),
                      occupied_bytes=occupied_bytes(out), jobs=[dict(job_id=s['job_id'], status=s['status'],
                                                                   request_id=s.get('request_id'), bytes_downloaded=s.get('bytes_downloaded', 0)) for s in states])
        write_json(out / 'status.json', status)
        if status.get('active_job'):
            state = next(s for s in states if s['job_id'] == status['active_job'])
            state['updated_utc'] = utcnow()
            write_json(out / 'jobs' / (state['job_id'] + '.json'), state)

    try:
        logging.disable(logging.CRITICAL)
        import cdsapi
        silent = lambda *a, **k: None
        client = cdsapi.Client(quiet=True, debug=False, progress=False, timeout=120,
                               retry_max=3, sleep_max=15, wait_until_complete=False,
                               info_callback=silent, warning_callback=silent,
                               error_callback=silent, debug_callback=silent)
        for job, state in zip(plan['jobs'], states):
            status['active_job'] = job['id']
            if state['request_sha256'] != json_digest(job):
                raise ReviewRequired('Stored request fingerprint mismatch.')
            target = out / 'files' / (job['id'] + '.nc')
            target.parent.mkdir(exist_ok=True)
            part = target.with_name(target.name + '.part')
            if state['status'] == 'completed':
                if not target.exists() or digest(target) != state['sha256']:
                    raise ReviewRequired('A completed file is missing or changed.')
                continue
            if target.exists():
                raise ReviewRequired('Unregistered completed file retained for review.')
            if state.get('request_id'):
                remote = client.client.get_remote(state['request_id'])
            else:
                if part.exists():
                    raise ReviewRequired('Partial exists without a recorded request ID.')
                state['status'] = 'submitting'
                status['status'] = 'submitting'
                save()
                remote = client.retrieve(job['dataset'], job['request'])
                state.update(request_id=remote.request_id, submitted_utc=utcnow(), status='submitted')
                save()
            while True:
                if time.monotonic() - started > max_hours * 3600:
                    status['status'] = 'waiting_time_limit_resume_existing_request'
                    save()
                    return
                remote_state = remote.status
                state.update(status=remote_state, provider_checked_utc=utcnow())
                status['status'] = 'waiting_for_provider'
                save()
                if remote_state == 'successful':
                    break
                if remote_state in {'failed', 'dismissed', 'deleted'}:
                    raise ReviewRequired('Provider request failed; request ID retained for inspection.')
                time.sleep(poll_seconds)
            result = remote.get_results()
            for attempt in range(3):
                try:
                    status['status'] = 'downloading'
                    bounded_download(result, part, state, out, save)
                    break
                except requests.RequestException as exc:
                    state.update(status='download_retry', transient_error_type=type(exc).__name__,
                                 download_attempt=attempt + 1)
                    save()
                    if attempt == 2:
                        raise
                    time.sleep(10)
            status['status'] = 'checking_download'
            state['status'] = 'checking_download'
            save()
            quality = inspect_file(part, job)
            checksum = digest(part)
            # Record the verified transfer before rename; restart never silently
            # overwrites an unregistered finished file after an interrupted commit.
            state.update(status='verified_pending_rename', sha256=checksum, bytes_downloaded=part.stat().st_size,
                         quality=quality)
            save()
            part.rename(target)
            state.update(status='completed', completed_utc=utcnow(), file=str(target.relative_to(out)))
            save()
            print('completed ' + job['id'] + ' bytes=' + str(state['bytes_downloaded']), flush=True)
        status.update(status='completed_downloads_pending_full_value_audit_and_regrid', active_job=None)
        write_json(out / 'download_manifest.json', dict(completed_utc=utcnow(), plan_sha256=digest(plan_path),
                   files=states, total_science_bytes=sum(s['bytes_downloaded'] for s in states),
                   validation_scope=plan['validation'], no_regridding_or_source_comparison_yet=True))
        save()
    except Exception as exc:
        # Exception text can contain signed URLs or headers. Store only class and
        # our fixed messages, never arbitrary provider/client exception strings.
        status.update(status='blocked_budget' if isinstance(exc, BudgetExceeded) else 'failed_review_required',
                      error_type=type(exc).__name__,
                      safe_explanation=str(exc) if isinstance(exc, (BudgetExceeded, ReviewRequired)) else
                      'Provider/client operation failed; inspect recorded request ID without exposing credentials.')
        save()
        print('Stopped: ' + status['status'] + ' / ' + type(exc).__name__, flush=True)
        raise SystemExit(1)
    finally:
        if lock.exists() and lock.read_text(encoding='ascii') == str(os.getpid()):
            lock.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze', action='store_true')
    action.add_argument('--run', action='store_true')
    ap.add_argument('--poll-seconds', type=int, default=30)
    ap.add_argument('--max-hours', type=float, default=24)
    args = ap.parse_args()
    if args.poll_seconds < 5 or args.max_hours <= 0:
        ap.error('poll-seconds must be >=5 and max-hours must be positive')
    if args.freeze:
        # Meaningful boundary checks before freezing; no remote or scientific I/O.
        make_jobs()
        check_capacity(HARD_CAP - BOOKKEEPING_RESERVE, 0)
        try:
            check_capacity(HARD_CAP - BOOKKEEPING_RESERVE, 1)
        except BudgetExceeded:
            pass
        else:
            raise AssertionError('Hard-cap boundary enforcement failed.')
        freeze(args.output)
    else:
        run(args.output, args.poll_seconds, args.max_hours)


if __name__ == '__main__':
    main()
