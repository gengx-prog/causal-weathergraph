"""Acquire the ERA5 control fields after the public WeatherBench2 cutoff.

The native CDS regular 0.25-degree grid is retained for subsequent, explicit
conservative remapping; these files are not silently appended to old arrays.
Authentication is read by cdsapi from the user's standard local configuration.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT.parent / 'supplementary_data/era5_controls/cds_extension'
GROUPS = [
    ('omega_500_700', 'reanalysis-era5-pressure-levels', ['vertical_velocity'], ['500', '700'], ['w']),
    ('geopotential_500', 'reanalysis-era5-pressure-levels', ['geopotential'], ['500'], ['z']),
    ('temperature_700', 'reanalysis-era5-pressure-levels', ['temperature'], ['700'], ['t']),
    ('surface_pressure_msl', 'reanalysis-era5-single-levels', ['surface_pressure', 'mean_sea_level_pressure'], None, ['sp', 'msl']),
]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temporary.replace(path)


def jobs(pilot=False):
    result = []
    for year in range(2023, 2026):
        for month in range(1, 13):
            if pilot and (year, month) != (2023, 1):
                continue
            first_day = 11 if (year, month) == (2023, 1) else 1
            last_day = first_day if pilot else calendar.monthrange(year, month)[1]
            for group, dataset, variables, levels, short_names in GROUPS:
                request = {
                    'product_type': ['reanalysis'], 'variable': variables,
                    'year': [str(year)], 'month': [f'{month:02d}'],
                    'day': [f'{d:02d}' for d in range(first_day, last_day + 1)],
                    'time': ['00:00', '06:00', '12:00', '18:00'],
                    'data_format': 'netcdf', 'download_format': 'unarchived',
                    'grid': [0.25, 0.25], 'area': [90, 0, -90, 359.75],
                }
                if levels:
                    request['pressure_level'] = levels
                result.append({'id': f'{group}_{year}{month:02d}' + ('_pilot' if pilot else ''),
                               'dataset': dataset, 'request': request, 'expected_variables': short_names,
                               'first_time': f'{year}-{month:02d}-{first_day:02d}T00:00:00',
                               'last_time': f'{year}-{month:02d}-{last_day:02d}T18:00:00'})
    return result


def inspect(path, job):
    with xr.open_dataset(path, engine='netcdf4') as ds:
        time_name = 'valid_time' if 'valid_time' in ds.coords else 'time'
        expected_time = pd.date_range(job['first_time'], job['last_time'], freq='6h')
        actual_time = pd.DatetimeIndex(ds[time_name].values)
        if not actual_time.equals(expected_time):
            raise ValueError('Downloaded response times differ from the frozen request.')
        lat = ds['latitude'].values; lon = ds['longitude'].values
        if len(lat) != 721 or len(lon) != 1440:
            raise ValueError('Unexpected native-grid dimensions.')
        if not np.allclose(np.sort(lat), np.arange(-90, 90.001, .25)):
            raise ValueError('Unexpected latitude coordinates.')
        if not np.allclose(np.sort(lon % 360), np.arange(0, 360, .25)):
            raise ValueError('Unexpected longitude coordinates.')
        if job['request'].get('pressure_level'):
            level_name = 'pressure_level' if 'pressure_level' in ds.coords else 'level'
            if set(np.atleast_1d(ds[level_name].values).astype(int)) != set(map(int, job['request']['pressure_level'])):
                raise ValueError('Unexpected pressure levels.')
        variables = {}
        for name in job['expected_variables']:
            if name not in ds:
                raise ValueError(f'Missing field {name}')
            da = ds[name]
            # Only read one time slice for a bounded ingestion check; full validation
            # and masked native-to-coarse remapping are a separate, explicit stage.
            values = da.isel({time_name: 0}).values
            variables[name] = {'units': da.attrs.get('units'), 'shape': list(da.shape),
                               'first_slice_nonfinite': int(np.sum(~np.isfinite(values)))}
        return {'time_count': len(actual_time), 'first_time': str(actual_time[0]),
                'last_time': str(actual_time[-1]), 'variables': variables,
                'validation_scope': 'All coordinates/times/required fields; nonfinite profile for the first time slice only.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run', action='store_true', help='Actually retrieve with local CDS credentials.')
    parser.add_argument('--pilot', action='store_true', help='Only 2023-01-11, all six target fields.')
    parser.add_argument('--limit-jobs', type=int)
    args = parser.parse_args(); out = args.output
    if args.pilot:
        out = out / 'pilot'
    out.mkdir(parents=True, exist_ok=True)
    plan_jobs = jobs(args.pilot)
    times = sorted({str(t) for j in plan_jobs for t in pd.date_range(j['first_time'], j['last_time'], freq='6h')})
    if not args.pilot:
        assert len(plan_jobs) == 144 and len(times) == 4344
    plan = {'created_utc': utcnow(), 'target_period': ['2023-01-11', '2025-12-31'],
            'pilot': args.pilot, 'response_times': len(times), 'job_count': len(plan_jobs),
            'grid': 'Native CDS regular 0.25 degrees; not yet remapped or merged.',
            'rationale': 'Missing period after public WeatherBench2 ends 2023-01-10; same six control fields.',
            'jobs': plan_jobs}
    write_json(out/'request_plan.json', plan)
    config = Path(os.environ.get('CDSAPI_RC', str(Path.home()/'.cdsapirc')))
    # A prepared template is not an authenticated configuration. Never submit
    # a job merely because .cdsapirc exists, and never print the token.
    config_values = {}
    config_parse_error = None
    if config.is_file():
        import yaml
        try:
            parsed = yaml.safe_load(config.read_text(encoding='utf-8-sig')) or {}
            if isinstance(parsed, dict):
                config_values = parsed
        except Exception as exc:
            config_parse_error = type(exc).__name__
    url = str(os.getenv('CDSAPI_URL') or config_values.get('url') or '').strip()
    key = str(os.getenv('CDSAPI_KEY') or config_values.get('key') or '').strip()
    configured = bool(url and key and '<' not in key and '>' not in key)
    status = {'updated_utc': utcnow(), 'total_jobs': len(plan_jobs), 'completed_jobs': 0,
              'authentication_config_present': configured,
              'config_file_exists': config.is_file(), 'config_parse_error_type': config_parse_error,
              'status': 'planned_not_submitted' if not args.run else 'starting'}
    if not args.run or not configured:
        if args.run and not configured:
            status['status'] = 'blocked_missing_personal_access_token' if config.is_file() and not key else 'blocked_missing_local_cds_configuration'
        write_json(out/'status.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        return
    import cdsapi
    client = cdsapi.Client(timeout=120, retry_max=3, sleep_max=30)
    complete = 0
    selected = plan_jobs[:args.limit_jobs] if args.limit_jobs else plan_jobs
    for job in selected:
        path = out/'monthly'/(job['id']+'.nc'); path.parent.mkdir(exist_ok=True)
        receipt_path = out/'receipts'/(job['id']+'.json')
        request_sha = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
        if path.is_file() and receipt_path.is_file():
            saved = json.loads(receipt_path.read_text(encoding='utf8'))
            if saved.get('request_sha256') == request_sha and saved.get('sha256') == digest(path):
                complete += 1
                continue
        status.update(status='retrieving', active_job=job['id'], completed_jobs=complete, updated_utc=utcnow())
        write_json(out/'status.json', status)
        print('retrieving ' + job['id'], flush=True)
        partial = path.with_suffix('.partial.nc')
        try:
            client.retrieve(job['dataset'], job['request'], str(partial))
            quality = inspect(partial, job)
            file_sha = digest(partial)
            partial.replace(path)
            receipt = {'downloaded_utc': utcnow(), 'request_sha256': request_sha, 'sha256': file_sha,
                       'bytes': path.stat().st_size, 'file': str(path), 'job': job, 'quality': quality}
            write_json(receipt_path, receipt)
            complete += 1
        except Exception as exc:
            # Do not serialize client objects, headers or credential-bearing configuration.
            explanation = str(exc)
            for secret in [key, *[part for part in key.split(':') if len(part) > 10]]:
                explanation = explanation.replace(secret, '[REDACTED]')
            response = getattr(exc, 'response', None)
            http_status = getattr(response, 'status_code', None)
            status.update(status='failed_review_required', error_type=type(exc).__name__,
                          http_status=http_status, server_explanation=explanation[:3000],
                          active_job=job['id'], completed_jobs=complete, updated_utc=utcnow())
            write_json(out/'status.json', status)
            print('Download failed; inspect local client log and status. Exception type: ' + type(exc).__name__, flush=True)
            raise SystemExit(1)
    status.update(status='completed' if complete == len(plan_jobs) else 'limited_batch_completed',
                  completed_jobs=complete, updated_utc=utcnow(), active_job=None)
    write_json(out/'status.json', status)
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
