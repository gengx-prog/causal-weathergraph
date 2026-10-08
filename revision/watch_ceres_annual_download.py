"""Retrieve only the two authorized annual-extension orders, then run the frozen analysis.

The provider email is received once over loopback and kept in process memory.
No account secret, email, status URL, or unredacted provider HTML is saved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup
import netCDF4
import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[2]
DEST = BASE / 'supplementary_data/cloud_validation/ceres_syn1deg_ed42_2019_extension'
HOST = 'https://ceres-tool.larc.nasa.gov'
ORDERS = [dict(region='north_atlantic', order_id='41742', suborder_id='99005'),
          dict(region='south_atlantic', order_id='41743', suborder_id='99006')]
CAP = 250_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    tmp.replace(path)


def receive_context(port):
    token = secrets.token_urlsafe(32)
    state = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path != '/challenge':
                self.send_error(404); return
            payload = json.dumps({'challenge': token}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(payload)

        def do_POST(self):
            if self.path != '/provider-context':
                self.send_error(404); return
            n = int(self.headers.get('Content-Length', '0'))
            if not 0 < n <= 4096:
                self.send_error(400); return
            try:
                payload = json.loads(self.rfile.read(n))
                email = payload['email']
                assert secrets.compare_digest(payload['challenge'], token)
                assert isinstance(email, str) and '@' in email and len(email) <= 254
                assert '\n' not in email and '\r' not in email
            except (ValueError, KeyError, TypeError, AssertionError):
                self.send_error(400); return
            cookies = payload.get('cookies', [])
            if not isinstance(cookies, list) or any(c.get('domain', '').lstrip('.') != 'ceres-tool.larc.nasa.gov' for c in cookies):
                self.send_error(400); return
            state['context'] = {'email': email, 'cookies': cookies}
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"accepted":true}')

    server = HTTPServer(('127.0.0.1', port), Handler)
    server.timeout = 5
    deadline = time.monotonic() + 300
    while 'context' not in state and time.monotonic() < deadline:
        server.handle_request()
    server.server_close()
    if 'context' not in state:
        raise RuntimeError('Provider context was not supplied within five minutes')
    return state.pop('context')


def allowed_download(url, order):
    p = urlparse(url)
    path = unquote(p.path)
    return (p.scheme == 'https' and p.hostname == 'ceres-tool.larc.nasa.gov'
            and not p.query and not p.fragment
            and f'CERES_2026-10-01:{order["order_id"]}/' in path
            and path.lower().endswith('.nc'))


def get_status(session, email, order):
    params = {'command': 'oneOrderStatus', 'email': email, 'orderId': order['order_id'], 'CERESProducts': 'SYN1degEd42'}
    r = session.get(HOST + '/ord-tool/order', params=params, timeout=(15, 60))
    r.raise_for_status()
    r.encoding = 'utf-8'
    doc = BeautifulSoup(r.text, 'html.parser')
    rows = [[c.get_text(' ', strip=True) for c in tr.find_all(['td', 'th'])]
            for tr in doc.find_all('tr')]
    row = next((x for x in rows if len(x) >= 12 and 'CERES_SYN1deg_Ed4.2' in x[1]), None)
    if row is None:
        write_json(DEST / 'provider_structure_diagnostic.json', {'checked_utc': now(),
                   'http_status': r.status_code, 'page_title': doc.title.get_text() if doc.title else None,
                   'final_path': urlparse(r.url).path, 'redirect_codes': [x.status_code for x in r.history],
                   'query_keys': sorted(__import__('urllib.parse', fromlist=['parse_qs']).parse_qs(urlparse(r.url).query)),
                   'row_lengths': [len(x) for x in rows],
                   'product_cells': [x[1] for x in rows if len(x) >= 12],
                   'expected_product_seen': 'CERES_SYN1deg_Ed4.2' in r.text,
                   'requested_order_id': order['order_id']})
        raise ValueError('Expected order status row is absent')
    public = dict(order, checked_utc=now(), provider_status=row[5], expected_files=int(row[6]),
                  completed_files=int(row[7]), processing_files=int(row[8]), failed_files=int(row[9]),
                  reported_bytes_label=row[10], estimated_size=row[11])
    if public['failed_files']:
        raise RuntimeError('Provider reported failed order files')
    params = {'command': 'oneSuborderStatus', 'email': email, 'SuborderId': order['suborder_id'], 'CERESProducts': 'SYN1degEd42'}
    r = session.get(HOST + '/ord-tool/order', params=params, timeout=(15, 60))
    r.raise_for_status()
    r.encoding = 'utf-8'
    doc = BeautifulSoup(r.text, 'html.parser')
    links = sorted({urljoin(r.url, a['href']) for a in doc.select('a[href]')
                    if allowed_download(urljoin(r.url, a['href']), order)})
    return public, links


def inspect_file(path, order):
    expected = pd.date_range('2019-02-01 00:30', '2019-12-31 23:30', freq='h')
    with netCDF4.Dataset(path) as ds:
        assert '4.2' in str(getattr(ds, 'Version', ''))
        assert np.array_equal(ds['lon'][:], np.arange(310.5, 346.0, 1.0))
        south = 28.5 if order['region'] == 'north_atlantic' else -61.5
        assert np.array_equal(ds['lat'][:], np.arange(south, south + 34, 1.0))
        t = ds['time']
        dates = pd.DatetimeIndex([str(x) for x in netCDF4.num2date(np.asarray(t[:], dtype=float),
                                  t.units, calendar=getattr(t, 'calendar', 'standard'))])
        nominal = dates.floor('h') + pd.Timedelta(minutes=30)
        assert nominal.equals(expected)
        max_offset = float(np.max(np.abs((dates - nominal).total_seconds())))
        assert max_offset <= 15
        var = ds['cldarea_total_1h']
        assert var.shape == (8016, 34, 36) and var.dimensions == ('time', 'lat', 'lon')
        assert str(var.units).lower() == 'percent'
        valid = missing = 0
        minimum, maximum = float('inf'), float('-inf')
        for i in range(0, 8016, 168):
            chunk = np.ma.asarray(var[i:i + 168])
            values = chunk.compressed()
            assert np.isfinite(values).all() and ((values >= 0) & (values <= 100)).all()
            valid += int(values.size); missing += int(chunk.size - values.size)
            if values.size:
                minimum = min(minimum, float(values.min())); maximum = max(maximum, float(values.max()))
        assert valid > 0
    return dict(shape=[8016, 34, 36], valid_values=valid, missing_values=missing,
                cloud_percent_min=minimum, cloud_percent_max=maximum,
                first_nominal_slot=expected[0].isoformat(), last_nominal_slot=expected[-1].isoformat(),
                max_abs_time_rounding_seconds=max_offset, checks='PASS_FULL_VALUE_RANGE_TIME_GRID')


def download(session, url, order, retained_bytes):
    assert allowed_download(url, order)
    folder = DEST / order['region']; folder.mkdir(exist_ok=True)
    name = Path(unquote(urlparse(url).path)).name
    if not name or name in {'.', '..'}:
        raise ValueError('Invalid provider filename')
    final = folder / name
    if final.exists():
        raise FileExistsError('Refusing to overwrite an existing science file')
    part = final.with_suffix('.nc.part')
    if part.exists():
        raise FileExistsError('An interrupted partial file requires review before retry')
    with session.get(url, stream=True, timeout=(15, 120), allow_redirects=False) as r:
        r.raise_for_status()
        if r.status_code != 200:
            raise ValueError('Unexpected download HTTP response')
        announced = int(r.headers.get('Content-Length', 0))
        if announced and announced + retained_bytes > CAP:
            raise ValueError('Download exceeds frozen storage cap')
        h = hashlib.sha256(); n = 0
        with part.open('xb') as handle:
            for block in r.iter_content(1024 * 1024):
                n += len(block)
                if n + retained_bytes > CAP:
                    raise ValueError('Download exceeds frozen storage cap')
                handle.write(block); h.update(block)
    if announced and announced != n:
        raise ValueError('Incomplete provider download')
    with part.open('rb') as handle:
        if handle.read(3) != b'CDF':
            raise ValueError('Expected NetCDF3 data; keeping partial for review')
    validation = inspect_file(part, order)
    part.rename(final)
    return dict(order, path=str(final), url=url, bytes=n, sha256=h.hexdigest(),
                downloaded_utc=now(), validation=validation)


def resume_completed():
    """Resume verified prior completions; never silently ignore untracked/partial files."""
    science = {p.resolve() for p in DEST.rglob('*.nc')}
    partials = list(DEST.rglob('*.nc.part'))
    total = sum(p.stat().st_size for p in science | {p.resolve() for p in partials})
    if total > CAP or partials:
        raise ValueError('Existing bytes or interrupted partial data require review')
    manifest = DEST / 'download_manifest.json'
    records = json.loads(manifest.read_text(encoding='utf8'))['files'] if manifest.exists() else []
    done = {}; known = set()
    for item in records:
        order = next((o for o in ORDERS if o['region'] == item['region']), None)
        if order is None or any(item[k] != order[k] for k in ('order_id', 'suborder_id')):
            raise ValueError('Resume manifest contains an unexpected order')
        if item['region'] in done or not allowed_download(item['url'], order):
            raise ValueError('Resume manifest contains duplicate or unexpected sources')
        path = Path(item['path']).resolve()
        if path.parent != (DEST / item['region']).resolve() or path not in science:
            raise ValueError('Resume source path is absent or outside its fixed region')
        if path.stat().st_size != item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Resume source hash/size mismatch')
        inspect_file(path, order)
        done[item['region']] = item; known.add(path)
    if science != known:
        raise ValueError('Untracked science files require review')
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--context-port', type=int, default=8766)
    ap.add_argument('--poll-seconds', type=int, default=300)
    ap.add_argument('--max-hours', type=float, default=24)
    ap.add_argument('--run-analysis', action='store_true')
    args = ap.parse_args()
    if args.poll_seconds < 60 or not 0 < args.max_hours <= 24:
        raise ValueError('Polling bounds must be respected')
    DEST.mkdir(parents=True, exist_ok=True)
    status_path = DEST / 'download_status.json'
    status = dict(started_utc=now(), state='awaiting_ephemeral_provider_context', orders=ORDERS,
                  max_new_bytes=CAP, poll_seconds=args.poll_seconds, max_hours=args.max_hours,
                  email_persisted=False, pid=__import__('os').getpid(), downloaded_files=[])
    write_json(status_path, status)
    try:
        done = resume_completed()
        context = receive_context(args.context_port)
        email = context.pop('email')
        session = requests.Session()
        session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36'
        for cookie in context.pop('cookies'):
            session.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'], path=cookie.get('path', '/'))
        deadline = time.monotonic() + args.max_hours * 3600
        transient_errors = 0
        while time.monotonic() < deadline:
            states = []
            for order in ORDERS:
                if order['region'] in done:
                    continue
                try:
                    public, links = get_status(session, email, order)
                    states.append(public)
                except requests.RequestException as exc:
                    transient_errors += 1
                    states.append(dict(order, checked_utc=now(), network_error_type=type(exc).__name__))
                    continue
                if links:
                    if len(links) != 1 or public['expected_files'] != 1:
                        raise ValueError('Expected exactly one provider science file per order')
                    item = download(session, links[0], order, sum(x['bytes'] for x in done.values()))
                    done[order['region']] = item
                    write_json(DEST / 'download_manifest.json', dict(updated_utc=now(), files=list(done.values()),
                               total_bytes=sum(x['bytes'] for x in done.values()), complete=len(done) == 2,
                               validation_scope='Downloader full range/grid/time checks; independent comparison audit pending'))
            status.update(updated_utc=now(), state='downloaded' if len(done) == 2 else 'waiting_for_provider',
                          provider_status=states, downloaded_files=list(done.values()), transient_errors=transient_errors)
            write_json(status_path, status)
            if len(done) == 2:
                break
            time.sleep(args.poll_seconds)
        if len(done) != 2:
            status.update(state='time_limit_reached', updated_utc=now())
            write_json(status_path, status); return 2
        if args.run_analysis:
            status.update(state='analysis_running', updated_utc=now()); write_json(status_path, status)
            script = Path(__file__).with_name('run_ceres_annual_comparison.py')
            with (DEST / 'annual_analysis.stdout.log').open('w', encoding='utf8') as out, (DEST / 'annual_analysis.stderr.log').open('w', encoding='utf8') as err:
                result = subprocess.run([sys.executable, str(script), '--run'], stdout=out, stderr=err)
            status.update(state='analysis_completed_pending_independent_audit' if result.returncode == 0 else 'analysis_failed',
                          analysis_exit_code=result.returncode, updated_utc=now())
            write_json(status_path, status)
            if result.returncode != 0:
                return result.returncode
        return 0
    except Exception as exc:
        # Never log raw HTTP exceptions: they can contain the provider email URL.
        status.update(state='stopped_requires_review', error_type=type(exc).__name__, updated_utc=now())
        write_json(status_path, status)
        print('CERES worker stopped; error type: ' + type(exc).__name__, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
