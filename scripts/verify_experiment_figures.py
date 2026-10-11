"""Redraw the circular experiment figures in isolated copies and run their checks.

Default invocation from the repository root:
    python scripts/verify_experiment_figures.py --output outputs/experiment-figure-checks

The source snapshot is never modified. PDF hashes are recorded as output identity,
not compared across operating systems. This does not refit statistical models.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from sanitize_records import sanitize, sanitize_tree

ROOT = Path(__file__).resolve().parents[1]
SCOPE = (
    'Experiment figure source integrity, isolated redraw, and numerical/vector '
    'self-checks only. No statistical models or meteorological preprocessing are rerun. '
    'Automated checks do not repeat a human visual review or establish causal identification.'
)
JOBS = (
    ('circular_networks', 'render_networks.py', 'verify_networks.py', 'validation.json'),
    ('seasonal_state_contrasts', 'render_state_contrasts.py', 'verify_state_contrasts.py', 'outputs/verification.json'),
)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def snapshot(source):
    """Record all original figure files, including archived exports, without writing."""
    return {p.relative_to(source).as_posix(): digest(p)
            for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def save_report(output, report):
    (output / 'figure-verification.json').write_text(
        json.dumps(sanitize(report), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT,
                        help='Repository or source directory containing visualization/circular_networks and visualization/seasonal_state_contrasts')
    parser.add_argument('--output', type=Path, required=True,
                        help='New output directory outside the immutable experiment figure source')
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error('Output already exists; select a new directory to preserve prior receipts')
    if output.is_relative_to(source / 'visualization') or source.is_relative_to(output):
        parser.error('Output must be outside the source visualization directories')
    for name, render, check, _ in JOBS:
        for filename in (render, check, 'provenance.json'):
            if not (source / 'visualization' / name / filename).is_file():
                parser.error(f'Missing experiment figure source: visualization/{name}/{filename}')
    originals = {name: snapshot(source / 'visualization' / name) for name, *_ in JOBS}
    output.mkdir(parents=True)
    report = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'scope': SCOPE, 'python': ".".join(map(str, sys.version_info[:3])),
              'packages': {name: version(name) for name in ('numpy', 'pandas', 'matplotlib', 'PyMuPDF')},
              'source_root': '.', 'output_root': output.name,
              'source_inventory_sha256': originals,
              'comparison_policy': 'Source data and numerical/vector assertions; no cross-platform PDF byte comparison.',
              'human_review_note': 'Visual-review prose in the source self-checks describes the author review; this automated job does not repeat it.',
              'jobs': [], 'errors': []}
    save_report(output, report)
    env = os.environ.copy()
    env.update({'PYTHONUTF8': '1', 'PYTHONNOUSERSITE': '1', 'MPLBACKEND': 'Agg',
                'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                'MKL_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2'})
    for name, render, check, receipt_path in JOBS:
        destination = output / name
        record = {'figure': name, 'status': 'running', 'commands': []}
        report['jobs'].append(record)
        try:
            # Drop prior exports/receipts, ensuring a skipped redraw cannot pass on stale PDFs.
            shutil.copytree(source / 'visualization' / name, destination,
                            ignore=shutil.ignore_patterns('__pycache__', 'outputs', '*.pyc',
                                '*.pdf', '*.svg', '*.png', 'validation.json', 'verification.json',
                                'value_audit.json', 'plotted_values.csv'))
            for script in (render, check):
                command = [sys.executable, script]
                log_path = destination / (Path(script).stem + '.log')
                started = time.perf_counter()
                print(f'{name}: python {script} (isolated copy)', flush=True)
                with log_path.open('w', encoding='utf-8') as log:
                    process = subprocess.run(command, cwd=destination, env=env,
                                             stdout=log, stderr=subprocess.STDOUT, timeout=180)
                record['commands'].append({'command': ['python', script],
                                           'exit_code': process.returncode,
                                           'elapsed_seconds': round(time.perf_counter() - started, 3),
                                           'log': log_path.relative_to(output).as_posix()})
                if process.returncode:
                    raise RuntimeError(f'{script} exited {process.returncode}; inspect {log_path.name}')
            receipt = json.loads((destination / receipt_path).read_text(encoding='utf-8'))
            passed = receipt.get('status') == 'PASS' if name == 'circular_networks' else receipt.get('all_checks_pass') is True
            if not passed:
                raise RuntimeError('The figure self-check did not report passing status')
            record['self_check_receipt'] = (destination / receipt_path).relative_to(output).as_posix()
            record['rendered_outputs_sha256'] = {
                p.relative_to(output).as_posix(): digest(p)
                for p in sorted(destination.rglob('*')) if p.is_file() and p.suffix in ('.pdf', '.svg', '.png')}
            if not record['rendered_outputs_sha256']:
                raise RuntimeError('No newly rendered assets found')
            record['status'] = 'passed'
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            record['status'] = 'failed'
            record['error'] = str(exc)
            report['errors'].append(f'{name}: {exc}')
        save_report(output, report)
    report['source_files_unchanged'] = all(
        originals[name] == snapshot(source / 'visualization' / name) for name, *_ in JOBS)
    if not report['source_files_unchanged']:
        report['errors'].append('The original figure source snapshot changed during validation')
    report['status'] = 'passed' if not report['errors'] else 'failed'
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    save_report(output, report)
    print(json.dumps({'status': report['status'], 'scope': SCOPE,
                      'source_files_unchanged': report['source_files_unchanged'],
                      'report': 'figure-verification.json',
                      'errors': report['errors']}, indent=2))
    sanitize_tree(output)
    return int(bool(report['errors']))


if __name__ == '__main__':
    sys.exit(main())
