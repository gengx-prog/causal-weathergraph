"""Create an isolated CPU environment, run the reviewer demo, and retain evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', action='store_true', help='Also rerun ten archived core reference tables (larger input download).')
    parser.add_argument('--data-root', type=Path, help='Existing extracted core inputs; their hashes are checked by the core runner.')
    parser.add_argument('--output', type=Path, help='New evidence directory; default is a unique directory under repository outputs/.')
    parser.add_argument('--venv', type=Path, help='Explicit dedicated environment to create or reuse; default creates a new isolated environment for each run.')
    args = parser.parse_args(argv)
    if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 13):
        parser.error('This pinned quick-start uses CPython 3.13. Run with py -3.13 (Windows) or python3.13 (Linux).')
    if args.data_root and not args.core:
        parser.error('--data-root requires --core')
    if args.data_root and not args.data_root.is_dir():
        parser.error('--data-root must be a directory containing extracted core inputs; see docs/REVIEWER_QUICKSTART.md')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'{stamp}-{uuid.uuid4().hex[:8]}'
    output = (args.output or ROOT / 'outputs' / f'reviewer-demo-{run_id}').resolve()
    env_dir = (args.venv or ROOT / f'.venv-reviewer-{run_id}').resolve()
    if output.exists():
        parser.error('Output already exists; choose a new directory to preserve the previous evidence.')
    if output == env_dir or output.is_relative_to(env_dir) or env_dir.is_relative_to(output):
        parser.error('Keep the environment and evidence directories separate.')
    if env_dir.exists() and not (env_dir / 'pyvenv.cfg').is_file():
        parser.error('The environment path already exists but is not a virtual environment; choose a new --venv directory.')
    if env_dir.exists():
        config = (env_dir / 'pyvenv.cfg').read_text(encoding='utf-8')
        if 'include-system-site-packages = false' not in config:
            parser.error('Use an isolated environment without system-site-packages.')
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'profile': 'demo_and_core' if args.core else 'demo',
              'scope': 'Synthetic statistical execution check; optional core comparisons start from archived processed inputs.',
              'environment_created': not env_dir.exists(),
              'environment_directory': env_dir.relative_to(ROOT).as_posix() if env_dir.is_relative_to(ROOT) else 'external_user_selected_environment',
              'steps': [],
              'source_sha256': {}, 'outputs': {},
              'limitations': ['This is not all raw-data preprocessing, all supplementary experiments, or optional neural training.',
                              'Runtime depends on installation, network, and machine; installation and scientific runtime are reported separately.']}
    for name in ('scripts/reviewer_quickstart.py', 'scripts/run_demo.py', 'requirements-demo.txt',
                 'requirements-test.txt', 'environment/constraints-core-py313.txt',
                 'revision/simulations.py', 'revision/inference.py', 'scripts/reproduce.py'):
        path = ROOT / name
        if path.is_file():
            report['source_sha256'][name] = digest(path)
    env = os.environ.copy()
    env.update({'PYTHONUTF8': '1', 'PYTHONNOUSERSITE': '1', 'MPLBACKEND': 'Agg',
                'PIP_DISABLE_PIP_VERSION_CHECK': '1'})
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '2'
    # An inherited PYTHONPATH could otherwise inject packages outside this venv.
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    for name in ('PIP_TARGET', 'PIP_PREFIX', 'PIP_USER', 'PIP_ROOT'):
        env.pop(name, None)
    env['PIP_CONFIG_FILE'] = os.devnull
    env['PIP_REQUIRE_VIRTUALENV'] = 'true'

    def run(label, command, filename, append=False):
        print(f'[{label}] {filename}', flush=True)
        step_started = time.perf_counter()
        with (output / filename).open('a' if append else 'w', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        step = {'name': label, 'exit_code': completed.returncode,
                'elapsed_seconds': round(time.perf_counter() - step_started, 3), 'log': filename}
        report['steps'].append(step)
        write_json(output / 'verification.json', report)
        if completed.returncode:
            raise RuntimeError(f'{label} did not complete (exit {completed.returncode}); please share {filename} and environment.json if present.')

    try:
        write_json(output / 'verification.json', report)
        if not env_dir.exists():
            run('create_environment', [sys.executable, '-m', 'venv', str(env_dir)], 'environment-create.log')
        python = env_dir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if not python.is_file():
            raise RuntimeError('The environment has no Python interpreter. Choose a new --venv path; previous files have been preserved.')
        run('check_interpreter', [str(python), '-I', '-c',
            'import sys; assert sys.implementation.name == "cpython" and sys.version_info[:2] == (3,13); assert sys.prefix != sys.base_prefix; print(sys.version)'], 'interpreter.log')
        requirements = 'requirements-test.txt' if args.core else 'requirements-demo.txt'
        run('install_dependencies', [str(python), '-m', 'pip', 'install', '-r', str(ROOT / requirements)], 'install.log')
        run('dependency_consistency', [str(python), '-m', 'pip', 'check'], 'pip-check.log')
        run('record_environment', [str(python), str(ROOT / 'scripts/collect_environment.py'), '--output', str(output / 'environment.json')], 'environment-collect.log')
        inventory = json.loads((output / 'environment.json').read_text(encoding='utf-8'))
        inventory['verification_os'] = {'system': platform.system(), 'release': platform.release(), 'architecture': platform.machine()}
        inventory['blas_thread_limits_by_stage'] = {'synthetic_demo': 1, 'core_analysis': 2 if args.core else None}
        inventory['cuda_required'] = False
        inventory['environment_created_by_this_run'] = report['environment_created']
        write_json(output / 'environment.json', inventory)
        run('record_resolved_dependencies', [str(python), '-m', 'pip', 'freeze'], 'pip-freeze.txt')
        run('synthetic_demo', [str(python), str(ROOT / 'scripts/run_demo.py'), '--output', str(output / 'demo')], 'run.log')
        demo = json.loads((output / 'demo/summary.json').read_text(encoding='utf-8'))
        if demo.get('status') != 'passed':
            raise RuntimeError('Demo finished without a passing summary. Please inspect demo/summary.json and run.log.')
        report['outputs']['demo'] = 'demo/summary.json'
        if args.core:
            if args.data_root:
                data_root = args.data_root.resolve()
                report['core_input_source'] = 'existing_extracted_inputs_verified_by_sha256'
            else:
                run('download_core_inputs', [str(python), str(ROOT / 'scripts/download_artifacts.py'), '--asset', 'core-inputs.zip'], 'download.log')
                data_root = ROOT / 'artifacts/data'
                report['core_input_source'] = 'public_release_download_verified_by_sha256'
            run('core_reference_comparison', [str(python), str(ROOT / 'scripts/reproduce.py'), '--experiment', 'core',
                '--data-root', str(data_root), '--output', str(output / 'core')], 'core-run.log')
            core = json.loads((output / 'core/verification.json').read_text(encoding='utf-8'))
            comparisons = [item for job in core['jobs'] for item in job['comparisons']]
            inputs_manifest = json.loads((ROOT / 'artifacts/release-manifest.json').read_text(encoding='utf-8'))
            core_asset = next(asset for asset in inputs_manifest['assets'] if asset['name'] == 'core-inputs.zip')
            expected_tables = {Path(item['path']).name for item in core_asset['files']
                               if item['path'].startswith('three_stage/') and item['path'].endswith('.csv')}
            observed_tables = [item['file'] for item in comparisons]
            if (core.get('status') != 'passed' or len(expected_tables) != 10
                    or len(observed_tables) != 10 or set(observed_tables) != expected_tables
                    or any(item['status'] != 'passed' for item in comparisons)):
                raise RuntimeError('The core run did not pass all ten reference comparisons. Inspect core/verification.json.')
            report['outputs']['core'] = 'core/verification.json'
            report['core_reference_tables_passed'] = 10
        report['status'] = 'passed'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        report['error'] = 'Run interrupted; partial logs are preserved. Choose a new output directory to retry.'
        print(report['error'], file=sys.stderr, flush=True)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        print(str(exc), file=sys.stderr, flush=True)
    finally:
        report['elapsed_seconds_total'] = round(time.perf_counter() - started, 3)
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['evidence_sha256'] = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*'))
                                      if p.is_file() and p != output / 'verification.json'}
        write_json(output / 'verification.json', report)
    print(f"{report['status'].upper()}: {output}", flush=True)
    print('Please keep verification.json, environment.json and the logs with any issue report.', flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
