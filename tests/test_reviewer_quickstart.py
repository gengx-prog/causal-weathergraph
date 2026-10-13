"""Verify quick-start orchestration without installing or downloading packages."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import reviewer_quickstart as runner


def fake_environment(tmp_path):
    environment = tmp_path / 'environment'
    environment.mkdir()
    (environment / 'pyvenv.cfg').write_text('include-system-site-packages = false\n', encoding='utf-8')
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    python.parent.mkdir()
    python.touch()
    return environment


def stub_subprocess(monkeypatch, demo_status='passed', core_comparisons=None):
    def run(command, **kwargs):
        joined = ' '.join(map(str, command))
        if 'collect_environment.py' in joined:
            target = Path(command[command.index('--output') + 1])
            target.write_text(json.dumps({'packages': {}}), encoding='utf-8')
        elif 'run_demo.py' in joined:
            target = Path(command[command.index('--output') + 1])
            target.mkdir()
            (target / 'summary.json').write_text(json.dumps({'status': demo_status}), encoding='utf-8')
        elif 'reproduce.py' in joined:
            target = Path(command[command.index('--output') + 1])
            target.mkdir()
            (target / 'verification.json').write_text(json.dumps({
                'status': 'passed', 'jobs': [{'comparisons': core_comparisons}],
            }), encoding='utf-8')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, 'run', run)
    monkeypatch.setattr(runner.sys, 'version_info', (3, 13, 0))


def test_environment_and_evidence_overlap_is_rejected_before_execution(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('No subprocess may start for overlapping environment/evidence paths')

    monkeypatch.setattr(runner.subprocess, 'run', unexpected)
    monkeypatch.setattr(runner.sys, 'version_info', (3, 13, 0))
    with pytest.raises(SystemExit) as failure:
        runner.main(['--venv', str(tmp_path / 'environment'),
                     '--output', str(tmp_path / 'environment/evidence')])
    assert failure.value.code != 0
    assert not (tmp_path / 'environment').exists()


def test_zero_exit_demo_with_failed_receipt_cannot_make_quickstart_pass(tmp_path, monkeypatch):
    environment = fake_environment(tmp_path)
    stub_subprocess(monkeypatch, demo_status='failed')
    output = tmp_path / 'evidence'
    code = runner.main(['--venv', str(environment), '--output', str(output)])
    report = json.loads((output / 'verification.json').read_text(encoding='utf-8'))
    assert code != 0
    assert report['status'] == 'failed'
    assert 'passing summary' in report['error']
    assert (output / 'run.log').exists()


def test_ten_duplicate_reference_receipts_do_not_mean_ten_tables_passed(tmp_path, monkeypatch):
    environment = fake_environment(tmp_path)
    duplicate = [{'file': 'three_stage_summary.csv', 'status': 'passed'} for _ in range(10)]
    stub_subprocess(monkeypatch, core_comparisons=duplicate)
    data = tmp_path / 'inputs'
    data.mkdir()
    output = tmp_path / 'evidence'
    code = runner.main(['--core', '--data-root', str(data),
                        '--venv', str(environment), '--output', str(output)])
    report = json.loads((output / 'verification.json').read_text(encoding='utf-8'))
    assert code != 0
    assert report['status'] == 'failed'
    assert 'core_reference_tables_passed' not in report
    assert report['evidence_sha256']['core/verification.json'] == runner.digest(output / 'core/verification.json')
