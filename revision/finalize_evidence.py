"""Validate the aligned holdout and inventory completed, non-smoke evidence."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import shutil
import subprocess
import sys
import pandas as pd
from revision.prepare_inputs import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--require-rpcmci-complete', action='store_true')
    args = parser.parse_args()
    root = args.root
    repo = Path(__file__).resolve().parents[1]
    out = root / 'validation'
    out.mkdir(exist_ok=True)
    checks = []
    for name in ['discovery_edges.csv', 'evaluation_edges.csv', 'frozen_prediction_all_candidates.csv']:
        original = pd.read_csv(root / 'holdout' / name)
        aligned = pd.read_csv(root / 'holdout_aligned' / name)
        common = [c for c in original.columns if c in aligned.columns]
        pd.testing.assert_frame_equal(original[common], aligned[common], check_exact=True)
        assert len(aligned) == 2574
        checks.append(dict(check='aligned_holdout_preserves_all_original_non_pcmci_values', file=name, rows=len(aligned), passed=True))
    pcmci = pd.read_csv(root / 'holdout_aligned/heldout_pcmci_all_candidates.csv')
    assert len(pcmci) == 2574
    assert set(pcmci.pcmci_response_n) == {10228}
    checks.append(dict(check='pcmci_complete_evaluation_response_alignment', rows=len(pcmci), effective_response_n=10228, passed=True))
    sources = [('inputs', 'data_manifest.json', 'prepare_inputs.py'),
               ('holdout_aligned', 'manifest.json', 'run_holdout.py'),
               ('regime_contrasts', 'manifest.json', 'run_regime_contrasts.py')]
    for folder, manifest, runner in sources:
        record = json.loads((root / folder / manifest).read_text(encoding='utf8'))
        source = repo / 'revision' / runner
        assert sha256(source) == record['code_sha256']
        dest = root / folder / 'code_snapshot'
        dest.mkdir(exist_ok=True)
        shutil.copy2(source, dest / runner)
        checks.append(dict(check='post_run_archive_matches_recorded_execution_hash', runner=runner, sha256=sha256(source), passed=True))
    if args.require_rpcmci_complete:
        rpc = pd.read_csv(root / 'rpcmci/summary.csv')
        assert (rpc.completed_repetitions == 20).all()
        assert (rpc.planned_repetitions == 20).all()
        checks.append(dict(check='rpcmci_all_planned_repetitions', repetitions=20, passed=True))
    (out / 'root_final_checks.json').write_text(json.dumps(dict(
        checked_utc=datetime.now(timezone.utc).isoformat(), checks=checks,
        statistical_scope='Numerical/provenance checks do not establish inferential calibration or causal identification.',
        snapshots='Root runner copies archived after execution only when byte hashes match the execution manifest.'
    ), indent=2), encoding='utf8')
    environment = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (repo / 'revision/environment.freeze.txt').write_text(environment, encoding='utf8')
    records=[]
    extensions={'.csv','.json','.jsonl','.py','.pdf','.svg','.png','.txt'}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix not in extensions:
            continue
        relative=path.relative_to(root)
        if any('smoke' in part or part in {'pilot','logs'} for part in relative.parts):
            continue
        if path.name == 'artifact_index.csv':
            continue
        records.append(dict(path=relative.as_posix(), bytes=path.stat().st_size, sha256=sha256(path)))
    pd.DataFrame(records).to_csv(out / 'artifact_index.csv', index=False)
    print(f'Validated {len(checks)} root checks; indexed {len(records)} non-smoke artifacts.')


if __name__ == '__main__':
    main()
