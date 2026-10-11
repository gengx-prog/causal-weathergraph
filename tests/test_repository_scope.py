"""Protect the public experiment-only boundary, including archive members."""
import importlib.util
from pathlib import Path
import zipfile

spec = importlib.util.spec_from_file_location('scope', Path(__file__).resolve().parents[1] / 'scripts/check_repository_scope.py')
scope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scope)


def test_experimental_response_alignment_and_plots_remain_allowed():
    for name in ['tests/test_castle_response_alignment.py', 'results/effect_distributions.pdf', 'visualization/network.svg']:
        assert not scope.excluded_path(name)


def test_private_documents_and_nested_archives_are_rejected(tmp_path):
    for name in ['manuscript/main.pdf', 'Response_to_Reviewers.pdf', 'source/main.tex', 'references.bib', 'review.docx']:
        assert scope.excluded_path(name)
    archive = tmp_path / 'results.zip'
    with zipfile.ZipFile(archive, 'w') as writer:
        writer.writestr('tables/results.csv', 'value\n1\n')
        writer.writestr('private/main.tex', 'document source')
    assert scope.check_paths(tmp_path, ['results.zip']) == ['results.zip:private/main.tex']
