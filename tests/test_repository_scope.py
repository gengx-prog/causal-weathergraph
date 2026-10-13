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


def test_only_exact_public_reviewer_entry_points_are_allowed():
    public = ['.github/workflows/reviewer-demo.yml', 'scripts/reviewer_quickstart.py',
              'tests/test_reviewer_demo.py', 'tests/test_reviewer_quickstart.py',
              'docs/REVIEWER_QUICKSTART.md']
    for name in public:
        assert not scope.excluded_path(name)
        assert not scope.excluded_path(name.replace('/', '\\'))
    for name in ['docs/REVIEWER_QUICKSTART.docx', 'docs/REVIEWER_QUICKSTART.tex',
                 'scripts/reviewer_quickstart_notes.py', 'private/docs/REVIEWER_QUICKSTART.md',
                 'Response_to_Reviewers.pdf', 'reviewer_letter.md']:
        assert scope.excluded_path(name)


def test_public_quickstart_in_archive_does_not_allow_private_response(tmp_path):
    archive = tmp_path / 'tools.zip'
    with zipfile.ZipFile(archive, 'w') as writer:
        writer.writestr('scripts/reviewer_quickstart.py', '# public execution entry point')
        writer.writestr('docs/REVIEWER_QUICKSTART.md', 'Run the synthetic demo.')
        writer.writestr('Response_to_Reviewers.pdf', 'private letter')
        writer.writestr('docs/REVIEWER_QUICKSTART.tex', 'private source')
    assert scope.check_paths(tmp_path, ['tools.zip']) == [
        'tools.zip:Response_to_Reviewers.pdf', 'tools.zip:docs/REVIEWER_QUICKSTART.tex']


def test_private_documents_and_nested_archives_are_rejected(tmp_path):
    for name in ['manuscript/main.pdf', 'Response_to_Reviewers.pdf', 'source/main.tex', 'references.bib', 'review.docx']:
        assert scope.excluded_path(name)
    archive = tmp_path / 'results.zip'
    with zipfile.ZipFile(archive, 'w') as writer:
        writer.writestr('tables/results.csv', 'value\n1\n')
        writer.writestr('private/main.tex', 'document source')
    assert scope.check_paths(tmp_path, ['results.zip']) == ['results.zip:private/main.tex']
