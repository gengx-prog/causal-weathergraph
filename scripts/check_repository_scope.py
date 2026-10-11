"""Check that tracked files belong to the public experiment repository."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SUFFIXES = {'.tex', '.bib', '.doc', '.docx', '.csl'}
PRIVATE_NAMES = re.compile(
    r'(^|[/_ -])(manuscript|reviewers?|submission|highlights)([/_. -]|$)', re.I
)


def excluded_path(name: str) -> bool:
    path = PurePosixPath(name.replace('\\', '/'))
    return path.suffix.lower() in PRIVATE_SUFFIXES or bool(PRIVATE_NAMES.search(path.as_posix()))


def check_paths(root: Path, names: list[str]) -> list[str]:
    errors = []
    for name in names:
        if excluded_path(name):
            errors.append(name)
        path = root / name
        if path.suffix.lower() == '.zip' and path.is_file():
            with zipfile.ZipFile(path) as archive:
                errors.extend(f'{name}:{member}' for member in archive.namelist() if excluded_path(member))
    return errors


def main() -> int:
    raw = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT)
    names = [name for name in raw.decode('utf-8').split('\0') if name and (ROOT / name).is_file()]
    errors = check_paths(ROOT, names)
    if errors:
        print('Files outside the experiment scope:\n' + '\n'.join(errors))
        return 1
    print(f'Experiment repository scope passed: {len(names)} tracked files.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
