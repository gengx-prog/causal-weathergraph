"""Compile the clean, marked and supplementary manuscripts; fail on build errors."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
STEMS = (
    "Causal_WeatherGraph_revised",
    "Causal_WeatherGraph_marked",
    "Causal_WeatherGraph_supplementary",
)


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    result = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        print(result.stdout[-5000:])
        print(result.stderr[-1000:])
        raise SystemExit(f"Build failed ({result.returncode}): {' '.join(command)}")


def compile_tex(stem: str) -> None:
    output = HERE / ".build"
    output.mkdir(exist_ok=True)
    env = os.environ.copy()
    for name in ("BIBINPUTS", "BSTINPUTS"):
        env[name] = str(HERE) + os.pathsep + env.get(name, "")
    latex = [
        "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
        "-file-line-error", f"-output-directory={output}", f"{stem}.tex",
    ]
    run(latex, HERE, env)
    source = (HERE / f"{stem}.tex").read_text(encoding="utf-8")
    if re.search(r"\\bibliography\{", source):
        run(["bibtex", stem], output, env)
    run(latex, HERE, env)
    run(latex, HERE, env)
    log = (output / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
    issues = [line for line in log.splitlines() if (
        line.startswith("!")
        or re.search(r"(?:Citation|Reference).*(?:undefined|multiply defined)", line)
        or "There were undefined" in line
        or "Rerun to get cross-references right" in line
    )]
    if issues:
        raise SystemExit("Final reference check failed:\n" + "\n".join(issues))
    print(f"Built {output / (stem + '.pdf')}; no undefined references or citations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stem", nargs="?", choices=STEMS)
    args = parser.parse_args()
    for manuscript in (args.stem,) if args.stem else STEMS:
        compile_tex(manuscript)
