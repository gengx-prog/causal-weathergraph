# Quickstart release: 13 October 2026

Thank you for inspecting the code and its computational documentation. This
update makes the installation steps, inputs, expected outputs and execution
records easier to inspect and run. It adds a CPU demo using the existing
scientific routines, with a separate option for core atmospheric comparisons.

## Run the code

Install CPython 3.13, download or clone the repository, and open a terminal in
its root directory. Initial dependency installation requires internet access.

Windows:

```powershell
py -3.13 scripts/reviewer_quickstart.py
```

Linux:

```bash
python3.13 scripts/reviewer_quickstart.py
```

The launcher creates a fresh isolated environment, installs pinned direct
dependencies, runs a generated three-node example and saves the inputs, fitted
results, logs, installed versions and a verification receipt. No GPU or CUDA
is required. To download the approximately 145 MB of processed core inputs,
verify them and compare ten reference tables, add `--core` to the same command.
See the [complete quickstart](REVIEWER_QUICKSTART.md) for outputs and remedies.

## Evidence and provenance

The [release](https://github.com/gengx-prog/causal-weathergraph/releases/tag/quickstart-20261013)
includes `Reviewer_reproducibility_bundle_20261013.zip`. Its SHA-256 is:

```text
1670e906238d5b5477cbf8940fc91a5ab9da88b559f48b55aa8da3b13654b287
```

The ZIP is preserved byte-for-byte from the local validation package prepared
before publication. It contains the tested source in `repo/`, execution records
in `evidence/demo/` and `evidence/core/`, and the separate test log/XML in
`evidence/tests/`. Its local-only or not-yet-pushed statements describe the
preparation stage recorded there. This release publishes those changes; use
the release tag for the published source and these current instructions.
Scientific code hashes in the execution receipts identify what was actually
run. Publication documentation does not change the scientific implementation.

The recorded local checks used two fresh CPython 3.13.7 virtual environments
on the same Windows 11 machine. All 13 demo checks and all ten distinct core
CSV comparisons passed; the core download verified 16 input files. A separate
test suite reported 200 passed and 13 skipped because optional trained
cloud-model artifacts were absent. The core tolerance was `rtol=1e-7,
atol=1e-10`. See the [local receipt](../validation/quickstart-local-20261013.json).

The [quickstart workflow](https://github.com/gengx-prog/causal-weathergraph/actions/workflows/reviewer-demo.yml)
and [CPU test workflow](https://github.com/gengx-prog/causal-weathergraph/actions/workflows/reproducibility.yml)
report their Windows/Ubuntu results separately for each commit. The release
notes link the observed hosted runs; their status should be checked directly.

## Scope and reporting problems

The demo checks execution and numerical calculations on one synthetic
realization. The core comparison starts from archived processed inputs. These
checks do not repeat full raw-data preparation, all supplementary experiments,
optional neural training or provide a tested Docker/Colab deployment. Local
runtime observations depend on package caching, network speed and hardware.

If a documented step fails, please report the command, commit, verification
receipt and relevant error trace in the
[issue tracker](https://github.com/gengx-prog/causal-weathergraph/issues).
Remove credentials and private paths before sharing logs. This evidence helps
us investigate and correct the code, dependencies or instructions.
