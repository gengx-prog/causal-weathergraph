# Reproduce and test the published revision

Start here after cloning the repository. Run commands from its root. The
[environment guide](ENVIRONMENT.md) gives the observed author hardware, exact
Python/dependency versions, and separate Windows and Linux installation steps.
Use CPython 3.13.7 and `python -m pip install -r requirements-test.txt` in a new
virtual environment. In PowerShell, replace `python` below with
`& .\.venv\Scripts\python.exe` if that environment is not activated.

## 1. Check the checkout without meteorological downloads

```bash
python scripts/verify_artifacts.py
python -m pip check
python -m pytest tests -q -rs
python -m revision.run_simulations --output outputs/my-smoke --stage all --null-repetitions 1 --graph-repetitions 1 --n-null 256 --n-graph 1200
```

The small simulation must report two completed null repetitions and five graph
repetitions in `outputs/my-smoke/run_status.json`. These are one repetition for
each scenario, not the full published benchmark. The 13 cloud simulator tests
that need separately trained production models are expected to skip. Unit tests,
file hashes and a synthetic smoke run do not establish reproduction of the paper.
Use a new output directory on each attempt.

On GitHub, every push/pull request runs the CPU checks on Windows and Ubuntu.
For real-data verification, open **Actions → Paper reproduction from archived
inputs → Run workflow**, then select `core` or `all`. This workflow downloads
the published release and uploads the comparison report, logs, result CSVs and
runner environment as an Actions artifact.

## 2. Recompute the core graph results from real archived inputs

```bash
python scripts/download_artifacts.py --asset core-inputs.zip
python scripts/verify_artifacts.py --data-root artifacts/data --asset core-inputs.zip
python scripts/reproduce.py --experiment core --data-root artifacts/data --output outputs/my-core-reproduction
```

The versioned [release](https://github.com/gengx-prog/causal-weathergraph/releases/tag/reproducibility-20261008.1)
contains the actual 68,668 × 66 × 4 discovery-fitted regional series, the
candidate graph, the screening reference, and the original result tables.
The downloader verifies both the ZIP and every extracted file using
[`release-manifest.json`](../artifacts/release-manifest.json). Existing files
with different contents are rejected. Downloads and extracted arrays are ignored
by Git. To work offline, transfer the ZIP and use
`--from-directory /path/to/downloaded/zips` with the same downloader.

`reproduce.py` runs screen–confirm–replicate and symmetric full-VAR analyses,
then compares every core reference CSV, including all 2,574 candidate-lag tests
and the symmetric family. Row order, labels, Boolean decisions and numeric
values are checked; the numeric tolerance is `rtol=1e-7, atol=1e-10`.
`verification.json` records each table, maximum absolute numeric difference,
runner hash, exit status and duration. The original core run recorded about
127 seconds and a 2.35 GB peak working set; these are observations on the author
machine, not guaranteed runtimes or resource requirements on another machine.

## 3. Recompute the satellite comparison and WHEC tests

```bash
python scripts/download_artifacts.py --asset all
python scripts/verify_artifacts.py --data-root artifacts/data
python scripts/reproduce.py --experiment all --data-root artifacts/data --output outputs/my-full-reproduction
```

This adds both CERES climatology settings and the frozen WHEC tests. The runner
copies the original WHEC `design.json` into the new output directory; its
published SHA-256 must still match. All paths are explicit and use downloaded
inputs. Choose `--experiment ceres` or `--experiment whec` to run either alone.
Allow several GB of download/extraction space and more memory than the core run.
Download sizes and complete file inventories are in the release manifest.

The supplementary release includes a deterministic 2017–2025 ERA5 cloud slice
in percent, with hashes of its parent NetCDF files. It preserves the original
loader's values and time order. CERES, regional vectors, WHEC series and fitted
preprocessing parameters are archived inputs; this workflow does **not** rerun
raw-data acquisition, conservative remapping, preprocessing, or WHEC index
construction. The retained original study inputs and provider subsets are now
available in the separate [raw-data archive](RAW_DATA.md), with [official source
links](RAW_DATA_SOURCES.md). See [artifact coverage](ARTIFACTS.md) for the
processed-input scope.

## Review the paper and original comments

| Material | Location |
| --- | --- |
| Exact original submission, including its original author/title pages | [Original PDF](../manuscript/original_submission/INS-D-26-9116_original_submission.pdf) |
| Original editor/reviewer comments | [Review PDF](../manuscript/reviews_20261008/Editor_and_Reviewers.pdf), [Reviewer 1 attachment](../manuscript/reviews_20261008/Reviewer_1_Report.docx) |
| Current point-by-point response | [Response PDF](../manuscript/reviews_20261008/Response_to_Editor_and_Reviewers.pdf) |
| Comment identifiers used by the response | [Comment JSON](../manuscript/reviews_20261008/canonical_comments.json) |
| Revised manuscript, marked changes, supplement and LaTeX | [Submission package](../manuscript/submission_20261002/) |
| Mapping from review topics to implementation/evidence | [Review evidence map](REVIEW_EVIDENCE.md) |

The original submission is a historical record; its contents and byline have
not been rewritten to match the revised manuscript. The review response is also
preserved as delivered. Passing software tests does not resolve a scientific
objection, establish intervention effects, or turn previously inspected years
into a pristine holdout.

## Historical scripts and additional experiments

[`revision/REPRODUCE.md`](../revision/REPRODUCE.md) preserves the chronology of
the research workspace. Paths such as `D:\Paper2\...`, `/home/vipuser/...`, and
sibling `revision_outputs/` there describe past executions. They are not the
installation instructions for a new clone. The commands above are the portable
entry points verified for this publication. Separate weather-token, intervention
and cloud-interface prototypes have their own inputs and are not part of these
paper-result replication checks.
