# Recorded numerical validation

These receipts preserve results from executions in October 2026. They are
historical numerical evidence, not a claim that the current checkout has already
rerun every experiment. Current checks should report their own version, inputs,
configuration and outcomes. The [software guide](../docs/SOFTWARE.md) describes the recorded Python and
dependency versions and the installation profiles.

## Current experiment-only checks

- CPU unit/numerical tests: **190 passed, 13 skipped** ([JUnit](tests-20261010.xml)). The skips require the
  optional trained cloud-model bundle and are not counted as passing tests.
- [Experimental figure verification](figure-verification-20261010.json):
  renderer execution and numerical/vector checks against the retained plot data.
- [Processed-input reproduction](experiment-reproduction-20261010.json):
  current job-level and table-level comparison outcomes.
- [Corrected WHEC verification](whec-correction-experiments-20261010.json):
  corrected-reference table and endpoint-series comparisons.

Each numerical receipt gives its own inputs and execution scope. Figure checks
do not rerun the underlying experiments; unit checks do not substitute for the
processed-input or corrected-WHEC comparisons.

## Historical numerical receipts

| Check | Recorded result | Evidence |
| --- | --- | --- |
| Independent installation and `pip check` | Passed in an isolated CPython 3.13.7 environment | [Installation receipt](../environment/validation-clean-20261008.json) |
| CPU unit/numerical/portability tests | 138 passed, 13 skipped, zero failures | [JUnit](tests-20261008.xml) |
| Small known-structure smoke run | Two null and five graph scenario repetitions; 12 and 30 summary rows | [Environment receipt](../environment/validation-clean-20261008.json) |
| Current repository integrity | The current tracked-file SHA-256 checks and their scope are recorded in the receipt | [Integrity receipt](artifact-integrity.json) |
| Core graph rerun | 10 reference tables matched; about 134 seconds | [Local comparisons](experiment-reproduction-20261008.json) |
| CERES monthly and month-by-UTC-hour reruns | Four tables per setting matched; about 46 seconds each | [Local comparisons](experiment-reproduction-20261008.json) |
| Original WHEC rerun | Six tables matched; about 54 seconds; design, periods and decisions matched | [Local comparisons](experiment-reproduction-20261008.json) |
| Remote experimental ZIP integrity | 85 server-side archive digests and sizes matched the current manifests | [Upload receipt](release-upload.json) |
| Ubuntu core/CERES/WHEC rerun | All 24 tables plus WHEC metadata checks passed; about 342 seconds excluding setup/download | [Independent comparisons](github-experiments-all-20261008.json) |

The 24-table local comparison used `rtol=1e-7, atol=1e-10`, with a largest
absolute difference of about `2.49e-11`. The independent Ubuntu comparison's
largest difference was about `3.41e-11`. Integer counts, Boolean decisions and
row/column identities were checked exactly. These timings are observations,
not promises for another machine.

The 13 skipped checks require a trained bundle for the separate cloud prototype.
They are not counted as passes. The smoke simulation does not execute the full
calibration repetition count. Processed-input reruns do not repeat the global
acquisition/remapping, fitted preprocessing, raw-grid WHEC construction or all
independent prototypes. See [execution scope](../docs/GITHUB_EXECUTION.md).

## Raw-data archive checks

The [raw archive](../docs/RAW_DATA.md) contains 1,100 scientific NetCDF files and
eight source/metadata files in 79 ZIPs. Every member was read back and checked
against its source SHA-256. Retained acquisition hashes matched for 1,023
scientific files; the other 77 received archive hashes. All 1,100 NetCDF headers
were readable. These are archive checks, not a new analysis of every field value.

The [local raw-data receipt](raw-data-local.json) records inventory, manifest hash,
packaging checks and the restore test. That test verified 85 files from the
legacy CDS extension, January 2019 CERES pilot and source metadata. Its test
results are in [the retained JUnit file](raw-data-tests-20261008.xml).
The configured raw-data workflow downloads the same selected groups and checks
their hashes; a new workflow run must be assessed on its actual result.
