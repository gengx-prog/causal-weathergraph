# Executed publication checks — October 8, 2026

The final local checks used a fresh CPython 3.13.7 virtual environment without
system-package inheritance. Hardware and package versions are in
[the environment record](../docs/ENVIRONMENT.md). Earlier clean-install receipts
are preserved with the test counts observed at that time; the final test suite
below also includes the new input and artifact-validation checks.

| Check | Actual result | Evidence |
| --- | --- | --- |
| Independent dependency installation and `pip check` | Passed | [Installation receipt](../environment/validation-clean-20261008.json) |
| Processed-input publication CPU unit/numerical/portability tests | 138 passed, 13 skipped, 0 failed | [JUnit](tests-20261008.xml) |
| Small known-structure simulation | 2 null scenario repetitions, 5 graph scenario repetitions; 12 and 30 summary rows | [Clean environment receipt](../environment/validation-clean-20261008.json) |
| Archived publication integrity | 58 repository files, 2 ZIP CRC checks, 459 released file hashes passed | [Integrity report](artifact-integrity.json) |
| Full core graph rerun from released inputs | 10 tables passed; about 134 seconds | [Per-table comparison](paper-reproduction-20261008.json) |
| CERES monthly and month × UTC-hour reruns | 4 tables per setting passed; about 46 seconds each | [Per-table comparison](paper-reproduction-20261008.json) |
| WHEC rerun | 6 tables passed; about 54 seconds; original design, periods and decisions match | [Per-table and decision comparison](paper-reproduction-20261008.json) |
| GitHub-uploaded experimental ZIPs | All three server-side SHA-256 digests and sizes match the local manifest | [Upload verification](release-upload.json) |

The 24 scientific tables match within `rtol=1e-7, atol=1e-10`; the largest
observed absolute difference is about `2.49e-11`. Integer counts, Boolean
decisions, row/column identities and the WHEC verdict are checked exactly.
The WHEC metadata check was additionally performed after the local rerun;
the published runner performs it automatically in subsequent reruns.

The 13 skipped checks require production artifacts of the separate cloud
interface prototype. They are not counted as passing tests. The final JUnit
file omits the local hostname; other test outcomes are unchanged. Published
comparison options replace the local input-directory prefix with `{data_root}`.

These checks reran analyses from archived **processed** arrays. They did not
rerun the original global downloads, preprocessing/remapping, WHEC construction
from raw fields, all historical benchmark repetitions, or separate prototypes.
The CPU smoke simulation is not the full calibration benchmark.

Live independent runner results are in
[GitHub Actions](https://github.com/gengx-prog/causal-weathergraph/actions).
The automatic workflow checks Windows and Ubuntu; the manual paper workflow
can reproduce `core` or `all` from the public release. Local execution and
GitHub runner results are separate evidence.

## Historical raw-data archive

The later [raw-data publication](../docs/RAW_DATA.md) adds 1,100 scientific
NetCDF files and eight source/metadata files in 79 ZIP assets. Every member
was read back from its ZIP and checked against its source SHA-256. Available
historical acquisition hashes matched for 1,023 scientific files; the other
77 scientific files have newly recorded publication hashes. All 1,100 NetCDF
headers were readable. These checks preserve the files as acquired; they do
not constitute a new scientific analysis of every field value.

The raw-data [local validation record](raw-data-local.json) reports the exact
inventory, manifest hash, packaging checks, unit-test counts and local restore
test. The local restore test covers 85 files from the original CDS extension,
CERES January 2019 pilot and source metadata. Its JUnit results are in
[raw-data-tests-20261008.xml](raw-data-tests-20261008.xml).

The manual [public raw-data workflow](https://github.com/gengx-prog/causal-weathergraph/actions/workflows/raw-data-check.yml)
downloads those same three dataset groups on Windows and Ubuntu using only
the Python standard library, then verifies all 85 file hashes. It downloads
approximately 222.5 MB per runner. The release includes a separate receipt
for the server-side size and SHA-256 verification of all 79 ZIP assets and
links to the executed runner checks.
