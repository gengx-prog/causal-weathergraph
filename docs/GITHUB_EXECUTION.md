# Running on GitHub: verified scope, limitations and remedies

**可以通过 GitHub Actions 运行代码测试、ERA5/CERES 文件抽样校验和基于已处理输入的论文分析。整套原始数据预处理、独立原型模型训练不属于当前已验证的一键流程；具体原因和补救办法如下。**

The executions below use code commit
[`fe9a312`](https://github.com/gengx-prog/causal-weathergraph/commit/fe9a3125bd5ab05a1480f2fc6d14cf9693e1d677),
checked on October 8, 2026. GitHub Actions runs the programs on a runner;
opening a Python file on the repository website does not execute it or host
the optional interactive estimator.

## What has actually run

| Task | GitHub result | Evidence and scope |
| --- | --- | --- |
| CPU tests and small synthetic simulation | Passed on Windows and Ubuntu: 173 tests passed, 13 model-dependent checks skipped | [Executed checks](https://github.com/gengx-prog/causal-weathergraph/actions/runs/37768579494). The simulation is one repetition per scenario, not the full calibration benchmark. |
| Public ERA5/CERES download and extraction | Passed on Windows and Ubuntu: 85 files checked per runner, 222,536,291 download bytes each | [Executed download checks](https://github.com/gengx-prog/causal-weathergraph/actions/runs/37768594879). Covers the original CDS extension, CERES January 2019 pilot and source metadata; it does not download all 108.69 GB. |
| Core paper analysis from processed inputs | Passed on Ubuntu: 10 result tables | [Earlier core execution](https://github.com/gengx-prog/causal-weathergraph/actions/runs/37762048386), using the preceding corrected publication commit `5444ac7`. |
| Core + both CERES settings + WHEC from processed inputs (`all`) | Passed on Ubuntu: all 24 result tables and WHEC design/period/decision checks | [Full analysis execution](https://github.com/gengx-prog/causal-weathergraph/actions/runs/37772222786); [per-table comparison and runner environment](../validation/github-paper-all-20261008.json). The four analysis jobs took about 342 seconds, excluding setup/download. |
| Complete raw archive download and all raw preprocessing | Not supported by the configured standard-runner workflows; no full raw preprocessing rerun is claimed | All 79 uploaded ZIP hashes were checked, but this is distinct from running every raw processing stage. Use the remedies below. |
| Cloud estimator and other independent prototypes | Not covered by the paper-reproduction workflow | The cloud estimator's 13 model-dependent checks skip until trained artifacts exist; other prototype limitations are listed below. |

## Run a supported workflow

1. Open **Actions** in this repository or an up-to-date fork with Actions enabled.
2. Choose **Paper reproduction from archived inputs**, click **Run workflow**,
   select **main**, and choose `core` or `all`. Here `all` means core analysis,
   two CERES climatology settings and WHEC from archived processed inputs. It
   does not mean every historical experiment or all raw preprocessing.
3. For download checks instead, choose **Public raw-data download and checksum
   checks**. For unit tests and the small simulation, choose **Reproducibility checks**.
4. When the run finishes, open its **Artifacts** section. The paper artifact
   includes `paper-reproduction/verification.json`, comparison CSVs, logs and
   `paper-environment.json`. The raw-sample artifacts contain checksum reports
   and each runner's environment. The workflows retain these artifacts for 14 days;
   committed validation receipts and versioned Release files persist separately.

The paper report must say `status: passed`, with all planned jobs completed;
each CSV comparison must pass, along with the metadata comparison when WHEC
is selected. The paper and unit-test workflows install the pinned CPU dependencies;
the raw checksum workflow uses only the Python standard library. Published input
downloads do not require CDS/NASA credentials.

Manually starting a workflow requires repository write access. A reviewer
without that permission can inspect the existing runs, fork the repository
and run the workflows in the fork, or follow [the local commands](REPRODUCING.md).
This is a GitHub permission restriction, not an experiment failure.
[GitHub's manual-run instructions](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
describe the permission and workflow requirements.

## Limits and what to do about them

| Situation | What is happening | Remedy |
| --- | --- | --- |
| Trying to download/extract the entire raw release on a standard runner | The archive is 108.69 GB and retaining both ZIPs and extracted files needs more than 220 GB. GitHub documents 14 GB SSD for the standard Windows/Ubuntu runners used here. This storage mismatch is a resource limit, not an observed failure of the published sample workflow. | Use the processed-input `all` workflow for paper-result checks. For raw work, download selected dataset groups or use a local machine, larger runner or configured self-hosted runner with sufficient free storage plus processing workspace. |
| Expecting raw files to reproduce every upstream stage automatically | Download/hash verification does not execute acquisition, conservative remapping, fitted preprocessing or WHEC construction. Some historical scripts still have author-machine paths; a portable, verified end-to-end raw workflow has not been provided. | Follow [RAW_DATA.md](RAW_DATA.md) to recover the files. Supply explicit paths, preserve the recorded preprocessing choices and frozen design, then validate each generated input against its archived counterpart before comparing downstream tables. Extra disk space alone does not establish equivalence. |
| Seeing 13 skipped tests or failing to open the cloud estimator | `../revision_outputs/cloud_simulator_v1/manifest.json` and its trained model bundle are absent from a clean checkout. This is a separate CPU scikit-learn prototype, not a GPU requirement. | Use the supported paper workflow for the manuscript analyses. To test the optional estimator, restore its complete compatible trained bundle or use the separate retraining steps below; then rerun its integration tests. |
| Running an old command with `D:\Paper2`, `/home/vipuser` or a sibling experiment folder | The command belongs to the historical workstation record; the path is not present on a fresh runner. | Start with [REPRODUCING.md](REPRODUCING.md) and its explicit input/output options. Use a new output directory. Historical scripts outside these entry points need path adaptation and separate verification. |
| Interrupted download or checksum error | An incomplete ZIP may be resumable; a complete file with a wrong hash is not accepted as valid input. | Rerun the same download command for an interrupted transfer. For a reported corrupt temporary file, remove only the indicated temporary file and retry. Preserve existing scientific outputs; use a fresh cache/output directory if an existing final file conflicts. |

Runner storage specifications were checked on October 8, 2026:
[standard runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
[larger runners](https://docs.github.com/en/actions/reference/runners/larger-runners).
The author's hardware and observed package versions are documented in
[ENVIRONMENT.md](ENVIRONMENT.md); they are not claimed minimum requirements.

## Why older Actions runs are red

The [old CPU check](https://github.com/gengx-prog/causal-weathergraph/actions/runs/37761742382)
and [old core reproduction attempt](https://github.com/gengx-prog/causal-weathergraph/actions/runs/37761757678)
used commit `270d4e6`. Its repository manifest incorrectly required 18 temporary
LaTeX files under `manuscript/revision_v2_20261002/.build/`. Those files were
ignored and never committed, so a clean GitHub checkout could not contain them.
The 58 real repository files and two ZIP checks passed; the extra missing entries
stopped verification before the scientific analysis began. The core attempt had
already installed its dependencies and verified its 16 downloaded input files.

[Fix `5444ac7`](https://github.com/gengx-prog/causal-weathergraph/commit/5444ac74d83a61c26c29ddd4c9efcab8e216e81e)
removed the 18 erroneous entries and published the corrected tag
`reproducibility-20261008.1`. Subsequent CPU and core runs passed. Current remote
`main` and `raw-data-20261008` include that fix. The old tag
`reproducibility-20261008` still identifies the earlier faulty checkout.

Use an updated checkout and start a **new Run workflow on main**. Re-running an
old red job retains its original commit and will not pick up the fix, as explained
by [GitHub's rerun documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).
A cancelled run is not itself evidence of a software failure; inspect its
individual job steps. Some superseded runs were cancelled, including one that
had already encountered the old manifest error.

## Optional cloud-estimator recovery

The model bundle needs `manifest.json`, `models.joblib` and `metadata.json`;
the integration checks also need `split_indices.npz`, `predictions.npz` and
`protocol.json`. Its two training/source-fingerprint inputs are included in the
processed release's `extended-inputs.zip` (about 1.58 GB).

For a fresh, isolated checkout with an absent or empty sibling
`../revision_outputs/cloud_simulator_v1` directory, the existing scripts expose
the following recovery route. **This retraining route has not been executed by
the publication workflows; the 13 checks remain unverified until it succeeds.**

```bash
python -m pip install -r requirements-test.txt
python scripts/download_artifacts.py --asset extended-inputs.zip --output ../revision_outputs
python scripts/verify_artifacts.py --asset extended-inputs.zip --data-root ../revision_outputs
python revision/cloud_simulator/engine.py --output ../revision_outputs/cloud_simulator_v1 --horizons 0
python -m pytest tests/test_cloud_simulator.py -q -rs
python revision/cloud_simulator/app.py --artifacts ../revision_outputs/cloud_simulator_v1 --port 8765
```

Preserve existing trained artifacts instead of overwriting them. These tests
currently look at the fixed `cloud_simulator_v1` path and require the horizon-0
protocol; adding 6/24-hour horizons requires separate protocols and tests.
Changing the training directory does not update the tests' fixed lookup path.
New training is not guaranteed to reproduce the historical model bytes or metrics.
The app is a local service, not a site deployed by GitHub Actions.

The [weather-token project](../revision/weather_tokens/) additionally needs
PyTorch and its own training/checkpoint validation; it supports CPU or CUDA and
is not run by the CPU paper workflow. The [intervention benchmark](../revision/intervention_benchmark/)
has a few numerical unit checks, not its complete formal simulation in CI. The
[transport prototype](../revision/transport_consistency/) retains an author-local
PDF dependency in its design generator. Those projects need their own dependencies,
portable inputs/designs and complete validation before claiming reproduction;
generating a new design is not recovery of a historical frozen design.
