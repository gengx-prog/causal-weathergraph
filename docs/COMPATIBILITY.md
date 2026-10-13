# Compatibility and troubleshooting / 运行环境与问题排查

Start with the [reviewer quickstart](REVIEWER_QUICKSTART.md): CPython 3.13, a new
isolated `.venv-reviewer-*` environment and pinned CPU dependencies. The launcher
handles installation and preserves the logs needed to investigate a problem.
Failures in the documented workflow are actionable reports for maintainers,
including installation, unclear instructions and numerical mismatches.

## Supported route and verification scope

The project provides Windows and Linux commands. Existing automated workflows
cover Windows and Ubuntu CPU checks; the archived-input scientific workflow has
separate recorded runs. Consult each run's receipt for its commit, inputs,
environment and result. Other Python versions, macOS, containers and hosted
notebooks need separate validation before being described as tested.

| Workload | Entry point | Inputs and resources |
| --- | --- | --- |
| Small reviewer demo | `scripts/reviewer_quickstart.py` | CPU, CPython 3.13 and internet for package installation. Generates a small three-node input. |
| Core result comparisons | Same command with `--core` | Broader CPU dependencies and approximately 145 MB processed-input ZIP; checks 10 result tables. Existing extracted inputs can be supplied with `--data-root`. |
| CERES and historical WHEC | [Reproduction guide](REPRODUCING.md) | Additional processed inputs and working memory. The `all` profile covers the October 8 configuration. |
| Corrected WHEC control | [Correction guide](WHEC_AMENDMENT_20261009.md) | Separate October 9 inputs and reference comparisons. |
| Raw acquisition/preprocessing | [Raw-data guide](RAW_DATA.md) | About 108.69 GB of archives; ZIPs plus extracted files need more than 220 GB, plus workspace. A complete portable raw-to-results rerun has not been verified. |
| Optional weather-token model | [Weather-token module](../revision/weather_tokens/) | Separate PyTorch setup, data and checkpoints; independent of the statistical workflow. |

The demo and core workflow do not require a GPU or CUDA. The launcher records
actual step times. The demo uses one numerical thread and the core runner uses
two; other experiments retain their own documented settings.

## Locate the failing stage

| Symptom | What to do |
| --- | --- |
| `py -3.13` or `python3.13` is not found | Install CPython 3.13, reopen the terminal and run the command from the repository root. |
| Script or repository file is missing | Check that the whole repository was downloaded and the terminal is in the folder containing `README.md`. |
| Dependency installation fails | Read `install.log` and `interpreter.log`. Report the complete pip error and Python version; the pinned profile or setup instructions may need a project fix. |
| `pip check` fails | Keep `pip-check.log` and the selected environment details. The default launcher creates a fresh environment; explicit `--venv` opts into creating or reusing a named one. |
| A selected environment uses the wrong Python | Use the default fresh environment or a new `--venv` path. Keep existing environments and records while investigating. |
| Public input download is interrupted | Rerun the command. The downloader retains partial transfers, resumes where supported and verifies the result. Resolve a wrong checksum before analysis. |
| `--data-root` fails verification | Point it to extracted inputs containing `inputs/region_trainfit.npz`; compare the reported paths with the release manifest. A ZIP filename is not an extracted data root. |
| Numerical comparison fails | Keep `verification.json` and the relevant computation log. Report expected and observed results; do not relax tolerances merely to obtain a pass. |
| Process exits, is slow or reports insufficient resources | Include the failing command, workload and available memory/disk in the report. Diagnose the error before attributing it to implementation, input or resource use. |
| Optional model tests are skipped | The reason identifies an absent trained bundle. Follow that model's separate guide if needed. A skip does not verify that model. |
| Optional model reports PyTorch/CUDA errors | Follow that model's environment and checkpoint instructions. CUDA troubleshooting is unnecessary for the CPU demo or core analysis. |

Each default launch creates a new environment and result directory. The terminal
prints stage-specific log names. The core option uses `download.log` and
`core-run.log`, with detailed table checks in `core/verification.json`.
An early interpreter or command-line error appears in the terminal; failures
after setup starts also leave a verification receipt and available logs.

## Report a reproducible problem

Open an [issue](https://github.com/gengx-prog/causal-weathergraph/issues) with:

- The repository commit and exact command.
- Python and dependency versions, preferably `environment.json` and `pip-freeze.txt`.
- The relevant error trace, verification receipt and failing stage's log.
- Input identifiers/hashes and expected versus observed behavior.

Remove credentials and private paths before sharing. A small reproducible
example helps maintainers reproduce the problem and fix code or instructions.
Establish what failed before attributing it to the reviewer's computer.

请优先使用快速入口。安装、路径、依赖或数值核对失败都可以作为需要维护者
排查的问题报告。保留输出目录并提供命令、版本、错误堆栈及核对记录；分享
前移除凭据和私人路径。运行 CPU 示例无需安装 CUDA 或准备 GPU。
