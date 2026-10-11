# Compatibility and troubleshooting / 运行环境与问题排查

Use the [software guide](SOFTWARE.md) for Python, dependency constraints and
isolated installation. Choose the experiment before downloading its inputs or
installing optional frameworks.

## Reference status and compatibility

All repository and release materials are provided for research and reference.
Recorded versions, example configurations, commands, timings, and successful
checks describe particular executions. They are not mandatory or exclusive
conditions for reproduction, minimum device specifications, or a guarantee of
compatibility with every environment. Users may select an equivalent compatible
setup and validate its outputs. Comparing the same scientific result still
requires the corresponding inputs, preprocessing, analysis settings, and
comparison criteria.

Execution also depends on conditions outside the project's control, including
third-party package installation, data-provider access, local permissions, and
available resources. A failure caused by such a condition is not, by itself,
evidence of a project defect. The cause should be established from the actual
error and the selected workflow; compatibility cannot be inferred from a device
label or a previous successful run alone.

本仓库及 Release 的全部内容仅供研究与参考。环境记录、版本列表、配置示例、
命令、耗时和历史验证结果，均不作为复现的强制条件或唯一条件，不代表最低
设备要求，也不构成适用于所有设备的运行保证。可采用等效的兼容环境并自行
验证结果。第三方依赖安装、数据服务访问、权限及可用资源等条件不由本项目
控制，由此引起的失败不能直接认定为本项目的问题；具体原因应依据实际错误
诊断。对照同一科学结果仍需匹配输入数据、预处理、分析设置与比较标准。

## Choose the workload first

| Workload | Supported entry point and evidence | Resource boundary |
| --- | --- | --- |
| Unit checks and small synthetic smoke run | Fresh Python 3.13.7 environment with `requirements-test.txt`; [recorded Windows/Ubuntu checks](GITHUB_EXECUTION.md) | CPU only; no scientific data download or GPU needed. A smoke run is not the full experiment. |
| Graph, satellite and WHEC calculations from archived processed inputs | [Portable reproduction guide](REPRODUCING.md), `scripts/reproduce.py` | CPU workflow. The combined analyses need additional inputs and working memory; resource use depends on the selected workload. |
| Raw-data acquisition and preprocessing | [Raw-data guide](RAW_DATA.md) and experiment-specific scripts | The published raw archive is about 108.69 GB; keeping ZIPs and extracted files requires more than 220 GB, plus processing workspace. A complete portable raw-to-results rerun has not been verified. |
| Optional neural weather-token experiment | Separate [weather-token module](../revision/weather_tokens/) | Requires its own PyTorch, data and checkpoints; supports CPU or CUDA. CUDA/VRAM needs depend on that configuration. No minimum VRAM benchmark is claimed. |

The recorded GitHub checks establish only their workflow and commit, not every
raw-processing stage, optional model or subsequent revision. A past passing run
does not guarantee that a new configuration will pass.

## Diagnose the actual failure

| Symptom or mismatch | What to check and do |
| --- | --- |
| Missing module or incompatible binary/package version | Confirm which Python executes the command. An operating system or Python version may not have a compatible package build; mixing environments can also produce import failures. To check against the reference profile, create a fresh virtual environment without `--system-site-packages`, install `requirements-test.txt`, and run `python -m pip check`. The historical full package inventory is not an installation requirement for every experiment. |
| Missing file, trained model or local path | Follow explicit input/output options in the portable guide and verify downloaded assets. Some helpers need explicit input and output paths. The 13 optional cloud-model checks in the recorded validation were skipped because their trained bundle was absent; that is not evidence of a GPU problem or a validated model. |
| Memory exhaustion, termination, very slow execution or insufficient disk | Record the failing command, workload size, free memory/disk and process error. Raw fields, dense arrays, concurrent workers and retained temporary files can increase resource use. Reduce concurrency or select a supported smaller workload when the command permits it; keep a new output directory. These are possible causes to investigate, not established diagnoses. |
| Excessive CPU threads or different numerical results | Use the documented two-thread settings for the portable checks. Preserve experiment-specific settings when reproducing a recorded experiment; the October 9 calibration record used one BLAS thread per worker. Compare using the reproduction runner's stated tolerances instead of demanding byte-identical floating-point output across platforms. |
| CUDA/PyTorch error in an optional model | Check the installed PyTorch build, driver, device availability and model configuration. The archived `+cu128` build describes PyTorch's CUDA build; it does not establish an independently installed CUDA toolkit. CUDA failures do not diagnose the CPU graph-analysis workflow. |
| Download/checksum failure | Follow the downloader's error and checksum record. An interrupted transfer, missing input or incorrect asset version must be distinguished from an analysis failure. Do not overwrite archived scientific outputs while troubleshooting. |
| Network, provider-access or filesystem-permission error | Check connectivity, proxy/firewall restrictions, any access required by the selected upstream provider, and write permission for the output directory. Public release downloads do not require the upstream acquisition credentials. Third-party service availability and local access policy are outside this project's control. |
| A failure persists in the supported clean environment | Treat it as a possible project defect. Report the exact command, commit, complete error/traceback with private paths removed, Python/dependency versions, input identifiers/hashes and a small reproducible example when possible. Include expected and observed behavior. |

Different operating systems, dependencies, inputs and resource limits can explain
a failure; a single failed attempt does not establish which cause applies. If a
failure persists after these checks, use the
[issue tracker](https://github.com/gengx-prog/causal-weathergraph/issues) with a
minimal reproducible example so it can be diagnosed. Do not include credentials,
private datasets, usernames, machine names, serial numbers or network identifiers.

遇到失败时，请先核对实际执行的 Python、依赖版本、输入文件与校验值、
可用内存和磁盘，以及所运行的具体任务。一次失败不能直接归因于代码，
也不能一概解释为使用者设备不足。如果核对后仍可复现，请提交命令、提交版本、
脱敏错误信息和最小复现示例，以便定位原因。无需公开账号、序列号、计算机名、
IP 或凭据。
