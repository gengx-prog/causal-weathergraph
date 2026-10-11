# Compatibility and troubleshooting / 运行环境与问题排查

Use the [software guide](SOFTWARE.md) for Python, dependency constraints and
isolated installation. Choose the experiment before downloading its inputs or
installing optional frameworks.

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
| Missing module or incompatible binary/package version | Confirm which Python executes the command. Create a fresh virtual environment without `--system-site-packages`; install `requirements-test.txt` and run `python -m pip check`. A different Python installation may lack the scientific packages even when its Python version matches. Do not substitute the historical full package inventory for the clean installation profile. |
| Missing file, trained model or local path | Follow explicit input/output options in the portable guide and verify downloaded assets. Some helpers need explicit input and output paths. The 13 optional cloud-model checks in the recorded validation were skipped because their trained bundle was absent; that is not evidence of a GPU problem or a validated model. |
| Memory exhaustion, termination, very slow execution or insufficient disk | Record the failing command, workload size, free memory/disk and process error. Raw fields, dense arrays, concurrent workers and retained temporary files can increase resource use. Reduce concurrency or select a supported smaller workload when the command permits it; keep a new output directory. These are possible causes to investigate, not established diagnoses. |
| Excessive CPU threads or different numerical results | Use the documented two-thread settings for the portable checks. Preserve experiment-specific settings when reproducing a recorded experiment; the October 9 calibration record used one BLAS thread per worker. Compare using the reproduction runner's stated tolerances instead of demanding byte-identical floating-point output across platforms. |
| CUDA/PyTorch error in an optional model | Check the installed PyTorch build, driver, device availability and model configuration. The archived `+cu128` build describes PyTorch's CUDA build; it does not establish an independently installed CUDA toolkit. CUDA failures do not diagnose the CPU graph-analysis workflow. |
| Download/checksum failure | Follow the downloader's error and checksum record. An interrupted transfer, missing input or incorrect asset version must be distinguished from an analysis failure. Do not overwrite archived scientific outputs while troubleshooting. |
| A failure persists in the supported clean environment | Treat it as a possible project defect. Report the exact command, commit, complete error/traceback with private paths removed, Python/dependency versions, input identifiers/hashes and a small reproducible example when possible. Include expected and observed behavior. |

Different operating systems, dependencies, inputs and resource limits can explain
a failure; a single failed attempt does not establish which cause applies.
Conversely, environment differences do **not** rule out bugs in this project.
Use the [issue tracker](https://github.com/gengx-prog/causal-weathergraph/issues)
for reproducible problems. Do not include credentials, private datasets,
usernames, machine names, serial numbers or network identifiers.

遇到失败时，请先核对实际执行的 Python、依赖版本、输入文件与校验值、
可用内存和磁盘，以及所运行的具体任务。一次失败不能直接归因于代码，
也不能一概解释为使用者设备不足。如果在说明中的干净环境和受支持流程中
仍可复现，请提交命令、提交版本、脱敏错误信息和最小复现示例；项目本身
的缺陷同样需要检查和修复。无需公开账号、序列号、计算机名、IP 或凭据。
