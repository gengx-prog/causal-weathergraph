"""Known-structure simulations for the major revision.

The null-calibration and graph-recovery estimands are deliberately separate.
All arrays retain original calendar rows when selecting a regime.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import importlib.metadata
import threading
import time
from typing import Callable

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 20260929
    n_null: int = 2048
    n_graph: int = 4096
    n_nodes: int = 6
    burn_in: int = 1024
    max_lag: int = 3
    hac_bandwidth: int = 64
    alpha: float = 0.05
    null_repetitions: int = 200
    graph_repetitions: int = 50
    pcmci_pc_alpha: float = 0.1
    null_external_only: bool = False
    null_variance: str = "hac"
    protocol_role: str = "primary"


NULL_SCENARIOS = ("ar1_homoskedastic", "ar4_heteroskedastic")
GRAPH_SCENARIOS = (
    "weak_ar", "strong_ar", "common_driver_observed",
    "common_driver_omitted", "known_external_regime_switching",
)


def rng_for(seed: int, family: int, scenario: int, repetition: int):
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(family, scenario, repetition)))


def bh(pvalues, alpha=0.05):
    """BH across the complete supplied family; invalid tests stay at p=1."""
    p = np.asarray(pvalues, dtype=float)
    p = np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), 1.0)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.minimum(adjusted, 1.0)
    return q <= alpha, q


def candidate_edges(n_nodes: int, max_lag: int):
    """Every cross-node directed lagged edge, including both orientations."""
    return [(source, target, lag) for target in range(n_nodes)
            for source in range(n_nodes) if source != target
            for lag in range(1, max_lag + 1)]


def lagged_design(data, max_lag=3):
    """Columns are (lag 1 all nodes, lag 2 all nodes, ...); no time compression."""
    data = np.asarray(data, dtype=float)
    n, p = data.shape
    design = np.full((n, p * max_lag), np.nan)
    for lag in range(1, max_lag + 1):
        design[lag:, (lag - 1) * p:lag * p] = data[:-lag]
    return design


def simulate_independent(config: SimulationConfig, scenario: str, rng):
    """Cross-series innovations and dynamics are independent under both DGPs.

    AR4 is intentionally misspecified by the lag-3 fitted target history.
    Its common deterministic variance schedule does not couple innovations.
    """
    if scenario not in NULL_SCENARIOS:
        raise ValueError(scenario)
    total = config.n_null + config.burn_in
    innovations = rng.normal(size=(total, config.n_nodes))
    if scenario == "ar4_heteroskedastic":
        sigma = np.where((np.arange(total) // 128) % 2 == 0, 0.35, 2.0)
        innovations *= sigma[:, None]
        coefficients = (0.4, 0.0, 0.0, 0.4)
    else:
        coefficients = (0.85,)
    data = np.zeros_like(innovations)
    for t in range(4, total):
        data[t] = innovations[t]
        for lag, coefficient in enumerate(coefficients, 1):
            data[t] += coefficient * data[t - lag]
    data = data[config.burn_in:]
    standardized = (data - data.mean(axis=0)) / data.std(axis=0)
    outcome_index = standardized.mean(axis=1)
    masks = {
        "all": np.ones(config.n_null, dtype=bool),
        "external_calendar": (np.arange(config.n_null) // 256) % 2 == 0,
        "outcome_selected": outcome_index >= np.quantile(outcome_index, 0.8),
    }
    return data, masks


def companion_radius(coefficients):
    """coefficients[lag - 1, target, source]."""
    lags, p, _ = coefficients.shape
    companion = np.zeros((p * lags, p * lags))
    companion[:p] = np.concatenate(coefficients, axis=1)
    if lags > 1:
        companion[p:, :-p] = np.eye(p * (lags - 1))
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def simulate_graph(config: SimulationConfig, scenario: str, rng):
    """Sparse ring-local VAR; the evaluated graph has the six measured nodes.

    For the observed-driver setting an extra series enters every method's
    conditioning/search data. Driver edges are not in the scored edge family.
    For omission, direct structural truth is scored, not a marginal VAR graph.
    """
    if scenario not in GRAPH_SCENARIOS:
        raise ValueError(scenario)
    p = config.n_nodes
    common_driver = scenario.startswith("common_driver_")
    full_p = p + int(common_driver)
    states = 2 if scenario == "known_external_regime_switching" else 1
    coefficients = np.zeros((states, config.max_lag, full_p, full_p))
    ar = 0.8 if scenario == "strong_ar" else (0.65 if common_driver else 0.25)
    delays = rng.integers(1, config.max_lag + 1, size=p)
    signs = rng.choice([-1.0, 1.0], size=p)
    truths = []
    for state in range(states):
        coefficients[state, 0, np.arange(p), np.arange(p)] = ar
        truth = set()
        for target in range(p):
            source = (target - 1) % p if state == 0 else (target + 1) % p
            lag = int(delays[target])
            coefficients[state, lag - 1, target, source] = 0.18 * signs[target] * (1 if state == 0 else -1)
            truth.add((source, target, lag))
        if common_driver:
            coefficients[state, 0, p, p] = 0.85
            coefficients[state, 0, :p, p] = 0.35
        truths.append(truth)
    radii = [companion_radius(a) for a in coefficients]
    if max(radii) >= 1:
        raise ValueError(f"Unstable stationary component: {radii}")
    total = config.n_graph + config.burn_in
    innovations = rng.normal(size=(total, full_p))
    data = np.zeros_like(innovations)
    state_array = (np.arange(total) // 512) % states
    for t in range(config.max_lag, total):
        data[t] = innovations[t]
        for lag in range(1, config.max_lag + 1):
            data[t] += coefficients[state_array[t], lag - 1] @ data[t - lag]
    data = data[config.burn_in:]
    state_array = state_array[config.burn_in:]
    if scenario == "common_driver_omitted":
        data = data[:, :p]
    masks = {f"state_{s}": state_array == s for s in range(states)}
    # Match PCMCI's default 2*tau_max response cutoff for every estimator.
    # The original predictor rows remain available on the full timeline.
    for mask in masks.values():
        mask[:2 * config.max_lag] = False
    return data, masks, truths, radii


def regression_pvalues(data, mask, edges, config, conditioning="own_history", variance_method="hac"):
    """Run exactly the same inference implementation as the ERA5 revision."""
    from revision.inference import fit_edge
    if variance_method == "cluster":
        from revision.inference import fit_edge_block_cluster
    elif variance_method != "hac":
        raise ValueError(variance_method)

    if conditioning not in ("own_history", "all_observed_history"):
        raise ValueError(conditioning)
    data = np.asarray(data, dtype=float)
    design = lagged_design(data, config.max_lag)
    p = data.shape[1]
    methods = ["ols", "hac64"] if variance_method == "hac" else ["ols", "cluster64", "cluster128", "cluster256"]
    output = {name: [] for name in methods}
    invalid = 0
    sample_sizes = []
    for source, target, lag in edges:
        source_col = (lag - 1) * p + source
        if conditioning == "own_history":
            control_cols = [k * p + target for k in range(config.max_lag)]
        else:
            control_cols = [k for k in range(p * config.max_lag) if k != source_col]
        if variance_method == "hac":
            fitted = fit_edge(data[:, target], design[:, control_cols], design[:, source_col],
                              mask, bandwidths=(config.hac_bandwidth,))
        else:
            fitted = fit_edge_block_cluster(data[:, target], design[:, control_cols], design[:, source_col],
                                            mask, block_lengths=(64, 128, 256))
        for method in methods:
            output[method].append(fitted[f"p_{method}"])
        invalid += int(not fitted.get("valid", True))
        sample_sizes.append(fitted["n_samples"])
    return {key: np.asarray(value) for key, value in output.items()}, {
        "invalid_tests": invalid, "min_samples": int(min(sample_sizes)),
        "max_samples": int(max(sample_sizes)),
    }


def pcmci_pvalues(data, mask, edges, config):
    """Official Tigramite PCMCI, ParCorr, target-only mask on full chronology."""
    from tigramite.data_processing import DataFrame
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    excluded = np.broadcast_to(~np.asarray(mask, dtype=bool)[:, None], np.shape(data)).copy()
    frame = DataFrame(np.asarray(data, dtype=float), mask=excluded)
    pcmci = PCMCI(dataframe=frame, cond_ind_test=ParCorr(significance="analytic", mask_type="y"), verbosity=0)
    result = pcmci.run_pcmci(tau_min=1, tau_max=config.max_lag,
                             pc_alpha=config.pcmci_pc_alpha, alpha_level=config.alpha)
    pvalues = np.asarray([result["p_matrix"][s, t, lag] for s, t, lag in edges])
    return pvalues, {"invalid_tests": int(np.sum(~np.isfinite(pvalues)))}


def score_discoveries(discovered, truth, edges):
    true = np.asarray([edge in truth for edge in edges], dtype=bool)
    selected = np.asarray(discovered, dtype=bool)
    tp = int(np.sum(selected & true))
    fp = int(np.sum(selected & ~true))
    fn = int(np.sum(~selected & true))
    tn = int(np.sum(~selected & ~true))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "discoveries": tp + fp, "precision": precision, "recall": recall,
            "f1": f1, "shd": fp + fn, "fdp": fp / max(tp + fp, 1),
            "any_false_discovery": int(fp > 0)}


def wilson_interval(successes, trials, confidence=0.95):
    if trials <= 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(0.5 + confidence / 2)
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (float(center - half), float(center + half))


def bounded_mean_interval(values, confidence=0.95):
    """Monte Carlo mean and Student-t CI, clipped to [0, 1]."""
    from scipy.stats import t
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    if len(x) < 2:
        return mean, float("nan"), float("nan"), float("nan")
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    half = float(t.ppf(0.5 + confidence / 2, len(x) - 1) * se)
    return mean, max(0.0, mean - half), min(1.0, mean + half), se


def measured_call(function: Callable, *args, **kwargs):
    """Wall time + sampled process RSS, without claiming isolated native peaks.

    A 10-ms monitor samples resident memory. Baseline and absolute peak are
    reported; allocations cached from preceding calls can affect these values.
    This is not tracemalloc and does include native process resident memory.
    """
    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        process = None
    gc.collect()
    baseline = process.memory_info().rss if process else float("nan")
    peak = [baseline]
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)

    watcher = threading.Thread(target=monitor, daemon=True) if process else None
    if watcher:
        watcher.start()
    start = time.perf_counter()
    try:
        result = function(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start
        if process:
            peak[0] = max(peak[0], process.memory_info().rss)
        stop.set()
        if watcher:
            watcher.join()
    return result, {"wall_seconds": elapsed, "rss_baseline_mib": baseline / 2**20,
                    "sampled_peak_rss_mib": peak[0] / 2**20,
                    "sampled_incremental_rss_mib": (peak[0] - baseline) / 2**20}


def package_versions():
    result = {}
    for package in ("numpy", "scipy", "tigramite", "statsmodels", "psutil"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result
