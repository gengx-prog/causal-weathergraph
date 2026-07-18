"""Small-scale PCMCI validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_io import AtmosphericDataset
from .utils import get_logger

logger = get_logger("pcmci_validation")

CORE_EDGE_TYPES = (
    "wind_to_humidity",
    "humidity_to_cloud_cover",
    "wind_to_cloud_cover",
)


def tigramite_imports() -> tuple[Any, Any, Any] | None:
    """Return Tigramite classes if available."""
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI

        return pp, ParCorr, PCMCI
    except Exception as exc:
        logger.warning("Tigramite unavailable; PCMCI validation skipped: %r", exc)
        return None


def select_top_stable_edges(
    stable_edges: pd.DataFrame,
    top_n: int = 20,
    edge_types: tuple[str, ...] = CORE_EDGE_TYPES,
    regime: str = "all",
) -> pd.DataFrame:
    """Select unique top stable Granger edges for PCMCI validation."""
    eligible = stable_edges.copy()
    if "regime" in eligible.columns and regime:
        eligible = eligible[eligible["regime"] == regime].copy()
    pieces = []
    for edge_type in edge_types:
        subset = eligible[eligible["edge_type"] == edge_type].copy()
        if subset.empty:
            continue
        subset["abs_effect_score"] = subset["effect_score"].abs()
        subset = subset.sort_values(
            ["stability", "abs_effect_score"],
            ascending=[False, False],
        ).drop_duplicates(
            ["source_region", "target_region", "source_var", "target_var", "lag"]
        ).head(top_n)
        pieces.append(subset)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def unavailable_outputs(selected: pd.DataFrame, reason: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build CSV outputs when PCMCI cannot be run."""
    summary_rows = []
    edge_rows = []
    for edge_type in CORE_EDGE_TYPES:
        subset = selected[selected["edge_type"] == edge_type] if not selected.empty else selected
        summary_rows.append(
            {
                "edge_type": edge_type,
                "granger_top_edges": int(len(subset)),
                "pcmci_confirmed": 0,
                "confirmed_ratio": 0.0,
                "mean_pcmci_pvalue": np.nan,
                "mean_pcmci_qvalue": np.nan,
                "mean_abs_pcmci_partial_correlation": np.nan,
                "mean_granger_effect_score": float(subset["effect_score"].abs().mean()) if len(subset) else np.nan,
                "status": reason,
            }
        )
    for row in selected.itertuples(index=False):
        edge_rows.append(
            {
                "source_region": int(row.source_region),
                "target_region": int(row.target_region),
                "source_var": str(row.source_var),
                "target_var": str(row.target_var),
                "edge_type": str(row.edge_type),
                "granger_lag": int(row.lag),
                "granger_q": float(row.q_value),
                "granger_effect_score": float(row.effect_score),
                "pcmci_confirmed": False,
                "pcmci_lag": np.nan,
                "pcmci_pvalue": np.nan,
                "pcmci_qvalue": np.nan,
                "pcmci_partial_correlation": np.nan,
                "status": reason,
            }
        )
    return pd.DataFrame.from_records(summary_rows), pd.DataFrame.from_records(edge_rows)


def run_pcmci_validation(
    dataset: AtmosphericDataset,
    selected: pd.DataFrame,
    tau_max: int = 3,
    alpha: float = 0.05,
    start_year: int | None = 2019,
    max_conds_dim: int | None = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run PCMCI validation for selected edges."""
    imports = tigramite_imports()
    if imports is None:
        return unavailable_outputs(selected, "tigramite_unavailable")
    pp, ParCorr, PCMCI = imports
    var_index = {name: idx for idx, name in enumerate(dataset.variable_names)}
    validation_mask = np.ones(dataset.data.shape[0], dtype=bool)
    if start_year is not None and dataset.timestamps is not None:
        validation_mask = pd.DatetimeIndex(dataset.timestamps).year.to_numpy() >= int(start_year)
    edge_rows: list[dict[str, Any]] = []
    grouped = selected.groupby(["source_region", "target_region"], sort=False)
    for (source_region_raw, target_region_raw), pair_edges in grouped:
        source_region = int(source_region_raw)
        target_region = int(target_region_raw)
        involved_regions = [source_region] if source_region == target_region else [source_region, target_region]
        labels = []
        columns = []
        for region in involved_regions:
            for var_name, var_idx in var_index.items():
                labels.append(f"r{region}:{var_name}")
                columns.append(dataset.data[:, region, var_idx])
        matrix = np.column_stack(columns)
        finite = np.isfinite(matrix).all(axis=1) & validation_mask
        matrix = matrix[finite]
        raw_p_matrix = q_matrix = val_matrix = None
        pair_status = "ok"
        if len(matrix) <= tau_max + 10:
            pair_status = "insufficient_samples"
        else:
            dataframe = pp.DataFrame(matrix, var_names=labels)
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
            result = pcmci.run_pcmci(
                tau_min=1,
                tau_max=int(tau_max),
                pc_alpha=alpha,
                max_conds_dim=max_conds_dim,
                max_combinations=1,
                alpha_level=alpha,
                fdr_method="none",
            )
            raw_p_matrix = result["p_matrix"]
            val_matrix = result["val_matrix"]
            q_matrix = pcmci.get_corrected_pvalues(
                p_matrix=raw_p_matrix,
                fdr_method="fdr_bh",
                exclude_contemporaneous=True,
                tau_min=1,
                tau_max=int(tau_max),
            )

        for row in pair_edges.itertuples(index=False):
            source_var = str(row.source_var)
            target_var = str(row.target_var)
            source_label = f"r{source_region}:{source_var}"
            target_label = f"r{target_region}:{target_var}"
            lag = int(min(max(1, int(row.lag)), int(tau_max)))
            if (
                pair_status != "ok"
                or source_label not in labels
                or target_label not in labels
                or raw_p_matrix is None
                or q_matrix is None
                or val_matrix is None
            ):
                p_value = np.nan
                q_value = np.nan
                partial_correlation = np.nan
                confirmed = False
            else:
                source_idx = labels.index(source_label)
                target_idx = labels.index(target_label)
                p_value = float(raw_p_matrix[source_idx, target_idx, lag])
                q_value = float(q_matrix[source_idx, target_idx, lag])
                partial_correlation = float(val_matrix[source_idx, target_idx, lag])
                confirmed = bool(np.isfinite(q_value) and q_value < alpha)

            edge_rows.append(
                {
                    "source_region": source_region,
                    "target_region": target_region,
                    "source_var": source_var,
                    "target_var": target_var,
                    "edge_type": str(row.edge_type),
                    "granger_lag": int(row.lag),
                    "granger_q": float(row.q_value),
                    "granger_effect_score": float(row.effect_score),
                    "pcmci_confirmed": confirmed,
                    "pcmci_lag": lag,
                    "pcmci_pvalue": p_value,
                    "pcmci_qvalue": q_value,
                    "pcmci_partial_correlation": partial_correlation,
                    "validation_start_year": start_year,
                    "validation_samples": int(len(matrix)),
                    "max_conds_dim": max_conds_dim,
                    "status": pair_status,
                }
            )
    edge_df = pd.DataFrame.from_records(edge_rows)
    summary = summarize_pcmci(edge_df)
    return summary, edge_df


def summarize_pcmci(edges: pd.DataFrame) -> pd.DataFrame:
    """Summarize edge-level PCMCI validation results."""
    rows = []
    for edge_type in CORE_EDGE_TYPES:
        subset = edges[edges["edge_type"] == edge_type]
        confirmed = subset[subset["pcmci_confirmed"].astype(bool)] if len(subset) else subset
        rows.append(
            {
                "edge_type": edge_type,
                "granger_top_edges": int(len(subset)),
                "pcmci_confirmed": int(len(confirmed)),
                "confirmed_ratio": float(len(confirmed) / len(subset)) if len(subset) else 0.0,
                "mean_pcmci_pvalue": float(subset["pcmci_pvalue"].mean()) if len(subset) else np.nan,
                "mean_pcmci_qvalue": float(subset["pcmci_qvalue"].mean()) if len(subset) else np.nan,
                "mean_abs_pcmci_partial_correlation": (
                    float(subset["pcmci_partial_correlation"].abs().mean()) if len(subset) else np.nan
                ),
                "mean_granger_effect_score": float(subset["granger_effect_score"].abs().mean()) if len(subset) else np.nan,
                "status": "ok" if len(subset) and (subset["status"] == "ok").all() else "partial",
            }
        )
    return pd.DataFrame.from_records(rows)
