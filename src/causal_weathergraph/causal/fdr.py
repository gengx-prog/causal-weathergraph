"""False-discovery-rate correction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(p_values, dtype=float)
    q = np.ones_like(p, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    p_finite = np.clip(p[finite], 0.0, 1.0)
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_finite = np.empty_like(adjusted)
    q_finite[order] = adjusted
    q[finite] = q_finite
    return q


def apply_fdr_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    p_col: str = "p_value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply BH FDR correction independently within each group."""
    if df.empty:
        df = df.copy()
        df["q_value"] = []
        df["significant"] = []
        return df
    corrected = df.copy()
    corrected["q_value"] = 1.0
    for _, idx in corrected.groupby(group_cols, dropna=False).groups.items():
        idx_list = list(idx)
        corrected.loc[idx_list, "q_value"] = benjamini_hochberg(
            corrected.loc[idx_list, p_col].to_numpy(dtype=float)
        )
    corrected["significant"] = corrected["q_value"] < alpha
    return corrected
