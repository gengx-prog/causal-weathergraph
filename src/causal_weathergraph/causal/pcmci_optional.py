"""Optional Tigramite PCMCI/PCMCI+ interface.

Tigramite is intentionally not a required dependency. This module keeps the
research code ready for PCMCI experiments while allowing the default Granger
pipeline to run in lightweight environments.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causal_weathergraph.utils import get_logger

logger = get_logger("causal.pcmci_optional")


def tigramite_available() -> bool:
    """Return True if Tigramite can be imported."""
    try:
        import tigramite  # noqa: F401

        return True
    except Exception:
        return False


def run_pcmci_optional(
    data: np.ndarray,
    variable_names: list[str],
    max_lag: int = 12,
    config: dict[str, Any] | None = None,
) -> Any:
    """Run PCMCI if Tigramite is installed, otherwise return None."""
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except Exception:
        logger.warning(
            "Tigramite is not installed. Skipping PCMCI and continuing with the default Granger backend."
        )
        return None

    # Flatten region-variable nodes for a compact optional experiment.
    t, r, v = data.shape
    flattened = data.reshape(t, r * v)
    labels = [f"r{region}:{name}" for region in range(r) for name in variable_names]
    dataframe = pp.DataFrame(flattened, var_names=labels)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"))
    return pcmci.run_pcmci(tau_max=max_lag, pc_alpha=(config or {}).get("pc_alpha", 0.05))
