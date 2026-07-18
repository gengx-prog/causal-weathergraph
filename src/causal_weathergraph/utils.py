"""Shared utilities for the Causal WeatherGraph pipeline."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


LOGGER_NAME = "causal_weathergraph"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure a compact console logger and return it."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the project namespace."""
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def project_root_from_file(file_path: str | Path) -> Path:
    """Return the repository root for a script under scripts/."""
    return Path(file_path).resolve().parents[1]


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a path, falling back to ``base / path`` for relative paths."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists() or base is None:
        return candidate
    based = Path(base) / candidate
    return based if based.exists() else candidate


def set_random_seed(seed: int = 42) -> None:
    """Seed NumPy and the standard random module."""
    random.seed(seed)
    np.random.seed(seed)


def json_default(obj: Any) -> Any:
    """JSON serializer for NumPy, pandas, and pathlib objects."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def save_json(data: Any, path: str | Path) -> None:
    """Write JSON with stable indentation."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=json_default)


def safe_float(value: Any) -> float:
    """Convert a value to float, returning NaN on failure."""
    try:
        return float(value)
    except Exception:
        return float("nan")


def canonicalize_variable_request(name: str) -> str:
    """Map common user-facing variable labels onto canonical project names."""
    normalized = "".join(ch for ch in str(name).lower() if ch.isalnum())
    aliases = {
        "temperature": {"temperature", "temp", "t", "t2m", "t850", "airtemperature"},
        "humidity": {
            "humidity",
            "specifichumidity",
            "relativehumidity",
            "q",
            "q850",
            "rh",
        },
        "wind": {"wind", "windspeed", "speed", "wspd", "u", "v", "u850", "v850"},
        "cloud_cover": {
            "cloudcover",
            "totalcloudcover",
            "tcc",
            "cloud",
            "cc",
            "cloudfraction",
        },
    }
    for canonical, names in aliases.items():
        if normalized in names:
            return canonical
    return str(name)
