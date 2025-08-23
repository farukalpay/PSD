"""Utilities and reference implementations for PSD."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import algorithms, functions
from .config import PSDConfig
from .feature_flags import FLAGS, FeatureFlags, disable, enable
from .graph import GraphConfig, find_optimal_path

try:  # Optional framework-specific optimisers
    from .framework_optimizers import PSDTensorFlow, PSDTorch
except Exception:  # pragma: no cover - dependencies may be missing
    PSDTorch = None  # type: ignore
    PSDTensorFlow = None  # type: ignore

try:
    __version__ = version("psd-optimizer")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "algorithms",
    "functions",
    "find_optimal_path",
    "GraphConfig",
    "PSDConfig",
    "FeatureFlags",
    "FLAGS",
    "enable",
    "disable",
    "PSDTorch",
    "PSDTensorFlow",
    "__version__",
]
