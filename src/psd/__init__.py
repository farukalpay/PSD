"""Utilities and reference implementations for PSD."""

from . import algorithms, functions
from .config import PSDConfig
from .feature_flags import FLAGS, FeatureFlags, disable, enable
from .graph import GraphConfig, find_optimal_path

try:  # Optional framework-specific optimisers
    from .framework_optimizers import PSDTensorFlow, PSDTorch
except Exception:  # pragma: no cover - dependencies may be missing
    PSDTorch = None  # type: ignore
    PSDTensorFlow = None  # type: ignore

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
]
