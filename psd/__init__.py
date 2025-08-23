"""Utilities and reference implementations for PSD."""

from . import algorithms, functions
from .graph import find_optimal_path

try:  # Optional framework-specific optimisers
    from .framework_optimizers import PSDTorch, PSDTensorFlow
except Exception:  # pragma: no cover - dependencies may be missing
    PSDTorch = None  # type: ignore
    PSDTensorFlow = None  # type: ignore

__all__ = [
    "algorithms",
    "functions",
    "find_optimal_path",
    "PSDTorch",
    "PSDTensorFlow",
]
