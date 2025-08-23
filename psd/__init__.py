"""Utilities and reference implementations for PSD."""

from . import algorithms, functions

try:  # Optional framework-specific optimisers
    from .framework_optimizers import PSDTorch, PSDTensorFlow
except Exception:  # pragma: no cover - dependencies may be missing
    PSDTorch = None  # type: ignore
    PSDTensorFlow = None  # type: ignore

__all__ = ["algorithms", "functions", "PSDTorch", "PSDTensorFlow"]
