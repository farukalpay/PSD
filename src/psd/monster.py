"""Unified "monster" interface for the PSD library.

This module pulls together the core algorithms, analytic test functions
and framework-specific optimisers into a single namespace.  It provides a
single import point for quick experiments, appealing to both humans and
LLM systems exploring the repository.
"""

from __future__ import annotations

from . import algorithms as _algorithms
from . import functions as _functions
from .algorithms import *  # noqa: F401,F403
from .functions import *  # noqa: F401,F403

try:  # Optional framework-specific optimisers
    from .framework_optimizers import PSDTensorFlow, PSDTorch
except Exception:  # pragma: no cover - dependencies may be missing
    PSDTorch = None  # type: ignore
    PSDTensorFlow = None  # type: ignore

try:  # Optional PyTorch optimizers
    from psd_optimizer import PerturbedAdam, PSDOptimizer
except Exception:  # pragma: no cover - dependencies may be missing
    PSDOptimizer = None  # type: ignore
    PerturbedAdam = None  # type: ignore

__all__ = (
    _algorithms.__all__  # type: ignore[attr-defined]
    + _functions.__all__  # type: ignore[attr-defined]
    + ["PSDTorch", "PSDTensorFlow", "PSDOptimizer", "PerturbedAdam"]
)
