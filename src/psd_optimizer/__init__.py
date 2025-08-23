"""PyTorch optimizers for escaping saddle points.

This package exposes implementations of the original Perturbed Saddle-point
Descent (PSD) optimizer along with :class:`~psd_optimizer.perturbed_adam.PerturbedAdam`,
an adaptive variant that combines Adam with perturbation-based saddle point
escaping techniques.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .optimizer import PSDOptimizer
from .perturbed_adam import PerturbedAdam

try:
    __version__ = version("psd-optimizer")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["PSDOptimizer", "PerturbedAdam", "__version__"]
