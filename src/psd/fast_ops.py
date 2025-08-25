"""Optional fast linear algebra operations.

This module attempts to import a Cython implementation for common
in‑place vector updates.  If the extension is not available (for example
when building from source without Cython), a pure Python fallback is
used.  The functions follow the BLAS ``axpy`` convention where
``x -= a * y`` is performed in place.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - speed‑critical Cython extension
    from ._fast_ops import axpy  # type: ignore[import]
except Exception:  # pragma: no cover - extension unavailable
    def axpy(x: np.ndarray, g: np.ndarray, eta: float) -> None:
        """Fallback implementation of ``x -= eta * g``.

        Parameters
        ----------
        x, g : np.ndarray
            Arrays of identical shape. ``x`` is modified in place.
        eta : float
            Scaling factor.
        """

        x -= eta * g
