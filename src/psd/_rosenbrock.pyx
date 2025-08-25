# cython: language_level=3
"""Cython implementation of the Rosenbrock Hessian.

The pure NumPy version in :mod:`psd.functions` is adequate for small
problems but becomes a bottleneck for large ``d``.  This Cython routine
uses explicit loops and typed memoryviews to eliminate Python overhead
and serves as a reference for how performance‑critical sections could be
ported to a compiled extension.
"""

import numpy as np
cimport numpy as np


def rosenbrock_hess_fast(np.ndarray[np.float64_t, ndim=1] x):
    """Compute the Rosenbrock Hessian using Cython loops.

    Parameters
    ----------
    x:
        Input vector of length ``d``.

    Returns
    -------
    np.ndarray
        Hessian matrix of shape ``(d, d)``.
    """
    cdef Py_ssize_t d = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] hess = np.zeros((d, d), dtype=np.float64)
    cdef Py_ssize_t i
    if d > 1:
        for i in range(d - 1):
            hess[i, i] = 1200.0 * x[i] * x[i] - 400.0 * x[i + 1] + 2.0
            hess[i + 1, i + 1] += 200.0
            hess[i, i + 1] = -400.0 * x[i]
            hess[i + 1, i] = -400.0 * x[i]
    else:
        hess[0, 0] = 200.0
    return hess
