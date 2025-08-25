# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython accelerated routines for PSD."""

import numpy as np
cimport numpy as np


def axpy(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] g, double eta):
    """Compute ``x -= eta * g`` in place.

    This function mirrors a BLAS ``axpy`` operation and is intended for
    performance‑critical loops.  It falls back to the Python implementation
    when the compiled extension is unavailable.
    """
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        x[i] -= eta * g[i]
