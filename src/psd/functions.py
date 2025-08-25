"""Test functions for the PSD experiments.

This module defines a small collection of smooth non‑convex functions
along with their gradients and Hessians.  These functions are used by
`experiments.py` and can also be used independently for further
experimentation.  Each function triple is wrapped in the :class:`TestFunction`
dataclass and registered in :data:`TEST_FUNCTIONS` for convenient access.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# Optional Cython acceleration for the Rosenbrock Hessian.  Importing is
# wrapped in a ``try`` so that the pure Python fall-back is used when the
# extension has not been built.
try:  # pragma: no cover - depends on build environment
    from ._rosenbrock import rosenbrock_hess_fast
except Exception:  # pragma: no cover - extension may be missing
    rosenbrock_hess_fast = None  # type: ignore

Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class TestFunction:
    """Container holding value, gradient and Hessian callables for a function."""

    value: Callable[[Array], float]
    grad: Callable[[Array], Array]
    hess: Callable[[Array], Array]


################################################################################
# Separable quartic function
################################################################################


def separable_quartic(x: Array) -> float:
    r"""Compute the value of the separable quartic function.

    The separable quartic is defined as

    .. math::

        f(x) = \sum_{i=1}^d (x_i^4 - x_i^2).

    It has multiple saddle points with Hessian having both positive and
    negative eigenvalues.

    Parameters
    ----------
    x : np.ndarray
        Point at which to evaluate the function, shape (d,).

    Returns
    -------
    float
        Function value at x.
    """
    return float(np.sum(x**4 - x**2))


def separable_quartic_grad(x: Array) -> Array:
    r"""Gradient of the separable quartic function.

    Gradient is given by :math:`\grad f(x) = 4 x^3 - 2 x`.

    Parameters
    ----------
    x : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Gradient at x.
    """
    return 4.0 * x**3 - 2.0 * x


def separable_quartic_hess(x: Array) -> Array:
    """Hessian of the separable quartic function.

    The Hessian is diagonal with entries :math:`12 x_i^2 - 2`.

    Parameters
    ----------
    x : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Diagonal Hessian matrix of shape (d, d).
    """
    diag = 12.0 * x**2 - 2.0
    return np.diag(diag)


################################################################################
# Coupled quartic function
################################################################################


def coupled_quartic(x: Array) -> float:
    r"""Compute the value of the coupled quartic function.

    This function couples the separable quartic with a weak quadratic term:

    .. math::

        f(x) = \sum_{i=1}^d (x_i^4 - x_i^2) + 0.1 \sum_{i<j} x_i x_j.

    Parameters
    ----------
    x : np.ndarray
        Point at which to evaluate the function.

    Returns
    -------
    float
        Function value.
    """
    # compute separable quartic part
    quartic = np.sum(x**4 - x**2)
    # sum_{i<j} x_i x_j = ( (sum x)^2 - sum x^2 ) / 2
    total = np.sum(x)
    sum_sq = np.dot(x, x)
    coupling = 0.05 * (total * total - sum_sq)
    return float(quartic + coupling)


def coupled_quartic_grad(x: Array) -> Array:
    """Gradient of the coupled quartic function.

    The gradient is the sum of the separable quartic gradient and the
    gradient of the coupling term.

    Parameters
    ----------
    x : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Gradient vector of shape (d,).
    """
    total = np.sum(x)
    grad = separable_quartic_grad(x)
    # gradient of coupling term: 0.1 * (sum_j x_j - x_i)
    return grad + 0.1 * (total - x)


def coupled_quartic_hess(x: Array) -> Array:
    """Hessian of the coupled quartic function.

    The Hessian has a diagonal part from the separable quartic and a dense
    0.1 everywhere except diagonal from the coupling term.

    Parameters
    ----------
    x : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Hessian matrix of shape (d, d).
    """
    d = x.size
    diag = 12.0 * x**2 - 2.0
    hess = np.full((d, d), 0.1)
    np.fill_diagonal(hess, diag)
    return hess


################################################################################
# Rosenbrock function
################################################################################


def rosenbrock(x: Array) -> float:
    r"""Compute the Rosenbrock function in d dimensions.

    The standard Rosenbrock function in d dimensions is defined as

    .. math::

        f(x) = \sum_{i=1}^{d-1} [100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2].

    Parameters
    ----------
    x : np.ndarray
        Input vector of length d.

    Returns
    -------
    float
        Function value.
    """
    x = np.asarray(x)
    xi = x[:-1]
    xip1 = x[1:]
    return float(np.sum(100.0 * (xip1 - xi**2) ** 2 + (1.0 - xi) ** 2))


def rosenbrock_grad(x: Array) -> Array:
    """Gradient of the Rosenbrock function.

    The gradient is computed component‑wise according to the standard
    expression.

    Parameters
    ----------
    x : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Gradient vector.
    """
    x = np.asarray(x)
    grad = np.zeros_like(x)
    grad[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    grad[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return grad


def rosenbrock_hess(x: Array) -> Array:
    r"""Hessian of the Rosenbrock function.

    The implementation dispatches to a Cython routine when available for
    efficiency.  The mathematical form of the Hessian is

    .. math::

        H_{i,i} = 1200 x_i^2 - 400 x_{i+1} + 2, \quad
        H_{i,i+1} = H_{i+1,i} = -400 x_i, \quad
        H_{d,d} = 200.

    Parameters
    ----------
    x : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Hessian matrix.
    """
    x = np.asarray(x)
    if rosenbrock_hess_fast is not None:
        # Use the compiled version for large ``d`` to avoid Python overhead.
        return rosenbrock_hess_fast(x)

    d = len(x)
    hess = np.zeros((d, d))
    if d > 1:
        idx = np.arange(d - 1)
        diag = 1200.0 * x[idx] ** 2 - 400.0 * x[idx + 1] + 2.0
        hess[idx, idx] = diag
        hess[idx + 1, idx + 1] += 200.0
        off = -400.0 * x[idx]
        hess[idx, idx + 1] = off
        hess[idx + 1, idx] = off
    else:
        hess[0, 0] = 200.0
    return hess


################################################################################
# Random quadratic function
################################################################################


def random_quadratic(d: int, seed: int | None = None) -> tuple[Array, Array]:
    r"""Generate a random quadratic function with controlled spectrum.

    The function has the form

    .. math::

        f(x) = \tfrac{1}{2} x^T A x - b^T x,

    where `A` is symmetric positive semi‑definite with eigenvalues drawn
    uniformly between 0.5 and 1.5, and `b` is a random vector.  This
    function is convex and therefore not strictly useful for saddle escape
    experiments but included for completeness.

    Parameters
    ----------
    d : int
        Dimension of the problem.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The pair `(A, b)`.
    """
    rng = np.random.default_rng(seed)
    # Random orthonormal matrix Q
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    # Eigenvalues uniformly in [0.5, 1.5]
    eigvals = 0.5 + rng.random(d)
    A = Q @ np.diag(eigvals) @ Q.T
    b = rng.standard_normal(d)
    return A, b


def random_quadratic_value(x: Array, A: Array, b: Array) -> float:
    """Value of the random quadratic function at x."""
    return 0.5 * float(x @ (A @ x)) - float(b @ x)


def random_quadratic_grad(x: Array, A: Array, b: Array) -> Array:
    """Gradient of the random quadratic function."""
    return A @ x - b


def random_quadratic_hess(A: Array) -> Array:
    """Hessian of the random quadratic function (constant)."""
    return A


# ---------------------------------------------------------------------------
# Convenience registry
# ---------------------------------------------------------------------------

SEPARABLE_QUARTIC = TestFunction(
    value=separable_quartic,
    grad=separable_quartic_grad,
    hess=separable_quartic_hess,
)

COUPLED_QUARTIC = TestFunction(
    value=coupled_quartic,
    grad=coupled_quartic_grad,
    hess=coupled_quartic_hess,
)

ROSENBROCK = TestFunction(
    value=rosenbrock,
    grad=rosenbrock_grad,
    hess=rosenbrock_hess,
)

TEST_FUNCTIONS: dict[str, TestFunction] = {
    "separable_quartic": SEPARABLE_QUARTIC,
    "coupled_quartic": COUPLED_QUARTIC,
    "rosenbrock": ROSENBROCK,
}


__all__ = [
    "TestFunction",
    "separable_quartic",
    "separable_quartic_grad",
    "separable_quartic_hess",
    "coupled_quartic",
    "coupled_quartic_grad",
    "coupled_quartic_hess",
    "rosenbrock",
    "rosenbrock_grad",
    "rosenbrock_hess",
    "random_quadratic",
    "random_quadratic_grad",
    "random_quadratic_hess",
    "SEPARABLE_QUARTIC",
    "COUPLED_QUARTIC",
    "ROSENBROCK",
    "TEST_FUNCTIONS",
]
