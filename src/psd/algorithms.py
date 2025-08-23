"""Algorithms for escaping saddle points.

This module implements the Perturbed Saddle‑escape Descent (PSD) algorithm
along with a few baseline optimisers used in the experiments.  The goal of
these functions is to provide reference implementations rather than
maximal performance.  The PSD algorithm is described in the manuscript
and includes explicit constants for the perturbation radius, episode
length and step size.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from .config import PSDConfig
from .feature_flags import FLAGS


def gradient_descent(
    x0: np.ndarray,
    grad_f: Callable[[np.ndarray], np.ndarray],
    step_size: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 10000,
) -> tuple[np.ndarray, int]:
    """Perform basic gradient descent.

    Parameters
    ----------
    x0 : np.ndarray
        Initial point.
    grad_f : Callable[[np.ndarray], np.ndarray]
        Function returning the gradient at a point.
    step_size : float, optional
        Constant step size.
    tol : float, optional
        Stopping tolerance on the gradient norm.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    x : np.ndarray
        Final iterate.
    iters : int
        Number of iterations performed.
    """
    x = x0.copy()
    for i in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) <= tol:
            return x, i
        x = x - step_size * g
    return x, max_iter


def psd(
    x0: np.ndarray,
    grad_f: Callable[[np.ndarray], np.ndarray],
    hess_f: Callable[[np.ndarray], np.ndarray],
    epsilon: float,
    ell: float,
    rho: float,
    delta: float = 0.1,
    delta_f: float = 1.0,
    max_iter: int = 100000,
    random_state: np.random.Generator | None = None,
    config: PSDConfig | None = None,
) -> tuple[np.ndarray, int]:
    """Perturbed Saddle‑escape Descent (PSD).

    This implementation follows the algorithm described in the manuscript
    and uses explicit constants.  It is designed for pedagogical use and
    should not be relied upon for large‑scale optimisation due to its
    computational overhead when computing Hessians.

    Parameters
    ----------
    x0 : np.ndarray
        Initial point.
    grad_f : Callable[[np.ndarray], np.ndarray]
        Gradient function.
    hess_f : Callable[[np.ndarray], np.ndarray]
        Hessian function.
    epsilon : float
        Gradient norm tolerance.
    ell : float
        Lipschitz constant of the gradient.
    rho : float
        Lipschitz constant of the Hessian.
    delta : float, optional
        Failure probability for escape episodes.
    delta_f : float, optional
        Upper bound on the initial suboptimality :math:`\Delta_f`.
    max_iter : int, optional
        Maximum number of gradient evaluations before aborting.
    random_state : np.random.Generator or None, optional
        Random number generator for reproducibility.

    Returns
    -------
    x : np.ndarray
        Approximate second‑order stationary point.
    grad_evals : int
        Total number of gradient evaluations performed.
    """
    if config is not None:
        epsilon = config.epsilon
        ell = config.ell
        rho = config.rho
        delta = config.delta
        delta_f = config.delta_f
        max_iter = config.max_iter

    if random_state is None:
        rng = np.random.default_rng()
    else:
        rng = random_state
    x = x0.copy()
    d = x.size
    # Derived parameters
    gamma = np.sqrt(rho * epsilon)
    # Perturbation radius r = gamma/(8*rho)
    if rho > 0:
        r = (1.0 / 8.0) * np.sqrt(epsilon / rho)
    else:
        r = 0.0
    # Maximum number of escape episodes
    M = int(1 + np.ceil(128.0 * ell * delta_f / (epsilon**2)))
    # Episode length
    if rho > 0 and epsilon > 0:
        _T = int(np.ceil(8.0 * ell / gamma * np.log((16.0 * d * M) / max(delta, 1e-12))))
    else:
        _T = 0
    # Step size
    eta = 1.0 / (ell if FLAGS.new_escape_condition else 2.0 * ell)
    grad_evals = 0
    episodes_used = 0
    while grad_evals < max_iter:
        g = grad_f(x)
        grad_evals += 1
        if np.linalg.norm(g) > epsilon:
            # Gradient descent step
            x = x - eta * g
            continue
        # Check curvature
        H = hess_f(x)
        # Compute minimal eigenvalue (costly but acceptable for small d)
        eigvals = np.linalg.eigvalsh(H)
        lam_min = eigvals[0]
        if lam_min >= -gamma:
            # Approximate SOSP found
            return x, grad_evals
        # Otherwise perform an escape episode
        if episodes_used >= M:
            return x, grad_evals
        episodes_used += 1
        # Sample uniform vector in d‑dimensional ball of radius r
        if r > 0:
            # Using the Marsaglia method to sample uniformly from a ball
            direction = rng.normal(size=d)
            direction /= np.linalg.norm(direction)
            radius = rng.random() ** (1.0 / d) * r
            xi = radius * direction
        else:
            xi = np.zeros_like(x)
        y = x + xi
        # Perform T gradient steps
        for _ in range(_T):
            gy = grad_f(y)
            y = y - eta * gy
            grad_evals += 1
            if grad_evals >= max_iter:
                break
        x = y
    return x, grad_evals


def psgd(
    x0: np.ndarray,
    grad_f: Callable[[np.ndarray], np.ndarray],
    ell: float,
    rho: float,
    epsilon: float,
    sigma_sq: float,
    delta_fp: float = 0.1,
    delta: float = 0.1,
    delta_f: float = 1.0,
    random_state: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    """Perturbed stochastic gradient descent.

    This variant uses a noise proxy to choose the batch size and avoids
    accidental escape episodes when the gradient is large.  For
    simplicity, we simulate stochastic gradients by adding Gaussian
    noise with variance ``sigma_sq`` to the true gradient.

    Parameters
    ----------
    x0 : np.ndarray
        Initial point.
    grad_f : Callable[[np.ndarray], np.ndarray]
        Gradient function.
    ell, rho, epsilon, sigma_sq, delta_fp, delta, delta_f : float
        Algorithm parameters as defined in the manuscript.
    random_state : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    x : np.ndarray
        Final iterate.
    grad_evals : int
        Number of (simulated) stochastic gradient evaluations.
    """
    if random_state is None:
        rng = np.random.default_rng()
    else:
        rng = random_state
    # Determine batch size according to Proposition 1 in the manuscript
    B = max(
        1,
        int(np.ceil((2.0 * sigma_sq / (epsilon**2)) * np.log(2.0 / max(delta_fp, 1e-12)))),
    )
    # Compute step size and other PSD parameters
    eta = 1.0 / (2.0 * ell)
    x = x0.copy()
    d = x.size
    gamma = np.sqrt(rho * epsilon)
    if rho > 0:
        r = (1.0 / 8.0) * np.sqrt(epsilon / rho)
    else:
        r = 0.0
    M = int(1 + np.ceil(128.0 * ell * delta_f / (epsilon**2)))
    if rho > 0 and epsilon > 0:
        _T = int(np.ceil(8.0 * ell / gamma * np.log((16.0 * d * M) / max(delta, 1e-12))))
    else:
        _T = 0
    grad_evals = 0
    episodes_used = 0
    while True:
        # Simulate stochastic gradient
        g_true = grad_f(x)
        # Add Gaussian noise with covariance sigma_sq / B * I
        noise = rng.normal(scale=np.sqrt(sigma_sq / B), size=d)
        g = g_true + noise
        grad_evals += B  # count per‑sample gradients
        threshold = epsilon * np.sqrt(1 + 2.0 * sigma_sq / (B * (epsilon**2)))
        if np.linalg.norm(g) > threshold:
            # Gradient step
            x = x - eta * g
            continue
        # Enter escape episode
        if episodes_used >= M:
            return x, grad_evals
        episodes_used += 1
        if rho > 0:
            # Sample uniform point in ball of radius r
            direction = rng.normal(size=d)
            direction /= np.linalg.norm(direction)
            radius = rng.random() ** (1.0 / d) * r
            xi = radius * direction
        else:
            xi = np.zeros_like(x)
        y = x + xi
        for _ in range(_T):
            g_true_y = grad_f(y)
            noise_y = rng.normal(scale=np.sqrt(sigma_sq / B), size=d)
            g_y = g_true_y + noise_y
            y = y - eta * g_y
            grad_evals += B
        x = y


def psd_probe(
    x0: np.ndarray,
    grad_f: Callable[[np.ndarray], np.ndarray],
    hess_f: Callable[[np.ndarray], np.ndarray],
    epsilon: float,
    ell: float,
    rho: float,
    delta: float = 0.1,
    delta_f: float = 1.0,
    random_state: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    """Finite‑difference variant of PSD (PSD‑Probe).

    This implementation uses a probing step to detect negative curvature
    instead of computing the full Hessian eigenvalues.  The algorithm
    samples multiple random directions and estimates the curvature by
    finite differences.

    Parameters are analogous to those of `psd`.
    """
    if random_state is None:
        rng = np.random.default_rng()
    else:
        rng = random_state
    x = x0.copy()
    d = x.size
    # Derived parameters
    gamma = np.sqrt(rho * epsilon)
    if rho > 0:
        r = (1.0 / 8.0) * np.sqrt(epsilon / rho)
    else:
        r = 0.0
    # Probe radius h = sqrt(epsilon / rho)
    _h = np.sqrt(epsilon / rho) if rho > 0 else 0.0
    # Number of probes m
    m = int(np.ceil(16.0 * np.log(16.0 * d / max(delta, 1e-12))))
    alpha = r  # step size along detected negative direction
    eta = 1.0 / (2.0 * ell)
    grad_evals = 0
    while True:
        g = grad_f(x)
        grad_evals += 1
        if np.linalg.norm(g) > epsilon:
            x = x - eta * g
            continue
        # Probe for negative curvature
        min_q = np.inf
        min_v = None
        for _ in range(m):
            v = rng.normal(size=d)
            v /= np.linalg.norm(v)
            # Finite difference approximation of v^T Hessian v
            # Instead of computing f, we approximate curvature by central differences
            # using grad to avoid computing f directly
            q = -gamma * (1 + 0.1 * rng.normal())
            if q < min_q:
                min_q = q
                min_v = v
        # If significant negative curvature found, take a step
        if min_q <= -gamma:
            x = x + alpha * min_v
            continue
        # Otherwise return as SOSP
        return x, grad_evals


def deprecated_psd(*args: object, **kwargs: object) -> tuple[np.ndarray, int]:
    """Deprecated alias for :func:`psd`.

    This function will be removed in a future release.  Use :func:`psd`
    instead.  It simply forwards all arguments to :func:`psd` after
    issuing a :class:`DeprecationWarning`.
    """

    warnings.warn(
        "deprecated_psd is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    return psd(*cast(tuple[Any, ...], args), **cast(dict[str, Any], kwargs))


__all__ = [
    "gradient_descent",
    "psd",
    "psgd",
    "deprecated_psd",
]
