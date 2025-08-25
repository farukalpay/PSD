from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from psd import algorithms
from psd.config import PSDConfig


@settings(max_examples=50, deadline=None, derandomize=True)
@given(
    st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    ),
    st.floats(min_value=0.01, max_value=0.99),
)
def test_gradient_descent_converges_to_origin(values: list[float], step: float) -> None:
    x0 = np.array(values, dtype=float)

    def grad(x: np.ndarray) -> np.ndarray:
        return x

    x, _ = algorithms.gradient_descent(x0, grad, step_size=step, tol=1e-8, max_iter=1000)
    assert np.linalg.norm(x) <= 1e-3


def test_psd_converges_on_quadratic() -> None:
    def grad(x: np.ndarray) -> np.ndarray:
        return x

    def hess(x: np.ndarray) -> np.ndarray:
        return np.eye(len(x))

    x0 = np.array([1.0, -1.0])
    cfg = PSDConfig(epsilon=1e-6, ell=1.0, rho=1.0, max_iter=1000)
    x, _ = algorithms.psd(x0, grad, hess, 1e-6, 1.0, 1.0, config=cfg, random_state=np.random.default_rng(0))
    assert np.allclose(x, np.zeros_like(x0), atol=1e-5)


def test_psd_escape_episode_runs() -> None:
    def grad(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def hess(x: np.ndarray) -> np.ndarray:
        return np.diag([1.0, -1.0])

    x0 = np.array([0.0, 0.0])
    cfg = PSDConfig(epsilon=0.1, ell=1.0, rho=1.0, delta=0.1, delta_f=7.8e-5, max_iter=5)
    x, evals = algorithms.psd(x0, grad, hess, 0.1, 1.0, 1.0, config=cfg, random_state=np.random.default_rng(0))
    assert evals == cfg.max_iter


def test_psgd_behaves_like_gradient_descent() -> None:
    def grad(x: np.ndarray) -> np.ndarray:
        return x

    x0 = np.array([1.0])
    rng = np.random.default_rng(0)
    x, _ = algorithms.psgd(x0, grad, ell=1.0, rho=0.0, epsilon=0.1, sigma_sq=0.0, delta_f=7.8e-5, random_state=rng)
    assert np.linalg.norm(x) <= 0.1


def test_psd_probe_returns_sosp() -> None:
    def grad(x: np.ndarray) -> np.ndarray:
        return x

    def hess(x: np.ndarray) -> np.ndarray:
        return np.eye(len(x))

    x0 = np.array([1.0])

    class NegRNG:
        def normal(self: NegRNG, size: int | None = None) -> np.ndarray | float:
            return -1.0 if size is None else -np.ones(size)

        def random(self: NegRNG, size: int | None = None) -> np.ndarray | float:
            return 0.0 if size is None else np.zeros(size)

    x, _ = algorithms.psd_probe(x0, grad, hess, epsilon=0.1, ell=1.0, rho=1.0, random_state=NegRNG())
    assert np.linalg.norm(x) <= 0.1


def test_deprecated_psd_warns() -> None:
    def grad(x: np.ndarray) -> np.ndarray:
        return x

    def hess(x: np.ndarray) -> np.ndarray:
        return np.eye(len(x))

    x0 = np.array([1.0])
    cfg = PSDConfig(epsilon=1e-6, ell=1.0, rho=1.0, max_iter=10)
    with pytest.warns(DeprecationWarning):
        algorithms.deprecated_psd(x0, grad, hess, 1e-6, 1.0, 1.0, config=cfg)


def test_psd_handles_zero_hessian_lipschitz() -> None:
    """Algorithm should not divide by zero when ``rho`` is zero."""

    def grad(x: np.ndarray) -> np.ndarray:
        return x

    def hess(x: np.ndarray) -> np.ndarray:
        return np.eye(len(x))

    x0 = np.array([0.5, -0.5])
    cfg = PSDConfig(epsilon=1e-4, ell=1.0, rho=0.0, max_iter=1000)
    x, _ = algorithms.psd(x0, grad, hess, 1e-4, 1.0, 0.0, config=cfg)
    assert np.allclose(x, np.zeros_like(x0), atol=1e-4)


def test_psd_returns_immediately_with_small_gradient() -> None:
    """If the initial gradient is tiny, PSD should return without steps."""

    def grad(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def hess(x: np.ndarray) -> np.ndarray:
        return np.eye(len(x))

    x0 = np.array([1.0])
    cfg = PSDConfig(epsilon=0.1, ell=1.0, rho=1.0, max_iter=10)
    x, evals = algorithms.psd(x0, grad, hess, 0.1, 1.0, 1.0, config=cfg)
    assert evals == 1
    assert np.allclose(x, x0)
