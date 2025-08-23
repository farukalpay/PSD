import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from psd import algorithms, functions
from psd.config import PSDConfig

# Strategies for vectors
_vector_1d = hnp.arrays(
    dtype=np.float64,
    shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=5),
    elements=st.floats(-2, 2, allow_nan=False, allow_infinity=False),
)

_vector_1d_ge2 = hnp.arrays(
    dtype=np.float64,
    shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=5),
    elements=st.floats(-2, 2, allow_nan=False, allow_infinity=False),
)


def _finite_diff_grad(f: callable, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    d = x.size
    grad = np.zeros(d)
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        grad[i] = (f(x + eps * e) - f(x - eps * e)) / (2 * eps)
    return grad


def _finite_diff_hess(g: callable, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    d = x.size
    hess = np.zeros((d, d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        hess[:, i] = (g(x + eps * e) - g(x - eps * e)) / (2 * eps)
    return hess


@settings(max_examples=25, deadline=None)
@given(_vector_1d)
@pytest.mark.fast
def test_separable_quartic_grad_hess_match(x: np.ndarray) -> None:
    f = functions.separable_quartic
    g = functions.separable_quartic_grad
    h = functions.separable_quartic_hess
    num_grad = _finite_diff_grad(f, x)
    num_hess = _finite_diff_hess(g, x)
    np.testing.assert_allclose(g(x), num_grad, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(h(x), num_hess, rtol=1e-4, atol=1e-4)


@settings(max_examples=20, deadline=None)
@given(_vector_1d_ge2)
@pytest.mark.slow
def test_rosenbrock_grad_hess_match(x: np.ndarray) -> None:
    f = functions.rosenbrock
    g = functions.rosenbrock_grad
    h = functions.rosenbrock_hess
    num_grad = _finite_diff_grad(f, x)
    num_hess = _finite_diff_hess(g, x)
    np.testing.assert_allclose(g(x), num_grad, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(h(x), num_hess, rtol=1e-4, atol=1e-4)


@settings(max_examples=20, deadline=None)
@given(_vector_1d, st.integers(min_value=0, max_value=2**32 - 1))
@pytest.mark.fast
def test_random_quadratic_determinism_and_derivatives(x: np.ndarray, seed: int) -> None:
    d = x.size
    A1, b1 = functions.random_quadratic(d, seed)
    A2, b2 = functions.random_quadratic(d, seed)
    assert np.allclose(A1, A2)
    assert np.allclose(b1, b2)
    A3, b3 = functions.random_quadratic(d, seed + 1)
    assert not (np.allclose(A1, A3) and np.allclose(b1, b3))
    eigvals = np.linalg.eigvalsh(A1)
    assert np.all(eigvals >= 0.5 - 1e-8)
    assert np.all(eigvals <= 1.5 + 1e-8)

    def f(z: np.ndarray) -> float:
        return functions.random_quadratic_value(z, A1, b1)

    def g(z: np.ndarray) -> np.ndarray:
        return functions.random_quadratic_grad(z, A1, b1)

    num_grad = _finite_diff_grad(f, x)
    num_hess = _finite_diff_hess(g, x)
    np.testing.assert_allclose(g(x), num_grad, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(functions.random_quadratic_hess(A1), num_hess, rtol=1e-5, atol=1e-6)


@settings(max_examples=5, deadline=None)
@given(st.integers(min_value=0, max_value=2**32 - 1))
@pytest.mark.slow
def test_psd_deterministic_given_seed(seed: int) -> None:
    def grad(_: np.ndarray) -> np.ndarray:
        return np.zeros(2)

    def hess(_: np.ndarray) -> np.ndarray:
        return np.diag([1.0, -1.0])

    x0 = np.array([0.0, 0.0])
    cfg = PSDConfig(epsilon=0.1, ell=1.0, rho=1.0, delta=0.1, delta_f=7.8e-5, max_iter=5)
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    x1, _ = algorithms.psd(x0, grad, hess, 0.1, 1.0, 1.0, config=cfg, random_state=rng1)
    x2, _ = algorithms.psd(x0, grad, hess, 0.1, 1.0, 1.0, config=cfg, random_state=rng2)
    assert np.allclose(x1, x2)
    rng3 = np.random.default_rng(seed + 1)
    x3, _ = algorithms.psd(x0, grad, hess, 0.1, 1.0, 1.0, config=cfg, random_state=rng3)
    assert not np.allclose(x1, x3)
