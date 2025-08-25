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


@settings(max_examples=25, deadline=None, derandomize=True)
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


@settings(max_examples=20, deadline=None, derandomize=True)
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


def test_rosenbrock_hess_edge_case() -> None:
    """The Rosenbrock Hessian should handle one‑dimensional inputs."""
    x = np.array([1.23])
    h = functions.rosenbrock_hess(x)
    assert h.shape == (1, 1)
    assert np.isclose(h[0, 0], 200.0)


def test_rosenbrock_hess_large_dim_is_finite() -> None:
    """Evaluate numerical stability on a large random vector."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    h = functions.rosenbrock_hess(x)
    assert np.all(np.isfinite(h))


def test_rosenbrock_hess_cython_matches_python() -> None:
    """Cython and NumPy implementations should agree."""
    try:
        from psd._rosenbrock import rosenbrock_hess_fast
    except Exception:  # pragma: no cover - extension not built
        pytest.skip("Cython extension not available")

    rng = np.random.default_rng(1)
    x = rng.standard_normal(10)
    h_fast = rosenbrock_hess_fast(x)

    # Re‑implement the Python version locally for comparison.  This keeps
    # the test independent of whether ``functions.rosenbrock_hess``
    # dispatches to the Cython version.
    d = len(x)
    h_py = np.zeros((d, d))
    if d > 1:
        idx = np.arange(d - 1)
        diag = 1200.0 * x[idx] ** 2 - 400.0 * x[idx + 1] + 2.0
        h_py[idx, idx] = diag
        h_py[idx + 1, idx + 1] += 200.0
        off = -400.0 * x[idx]
        h_py[idx, idx + 1] = off
        h_py[idx + 1, idx] = off
    else:
        h_py[0, 0] = 200.0
    np.testing.assert_allclose(h_fast, h_py, rtol=1e-12, atol=1e-12)


@settings(max_examples=20, deadline=None, derandomize=True)
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


@settings(max_examples=5, deadline=None, derandomize=True)
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
