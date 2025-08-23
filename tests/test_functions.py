import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

# Ensure the src directory is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from psd import functions  # noqa: E402


def numerical_grad(func: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute numerical gradient of ``func`` at ``x`` using finite differences."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = eps
        grad[i] = (func(x + e) - func(x - e)) / (2 * eps)
    return grad


float_arrays = st.lists(
    st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=4,
).map(lambda xs: np.array(xs, dtype=np.float64))


@given(float_arrays)
def test_separable_quartic_grad_matches_fd(x: np.ndarray) -> None:
    np.testing.assert_allclose(
        functions.separable_quartic_grad(x),
        numerical_grad(functions.separable_quartic, x),
        rtol=1e-5,
        atol=1e-5,
    )


@given(float_arrays)
def test_coupled_quartic_grad_matches_fd(x: np.ndarray) -> None:
    np.testing.assert_allclose(
        functions.coupled_quartic_grad(x),
        numerical_grad(functions.coupled_quartic, x),
        rtol=1e-5,
        atol=1e-5,
    )


@given(float_arrays)
def test_rosenbrock_grad_matches_fd(x: np.ndarray) -> None:
    if x.size < 2:
        pytest.skip("rosenbrock requires at least two dimensions")
    np.testing.assert_allclose(
        functions.rosenbrock_grad(x),
        numerical_grad(functions.rosenbrock, x),
        rtol=1e-5,
        atol=1e-5,
    )


@given(float_arrays)
def test_hessians_are_symmetric(x: np.ndarray) -> None:
    np.testing.assert_allclose(
        functions.separable_quartic_hess(x),
        functions.separable_quartic_hess(x).T,
    )
    np.testing.assert_allclose(
        functions.coupled_quartic_hess(x),
        functions.coupled_quartic_hess(x).T,
    )
    if x.size >= 2:
        np.testing.assert_allclose(
            functions.rosenbrock_hess(x),
            functions.rosenbrock_hess(x).T,
        )


@given(st.integers(min_value=1, max_value=4), st.integers(min_value=0, max_value=1000))
def test_random_quadratic_properties(d: int, seed: int) -> None:
    A, b = functions.random_quadratic(d, seed)
    x = np.random.default_rng(seed + 1).standard_normal(d)
    np.testing.assert_allclose(functions.random_quadratic_grad(x, A, b), A @ x - b)
    np.testing.assert_allclose(functions.random_quadratic_hess(A), A)
    eigvals = np.linalg.eigvalsh(A)
    assert np.all(eigvals >= 0.5)
    assert np.all(eigvals <= 1.5)
