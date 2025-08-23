import sys
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from psd.algorithms import gradient_descent  # noqa: E402


@given(
    st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=100)
def test_gradient_descent_converges_to_zero(x0: list[float]) -> None:
    arr = np.array(x0, dtype=float)

    def grad(x: np.ndarray) -> np.ndarray:
        return x

    x, iters = gradient_descent(arr, grad, step_size=0.5, tol=1e-8, max_iter=1000)
    assert np.linalg.norm(x) <= 1e-6
    assert iters <= 1000
