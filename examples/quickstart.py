"""Minimal example using the PSD reference algorithms."""

from __future__ import annotations

import numpy as np

from psd import algorithms, functions


def main() -> None:
    """Run a single gradient descent step on a toy function."""

    x0 = np.array([1.0, -1.0])
    x_star, _ = algorithms.gradient_descent(
        x0, functions.SEPARABLE_QUARTIC.grad
    )
    print("Optimised point:", x_star)


if __name__ == "__main__":
    main()

