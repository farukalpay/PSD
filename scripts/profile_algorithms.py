"""Profile core PSD algorithms with cProfile.

This script runs gradient descent on the Rosenbrock function and prints
cProfile statistics sorted by cumulative time.  A ``profile_stats.prof``
file is also written for further inspection with ``snakeviz`` or
``pstats``.
"""

from __future__ import annotations

import cProfile
import pstats
from pathlib import Path
import sys
import numpy as np

# ensure src directory on path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from psd import algorithms, functions


def main() -> None:
    """Run gradient descent on the Rosenbrock function under cProfile."""
    x0 = np.array([-1.2, 1.0])
    profiler = cProfile.Profile()
    profiler.enable()
    algorithms.gradient_descent(
        x0, functions.rosenbrock_grad, step_size=1e-3, max_iter=100_000
    )
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)
    stats.dump_stats("profile_stats.prof")


if __name__ == "__main__":
    main()
