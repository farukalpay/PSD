import json
from pathlib import Path

import numpy as np
import tracemalloc

from psd.functions import rosenbrock_hess

BASELINE = Path(__file__).with_name("rosenbrock_hess_baseline.json")
TOL = 0.05


def test_rosenbrock_hess_speed(benchmark):
    x = np.random.rand(1000)
    benchmark(rosenbrock_hess, x)
    mean = benchmark.stats["mean"]
    tracemalloc.start()
    rosenbrock_hess(x)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    data = {"mean": mean, "peak": peak}
    if BASELINE.exists():
        baseline = json.loads(BASELINE.read_text())
        assert mean <= baseline["mean"] * (1 + TOL)
        assert mean >= baseline["mean"] * (1 - TOL)
        assert peak <= baseline["peak"] * (1 + TOL)
    else:
        BASELINE.write_text(json.dumps(data))
