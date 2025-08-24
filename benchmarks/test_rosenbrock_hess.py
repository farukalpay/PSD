import json
import tracemalloc
from pathlib import Path

import numpy as np

from psd.functions import rosenbrock_hess

BASELINE = Path(__file__).with_name("rosenbrock_hess_baseline.json")
MAX_MEAN = 1e-2  # 10 ms
MAX_PEAK = 9_000_000  # 9 MB


def test_rosenbrock_hess_speed(benchmark):
    rng = np.random.default_rng(0)
    x = rng.random(1000)
    benchmark(rosenbrock_hess, x)
    mean = benchmark.stats["mean"]
    tracemalloc.start()
    rosenbrock_hess(x)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    data = {"mean": mean, "peak": peak}
    if not BASELINE.exists():
        BASELINE.write_text(json.dumps(data))
    assert mean < MAX_MEAN
    assert peak < MAX_PEAK
