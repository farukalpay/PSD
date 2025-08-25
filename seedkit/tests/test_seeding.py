import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from seedkit.seeding import make_subseed, philox_generator  # noqa: E402


def test_subseed_determinism():
    s1 = make_subseed(123, "component", "run", 0)
    s2 = make_subseed(123, "component", "run", 0)
    assert s1 == s2

    s3 = make_subseed(123, "other", "run", 0)
    assert s1 != s3

    s4 = make_subseed(123, "component", "run", 1)
    assert s1 != s4


def test_stream_independence():
    sub0 = make_subseed(123, "component", "run", 0)
    sub1 = make_subseed(123, "component", "run", 1)
    g0 = philox_generator(sub0)
    g1 = philox_generator(sub1)
    x = g0.random(1000)
    y = g1.random(1000)
    r = np.corrcoef(x, y)[0, 1]
    assert abs(r) < 0.1
