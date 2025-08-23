import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from psd import algorithms
from psd.config import PSDConfig


def test_env_var_enables_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from psd.feature_flags import FeatureFlags

    monkeypatch.setenv("PSD_NEW_ESCAPE_CONDITION", "1")
    flags = FeatureFlags.from_env()
    assert flags.new_escape_condition is True


def test_new_escape_condition_changes_step_size() -> None:
    import psd.feature_flags as ff

    def grad(x: np.ndarray) -> np.ndarray:
        return x

    def hess(x: np.ndarray) -> np.ndarray:
        return np.eye(len(x))

    x0 = np.array([1.0])
    cfg = PSDConfig(epsilon=1e-12, ell=1.0, rho=1.0, max_iter=1)

    ff.disable("new_escape_condition")
    x, _ = algorithms.psd(x0, grad, hess, 1e-12, 1.0, 1.0, config=cfg)
    assert np.allclose(x, np.array([0.5]))

    ff.enable("new_escape_condition")
    x, _ = algorithms.psd(x0, grad, hess, 1e-12, 1.0, 1.0, config=cfg)
    assert np.allclose(x, np.array([0.0]))
    ff.disable("new_escape_condition")
