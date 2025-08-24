"""Pytest configuration for deterministic test runs.

This file seeds all common sources of randomness before each test so that
results are reproducible across runs and on CI.  Hypothesis based property
tests are additionally configured with ``derandomize=True`` in the individual
modules.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch might not be installed
    torch = None  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Seed RNGs for ``random``, ``numpy`` and ``torch`` if available.

    Seeding happens automatically for every test via the ``autouse`` fixture
    mechanism.  ``PYTHONHASHSEED`` is also set to ensure deterministic hashing
    behaviour in dictionaries and other hash based collections.
    """

    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(0)
    np.random.seed(0)
    if torch is not None:
        torch.manual_seed(0)
