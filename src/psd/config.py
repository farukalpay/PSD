from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PSDConfig:
    """Configuration for the :func:`psd.algorithms.psd` routine.

    The class validates basic constraints on the parameters to avoid
    silent misuse.  It is intentionally minimal and only captures the
    parameters used by the reference implementation.
    """

    epsilon: float
    ell: float
    rho: float
    delta: float = 0.1
    delta_f: float = 1.0
    max_iter: int = 100000

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.ell <= 0:
            raise ValueError("ell must be positive")
        if self.rho < 0:
            raise ValueError("rho must be non-negative")
        if not 0 < self.delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if self.delta_f <= 0:
            raise ValueError("delta_f must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")


__all__ = ["PSDConfig"]
