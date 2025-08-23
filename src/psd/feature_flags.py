from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class FeatureFlags:
    """Runtime feature flags controlling experimental behaviour."""

    new_escape_condition: bool = False

    @classmethod
    def from_env(cls) -> FeatureFlags:
        """Create flags based on environment variables."""

        def _env_true(name: str, default: bool = False) -> bool:
            return os.getenv(name, str(default)).lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

        return cls(new_escape_condition=_env_true("PSD_NEW_ESCAPE_CONDITION"))


FLAGS = FeatureFlags.from_env()


def enable(flag: str) -> None:
    """Enable a feature flag by name."""
    if not hasattr(FLAGS, flag):  # pragma: no cover - validation
        raise AttributeError(f"Unknown feature flag: {flag}")
    setattr(FLAGS, flag, True)


def disable(flag: str) -> None:
    """Disable a feature flag by name."""
    if not hasattr(FLAGS, flag):  # pragma: no cover - validation
        raise AttributeError(f"Unknown feature flag: {flag}")
    setattr(FLAGS, flag, False)


__all__ = ["FeatureFlags", "FLAGS", "enable", "disable"]
