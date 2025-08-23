from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeatureFlags:
    """Runtime feature flags controlling experimental behaviour."""

    new_escape_condition: bool = False


FLAGS = FeatureFlags()


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
