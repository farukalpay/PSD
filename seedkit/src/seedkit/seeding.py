"""Deterministic seeding utilities."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np

try:  # Optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - import guarded
    torch = None  # type: ignore


def make_subseed(
    master_seed: int | str,
    component_id: str,
    run_id: str,
    stream_id: int | str = 0,
) -> int:
    """Derive a 64-bit subseed via SHA-256.

    Parameters
    ----------
    master_seed:
        Global experiment seed.
    component_id, run_id, stream_id:
        Identifiers used to create independent streams.

    Returns
    -------
    int
        First eight bytes of the SHA-256 hash interpreted as a big-endian integer.
    """

    parts = [str(master_seed), component_id, run_id, str(stream_id)]
    data = b"\x00".join(p.encode("utf-8") for p in parts)
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], "big")


def philox_generator(subseed: int) -> np.random.Generator:
    """Create a NumPy Generator backed by Philox."""
    bitgen = np.random.Philox(subseed)
    return np.random.Generator(bitgen)


def python_random(subseed: int) -> random.Random:
    """Create a Python ``random.Random`` seeded with ``subseed``."""
    return random.Random(subseed)


def set_torch_deterministic() -> None:
    """Enable deterministic algorithms in PyTorch if available."""
    if torch is None:  # pragma: no cover - optional
        return
    torch.use_deterministic_algorithms(True)


def torch_generator(device: str = "cpu", subseed: int | None = None):
    """Create a ``torch.Generator`` seeded with ``subseed``.

    Parameters
    ----------
    device:
        Device for the generator, e.g. ``"cpu"`` or ``"cuda"``.
    subseed:
        Optional seed for the generator.  If ``None`` an unseeded generator is
        returned.
    """
    if torch is None:  # pragma: no cover - optional
        raise RuntimeError("PyTorch not installed")

    gen = torch.Generator(device=device)
    if subseed is not None:
        gen.manual_seed(subseed)
    set_torch_deterministic()
    return gen


@dataclass
class SeedStreams:
    """Bundle of deterministic generators."""

    numpy: np.random.Generator
    python: random.Random
    torch: torch.Generator | None = None


def seed_streams(
    master_seed: int | str,
    component_id: str,
    run_id: str,
    stream_id: int | str = 0,
) -> SeedStreams:
    """Convenience factory creating ``SeedStreams`` from identifiers."""
    subseed = make_subseed(master_seed, component_id, run_id, stream_id)
    np_gen = philox_generator(subseed)
    py_gen = python_random(subseed)
    try:
        torch_gen = torch_generator(subseed=subseed)
    except Exception:  # pragma: no cover - optional
        torch_gen = None
    return SeedStreams(np_gen, py_gen, torch_gen)
