"""Deterministic seeding utilities.

This module hashes a master seed together with component, run and stream
identifiers to derive 64-bit sub-seeds. The message is encoded as
``"{master_seed}|{component_id}|{run_id}|{stream_id}"`` and hashed with
SHA-256, taking the first eight bytes as a big-endian integer. The resulting
sub-seed can be fed to independent random number generators.

Philox, a counter-based RNG available in NumPy, is chosen because it allows
reproducible, stateless streams that can be advanced independently across
parallel processes.
"""

from __future__ import annotations

import argparse
import hashlib
import random

import numpy as np

try:  # Optional PyTorch integration
    import torch
except Exception:  # pragma: no cover - environment without torch
    torch = None  # type: ignore


def make_subseed(
    master_seed: int | str, component_id: str, run_id: str, stream_id: int | str = 0
) -> int:
    """Derive a deterministic 64-bit sub-seed.

    Args:
        master_seed: Global seed as an integer or string.
        component_id: Identifier for the component (e.g., "dataloader").
        run_id: Identifier for the current experiment/run.
        stream_id: Optional sub-stream identifier.

    Returns:
        The first eight bytes of the SHA-256 digest interpreted as a
        big-endian integer.
    """

    message = f"{master_seed}|{component_id}|{run_id}|{stream_id}"
    digest = hashlib.sha256(message.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def philox_rng(subseed: int) -> np.random.Generator:
    """Create a NumPy Philox generator seeded with ``subseed``."""

    return np.random.Generator(np.random.Philox(subseed))


def python_rng(subseed: int) -> random.Random:
    """Return a ``random.Random`` instance seeded with ``subseed``."""

    rng = random.Random()
    rng.seed(subseed)
    return rng


def torch_rng(subseed: int, device: str | torch.device = "cpu"):
    """Return a torch ``Generator`` seeded with ``subseed``.

    Args:
        subseed: Seed value for the generator.
        device: Torch device string or ``torch.device``. Defaults to ``"cpu"``.

    Raises:
        ImportError: If PyTorch is not installed.
    """

    if torch is None:  # pragma: no cover - only hit when torch missing
        raise ImportError("PyTorch is not installed")

    gen = torch.Generator(device)
    gen.manual_seed(int(subseed))
    return gen


def set_torch_deterministic(enabled: bool = True) -> None:
    """Toggle deterministic algorithms and cuDNN flags in PyTorch.

    Args:
        enabled: Whether to enable deterministic behaviour.

    Raises:
        ImportError: If PyTorch is not installed.
    """

    if torch is None:  # pragma: no cover - only hit when torch missing
        raise ImportError("PyTorch is not installed")

    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled
    torch.use_deterministic_algorithms(enabled)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic seeding utility")
    parser.add_argument("--master-seed", required=True, help="Master seed (int or str)")
    parser.add_argument("--component", required=True, help="Component identifier")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--stream-id", type=int, default=0, help="Stream identifier")
    parser.add_argument("--n", type=int, default=5, help="How many numbers to draw")
    args = parser.parse_args()

    subseed = make_subseed(
        args.master_seed, args.component, args.run_id, args.stream_id
    )
    print(f"Subseed: {subseed}")

    np_rng = philox_rng(subseed)
    print("NumPy Philox:", np_rng.random(args.n))

    py_rng = python_rng(subseed)
    print("Python random:", [py_rng.random() for _ in range(args.n)])

    if torch is not None:
        tgen = torch_rng(subseed)
        print("PyTorch:", torch.rand(args.n, generator=tgen).tolist())
