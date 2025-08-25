"""Console interface for seedkit."""
from __future__ import annotations

import argparse
from typing import Sequence, Optional

from .seeding import (
    make_subseed,
    philox_generator,
    python_random,
    torch_generator,
)


def _demo(args: argparse.Namespace) -> None:
    subseed = make_subseed(args.master, args.component, args.run, args.stream)
    np_gen = philox_generator(subseed)
    py_gen = python_random(subseed)

    print(f"Derived subseed: {subseed}")
    print("NumPy samples:", np_gen.random(args.n).tolist())
    print("Python samples:", [py_gen.random() for _ in range(args.n)])

    try:
        torch_gen = torch_generator(subseed=subseed)
        import torch

        print("Torch samples:", torch.rand(args.n, generator=torch_gen).tolist())
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seedkit")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Showcase deterministic seeding")
    demo.add_argument("--master", required=True, help="Master seed")
    demo.add_argument("--component", required=True, help="Component identifier")
    demo.add_argument("--run", required=True, help="Run identifier")
    demo.add_argument("--stream", type=int, default=0, help="Stream identifier")
    demo.add_argument("--n", type=int, default=5, help="Number of samples to draw")
    demo.set_defaults(func=_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
