#!/usr/bin/env python3
"""Utility script to analyse PSD log files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def analyse(path: Path) -> None:
    levels: Counter[str] = Counter()
    if not path.is_file():
        print(f"No log file found at {path}")
        return
    with path.open() as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            levels[record.get("level", "UNKNOWN")] += 1
    for level, count in sorted(levels.items()):
        print(f"{level}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse structured PSD logs")
    parser.add_argument("logfile", nargs="?", default="logs/psd.log", type=Path)
    args = parser.parse_args()
    analyse(args.logfile)


if __name__ == "__main__":
    main()
