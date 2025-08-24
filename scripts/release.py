#!/usr/bin/env python3
"""Simple semantic-versioning release utility.

The script bumps the version in ``pyproject.toml``, appends entries to
``CHANGELOG.md`` based on recent git commits and creates a tagged commit.
Use ``--rollback`` to delete the most recent tag if a release goes wrong."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

VERSION_FILE = Path("pyproject.toml")
CHANGELOG = Path("CHANGELOG.md")


def current_version() -> str:
    for line in VERSION_FILE.read_text().splitlines():
        if line.startswith("version ="):
            return line.split("=")[1].strip().strip('"')
    return "0.0.0"


def update_version(version: str) -> None:
    lines: list[str] = []
    for line in VERSION_FILE.read_text().splitlines():
        if line.startswith("version ="):
            lines.append(f'version = "{version}"')
        else:
            lines.append(line)
    VERSION_FILE.write_text("\n".join(lines) + "\n")


def generate_changelog(previous: str, new: str) -> None:
    log = subprocess.check_output(
        ["git", "log", "--oneline", f"v{previous}..HEAD"], text=True
    )
    with CHANGELOG.open("a") as fh:
        fh.write(f"\n## v{new}\n\n{log}\n")


def bump(bump_type: str) -> str:
    major, minor, patch = map(int, current_version().split("."))
    if bump_type == "major":
        major += 1
        minor = patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Release helper")
    parser.add_argument("bump", choices=["major", "minor", "patch"])
    parser.add_argument(
        "--rollback", action="store_true", help="Rollback the most recent tag"
    )
    args = parser.parse_args()

    if args.rollback:
        prev = current_version()
        subprocess.check_call(["git", "tag", "-d", f"v{prev}"])
        return

    new_version = bump(args.bump)
    prev_version = current_version()
    update_version(new_version)
    generate_changelog(prev_version, new_version)
    subprocess.check_call(["git", "commit", "-am", f"chore(release): v{new_version}"])
    subprocess.check_call(["git", "tag", f"v{new_version}"])


if __name__ == "__main__":
    main()
