#!/usr/bin/env python3
"""Release helper implementing conventional commits and semantic versioning.

This utility inspects the commit history since the previous ``vX.Y.Z`` tag,
derives the next semantic version according to the `conventional commits`_
specification, updates ``pyproject.toml`` and ``CHANGELOG.md`` accordingly and
creates a tagged commit.

Use ``--rollback`` to delete the most recent tag if a release goes wrong.

.. _conventional commits: https://www.conventionalcommits.org/en/v1.0.0/
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

VERSION_FILE = Path("pyproject.toml")
CHANGELOG = Path("CHANGELOG.md")


COMMIT_RE = re.compile(r"(?P<type>\w+)(?P<breaking>!)?(?:\([^)]*\))?: (?P<desc>.+)")

SECTION_TITLES = {
    "feat": "Features",
    "fix": "Bug Fixes",
    "docs": "Documentation",
    "refactor": "Refactoring",
    "perf": "Performance",
    "test": "Tests",
    "chore": "Chores",
}


@dataclass
class Commit:
    type: str
    description: str
    hash: str
    breaking: bool = False


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


def last_tag() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "describe", "--tags", "--abbrev=0"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except subprocess.CalledProcessError:
        return None


def parse_commits(since: str | None) -> list[Commit]:
    rev_range = f"{since}..HEAD" if since else "HEAD"
    log = subprocess.check_output(
        ["git", "log", rev_range, "--pretty=format:%s:::%h"], text=True
    )
    commits: list[Commit] = []
    for line in log.splitlines():
        if not line.strip():
            continue
        subject, sha = line.split(":::")
        match = COMMIT_RE.match(subject)
        if match:
            commits.append(
                Commit(
                    type=match.group("type"),
                    description=match.group("desc"),
                    hash=sha,
                    breaking=bool(match.group("breaking")),
                )
            )
        else:
            commits.append(Commit(type="other", description=subject, hash=sha))
    return commits


def bump_level(commits: Iterable[Commit]) -> str:
    if any(c.breaking for c in commits):
        return "major"
    if any(c.type == "feat" for c in commits):
        return "minor"
    return "patch"


def increment(version: str, level: str) -> str:
    major, minor, patch = map(int, version.split("."))
    if level == "major":
        major += 1
        minor = patch = 0
    elif level == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


def generate_changelog(version: str, commits: list[Commit]) -> None:
    date = dt.date.today().isoformat()
    sections: dict[str, list[str]] = {k: [] for k in SECTION_TITLES}
    misc: list[str] = []
    for commit in commits:
        line = f"- {commit.description} ({commit.hash})"
        if commit.type in sections:
            sections[commit.type].append(line)
        else:
            misc.append(line)

    lines: list[str] = [f"## v{version} - {date}"]
    for key, title in SECTION_TITLES.items():
        entries = sections[key]
        if entries:
            lines.append(f"### {title}")
            lines.extend(entries)
            lines.append("")
    if misc:
        lines.append("### Miscellaneous")
        lines.extend(misc)
        lines.append("")

    changelog_lines = CHANGELOG.read_text().splitlines()
    try:
        insert_at = changelog_lines.index("## Unreleased") + 1
    except ValueError:
        insert_at = 1
    changelog_lines[insert_at:insert_at] = ["", *lines]
    CHANGELOG.write_text("\n".join(changelog_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a release from conventional commit messages."
    )
    parser.add_argument(
        "--rollback", action="store_true", help="Delete the most recent tag"
    )
    args = parser.parse_args()

    if args.rollback:
        tag = last_tag()
        if tag:
            subprocess.check_call(["git", "tag", "-d", tag])
        return

    prev_tag = last_tag()
    commits = parse_commits(prev_tag)
    if not commits:
        print("No commits to release.")
        return

    level = bump_level(commits)
    new_version = increment(current_version(), level)
    update_version(new_version)
    generate_changelog(new_version, commits)
    subprocess.check_call(
        ["git", "commit", "-am", f"chore(release): v{new_version}"]
    )
    subprocess.check_call(
        ["git", "tag", "-a", f"v{new_version}", "-m", f"v{new_version}"]
    )


if __name__ == "__main__":
    main()

