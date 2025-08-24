from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .utils import retry


@dataclass
class LogStats:
    """Summary statistics for a log file."""

    error_count: int
    avg_latency: float
    latency_count: int


@retry(OSError, tries=3, delay=0.01)
def analyze_log(path: str | Path) -> LogStats:
    """Summarise errors and latency metrics in a structured log file.

    Parameters
    ----------
    path:
        Path to a log file containing one JSON object per line.  The function
        tolerates transient ``OSError`` issues (for example when a log file is
        being rotated) by retrying with a short exponential backoff.
    """

    p = Path(path)
    errors = 0
    latencies: list[float] = []
    if not p.is_file():
        return LogStats(0, 0.0, 0)
    with p.open() as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("level") == "ERROR":
                errors += 1
            if "duration" in record:
                try:
                    latencies.append(float(record["duration"]))
                except (TypeError, ValueError):
                    continue
    avg = sum(latencies) / len(latencies) if latencies else 0.0
    return LogStats(errors, avg, len(latencies))


def summarize_logs(directory: str | Path = "logs") -> dict[str, LogStats]:
    """Summarise all log files in ``directory``.

    Returns a mapping of filename to :class:`LogStats`.
    """

    dir_path = Path(directory)
    summaries: dict[str, LogStats] = {}
    for file in dir_path.glob("*.log*"):
        summaries[file.name] = analyze_log(file)
    return summaries


__all__ = ["LogStats", "analyze_log", "summarize_logs"]
