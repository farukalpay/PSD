from __future__ import annotations

import json
from pathlib import Path

import pytest

from psd.log_analyzer import analyze_log


def test_analyze_log(tmp_path: Path) -> None:
    log_file = tmp_path / "psd.log"
    records = [
        {"time": "0", "level": "INFO", "name": "t", "message": "ok", "duration": 0.1},
        {"time": "1", "level": "ERROR", "name": "t", "message": "fail"},
        {"time": "2", "level": "INFO", "name": "t", "message": "ok", "duration": 0.3},
    ]
    log_file.write_text("\n".join(json.dumps(r) for r in records))
    stats = analyze_log(log_file)
    assert stats.error_count == 1
    assert stats.latency_count == 2
    assert stats.avg_latency == pytest.approx(0.2)


def test_analyze_log_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure transient ``OSError`` during file access are retried."""

    log_file = tmp_path / "psd.log"
    log_file.write_text("{}\n")

    calls = {"count": 0}
    path_cls = type(log_file)
    real_open = path_cls.open

    def flaky_open(self: Path, *args, **kwargs):
        if self == log_file and calls["count"] == 0:
            calls["count"] += 1
            raise OSError("transient")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(path_cls, "open", flaky_open)
    stats = analyze_log(log_file)
    assert stats.error_count == 0
    assert calls["count"] == 1
