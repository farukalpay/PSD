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
