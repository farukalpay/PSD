from __future__ import annotations

import json
import logging
import logging.config
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """Format log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial
        data = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data)


def setup_logging(config_path: str | Path | None = None) -> None:
    """Load logging configuration from ``logging.ini`` if present."""

    path = Path(config_path or "logging.ini")
    if not path.is_file():
        return
    Path("logs").mkdir(exist_ok=True)
    logging.config.fileConfig(path, disable_existing_loggers=False)
