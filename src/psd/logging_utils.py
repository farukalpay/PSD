from __future__ import annotations

import json
import logging
import logging.config
from logging.handlers import RotatingFileHandler
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
        # Include extra attributes added via the ``extra`` argument
        reserved = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }
        for key, value in record.__dict__.items():
            if key in reserved or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                data[key] = value
            except TypeError:  # pragma: no cover - non-serialisable extras
                data[key] = str(value)
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data)


def setup_logging(config_path: str | Path | None = None) -> None:
    """Initialise structured logging with optional configuration file.

    If ``config_path`` (or ``logging.ini``) is present, it will be loaded via
    :func:`logging.config.fileConfig`.  Otherwise a sensible default
    configuration with a rotating JSON log file and console output is
    installed.
    """

    path = Path(config_path or "logging.ini")
    if path.is_file():
        Path("logs").mkdir(exist_ok=True)
        logging.config.fileConfig(path, disable_existing_loggers=False)
        return

    root = logging.getLogger()
    if root.handlers:  # pragma: no cover - defensive programming
        return
    root.setLevel(logging.INFO)
    Path("logs").mkdir(exist_ok=True)

    formatter = JsonFormatter()

    file_handler = RotatingFileHandler("logs/psd.log", maxBytes=1_048_576, backupCount=3)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
