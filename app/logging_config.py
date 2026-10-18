"""Structured (JSON-lines) logging."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger("rag")
    root.setLevel(level)
    if not any(getattr(h, "_rag_handler", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(JsonFormatter())
        h._rag_handler = True  # type: ignore[attr-defined]
        root.addHandler(h)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"rag.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, event, extra={"event": event, **fields})
