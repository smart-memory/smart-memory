"""
JSON logging formatter that includes the injected observability context.

Usage:
    from smartmemory.observability.json_formatter import JsonFormatter
    handler.setFormatter(JsonFormatter())

The filter installed by install_log_context_filter() sets record.context.
This formatter serializes that map alongside standard fields.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    def __init__(self, *, default_level: str = "INFO") -> None:
        super().__init__()
        self.default_level = default_level

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: Dict[str, Any] = {}
        # Timestamp in ISO-like format with milliseconds
        payload["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)) + f".{int(record.msecs):03d}Z"
        payload["level"] = record.levelname or self.default_level
        payload["logger"] = record.name
        payload["message"] = record.getMessage()
        # Context injected by LogContextFilter (safe and redacted)
        ctx = getattr(record, "context", {})
        if isinstance(ctx, dict) and ctx:
            payload["context"] = ctx
        # Exception info
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)
