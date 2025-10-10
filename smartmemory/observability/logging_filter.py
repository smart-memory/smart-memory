"""
Logging filter that injects whitelisted observability context into every log record.

Reads context from smartmemory.observability.instrumentation.get_obs_context()
so logs, metrics, and traces share the same identifiers (run_id, pipeline_id, etc.).

Usage:
    from smartmemory.observability.logging_filter import install_log_context_filter
    install_log_context_filter()  # once during app startup

Ensure your formatter includes %(context)s to render the injected context. For JSON
formatters, emit the record.__dict__["context"] map.
"""
from __future__ import annotations

import logging
from typing import Dict, Any

from smartmemory.observability.instrumentation import get_obs_context

# Whitelist of safe keys to include in logs
_ALLOWED_KEYS = {
    "run_id",
    "pipeline_id",
    "stage",
    "stage_id",
    "ingestion_id",
    "request_id",
    "change_set_id",
    "user_id",
    "session_id",
    "component",
    "env",
    "service",
    "version",
}

# Keys or substrings that must be redacted if present
_SENSITIVE_HINTS = {
    "api_key",
    "access_token",
    "authorization",
    "password",
    "secret",
    "token",
}

_MAX_VALUE_LEN = 512  # avoid log bloat


def _redact(key: str, value: Any) -> Any:
    try:
        k = (key or "").lower()
        if any(hint in k for hint in _SENSITIVE_HINTS):
            return "[REDACTED]"
        if isinstance(value, (dict, list)):
            # do not inline large structures in the context
            return "[OMITTED]"
        s = str(value)
        if len(s) > _MAX_VALUE_LEN:
            return s[:_MAX_VALUE_LEN] + "…"
        return value
    except Exception:
        return "[FILTER_ERROR]"


class LogContextFilter(logging.Filter):
    """Inject whitelisted observability context onto LogRecord.context."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        try:
            ctx = get_obs_context() or {}
            if not isinstance(ctx, dict):
                record.context = {}
                return True
            out: Dict[str, Any] = {}
            for k, v in ctx.items():
                if k in _ALLOWED_KEYS:
                    out[k] = _redact(k, v)
            record.context = out
        except Exception:
            # do not fail logging; just omit context
            record.context = {}
        return True


def install_log_context_filter(level: int | None = None) -> None:
    """Install LogContextFilter on the root logger and all existing handlers.

    Optionally set a default level (when provided) on the root logger.
    """
    root = logging.getLogger()
    if level is not None:
        root.setLevel(level)
    # Attach to root so child loggers inherit the filter
    _ensure_filter(root)
    for h in root.handlers:
        _ensure_filter(h)


def _ensure_filter(target: logging.Logger | logging.Handler) -> None:
    try:
        has = any(isinstance(f, LogContextFilter) for f in getattr(target, "filters", []))
        if not has:
            target.addFilter(LogContextFilter())
    except Exception:
        pass
