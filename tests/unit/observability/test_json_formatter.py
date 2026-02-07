"""Tests for JsonFormatter."""

import json
import logging

from smartmemory.observability.json_formatter import JsonFormatter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    msg: str = "hello",
    level: int = logging.INFO,
    name: str = "test.logger",
    exc_info: tuple | None = None,
) -> logging.LogRecord:
    """Create a minimal LogRecord for testing."""
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname="test.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )
    return record


# ---------------------------------------------------------------------------
# format() output
# ---------------------------------------------------------------------------


def test_format_returns_valid_json():
    """format() returns a string that is valid JSON."""
    formatter = JsonFormatter()
    record = _make_record()
    result = formatter.format(record)
    parsed = json.loads(result)
    assert isinstance(parsed, dict)


def test_format_includes_required_keys():
    """format() output includes ts, level, logger, message."""
    formatter = JsonFormatter()
    record = _make_record(msg="test message", level=logging.WARNING, name="my.logger")
    result = json.loads(formatter.format(record))

    assert "ts" in result
    assert result["level"] == "WARNING"
    assert result["logger"] == "my.logger"
    assert result["message"] == "test message"


def test_format_includes_context_when_present():
    """format() includes context dict when record.context is set."""
    formatter = JsonFormatter()
    record = _make_record()
    record.context = {"run_id": "abc-123", "stage": "classify"}

    result = json.loads(formatter.format(record))
    assert "context" in result
    assert result["context"]["run_id"] == "abc-123"
    assert result["context"]["stage"] == "classify"


def test_format_omits_context_when_empty():
    """format() does not include context key when record.context is empty."""
    formatter = JsonFormatter()
    record = _make_record()
    record.context = {}

    result = json.loads(formatter.format(record))
    assert "context" not in result


def test_format_omits_context_when_absent():
    """format() does not include context key when record has no context attr."""
    formatter = JsonFormatter()
    record = _make_record()
    # LogRecord does not get a context attribute by default

    result = json.loads(formatter.format(record))
    assert "context" not in result


def test_format_includes_exc_info():
    """format() includes exc_info when an exception is present."""
    formatter = JsonFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = _make_record(exc_info=exc_info)
    result = json.loads(formatter.format(record))

    assert "exc_info" in result
    assert "ValueError" in result["exc_info"]
    assert "test error" in result["exc_info"]


def test_format_no_exc_info_when_no_exception():
    """format() omits exc_info when there is no exception."""
    formatter = JsonFormatter()
    record = _make_record()

    result = json.loads(formatter.format(record))
    assert "exc_info" not in result


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_constructor_default_level():
    """Constructor stores the default_level parameter."""
    formatter = JsonFormatter(default_level="DEBUG")
    assert formatter.default_level == "DEBUG"


def test_constructor_default_level_fallback():
    """Constructor defaults to 'INFO' when no default_level is given."""
    formatter = JsonFormatter()
    assert formatter.default_level == "INFO"
