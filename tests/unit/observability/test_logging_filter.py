"""Tests for LogContextFilter."""

import pytest

pytestmark = pytest.mark.unit


import logging
from unittest.mock import patch

from smartmemory.observability.logging_filter import LogContextFilter, _ALLOWED_KEYS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(msg: str = "test") -> logging.LogRecord:
    """Create a minimal LogRecord for testing."""
    return logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


# ---------------------------------------------------------------------------
# filter() behavior
# ---------------------------------------------------------------------------


@patch("smartmemory.observability.logging_filter.get_obs_context", return_value={})
def test_filter_returns_true(mock_ctx):
    """filter() always returns True so all records pass through."""
    f = LogContextFilter()
    record = _make_record()
    assert f.filter(record) is True


@patch("smartmemory.observability.logging_filter.get_obs_context", return_value={})
def test_filter_adds_context_attribute(mock_ctx):
    """filter() attaches a context dict to the LogRecord."""
    f = LogContextFilter()
    record = _make_record()
    f.filter(record)
    assert hasattr(record, "context")
    assert isinstance(record.context, dict)


@patch(
    "smartmemory.observability.logging_filter.get_obs_context",
    return_value={"run_id": "abc-123", "pipeline_id": "pipe-1", "stage": "classify"},
)
def test_filter_includes_allowed_keys(mock_ctx):
    """filter() copies whitelisted keys from the observability context."""
    f = LogContextFilter()
    record = _make_record()
    f.filter(record)

    assert record.context["run_id"] == "abc-123"
    assert record.context["pipeline_id"] == "pipe-1"
    assert record.context["stage"] == "classify"


@patch(
    "smartmemory.observability.logging_filter.get_obs_context",
    return_value={"run_id": "r1", "secret_internal": "do-not-leak"},
)
def test_filter_excludes_non_allowed_keys(mock_ctx):
    """filter() excludes keys not in _ALLOWED_KEYS."""
    f = LogContextFilter()
    record = _make_record()
    f.filter(record)

    assert "run_id" in record.context
    assert "secret_internal" not in record.context


@patch(
    "smartmemory.observability.logging_filter.get_obs_context",
    return_value={"run_id": "r1", "api_key": "sk-secret"},
)
def test_filter_redacts_sensitive_keys(mock_ctx):
    """filter() redacts values whose key matches a sensitive hint, if the key is also allowed."""
    f = LogContextFilter()
    record = _make_record()
    f.filter(record)

    # api_key is NOT in _ALLOWED_KEYS, so it should be excluded entirely
    assert "api_key" not in record.context


# ---------------------------------------------------------------------------
# _ALLOWED_KEYS whitelist
# ---------------------------------------------------------------------------


def test_allowed_keys_contains_expected_entries():
    """_ALLOWED_KEYS contains the standard observability identifiers."""
    expected = {
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
    assert expected == _ALLOWED_KEYS


@patch(
    "smartmemory.observability.logging_filter.get_obs_context",
    return_value={
        "run_id": "r",
        "pipeline_id": "p",
        "stage": "s",
        "stage_id": "sid",
        "ingestion_id": "i",
        "request_id": "req",
        "change_set_id": "cs",
        "user_id": "u",
        "session_id": "sess",
        "component": "c",
        "env": "dev",
        "service": "api",
        "version": "0.3.1",
    },
)
def test_all_allowed_keys_pass_through(mock_ctx):
    """Every key in _ALLOWED_KEYS is included in the filtered context."""
    f = LogContextFilter()
    record = _make_record()
    f.filter(record)

    for key in _ALLOWED_KEYS:
        assert key in record.context, f"Expected '{key}' in context"


@patch(
    "smartmemory.observability.logging_filter.get_obs_context",
    side_effect=RuntimeError("broken"),
)
def test_filter_returns_true_on_exception(mock_ctx):
    """filter() returns True even when get_obs_context raises."""
    f = LogContextFilter()
    record = _make_record()
    assert f.filter(record) is True
    assert record.context == {}
