"""Tests for EventSpooler and module-level constants."""

import pytest

pytestmark = pytest.mark.unit


import uuid
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_config():
    """Return a MagicMock that satisfies get_config() contract."""
    config = MagicMock()
    config.cache.redis.host = "localhost"
    config.cache.redis.port = 6379
    config.get.return_value = None  # no observability or namespace overrides
    return config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


@patch("smartmemory.observability.events.get_config", return_value=_make_mock_config())
@patch("redis.Redis")
def test_stream_name_constant(mock_redis_cls, mock_get_config):
    """STREAM_NAME equals 'smartmemory:events'."""
    from smartmemory.observability.events import STREAM_NAME

    assert STREAM_NAME == "smartmemory:events"


@patch("smartmemory.observability.events.get_config", return_value=_make_mock_config())
@patch("redis.Redis")
def test_redis_db_events_constant(mock_redis_cls, mock_get_config):
    """REDIS_DB_EVENTS equals 1."""
    from smartmemory.observability.events import REDIS_DB_EVENTS

    assert REDIS_DB_EVENTS == 1


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


@patch("smartmemory.observability.events.get_config", return_value=_make_mock_config())
@patch("redis.Redis")
def test_constructor_auto_generates_session_id(mock_redis_cls, mock_get_config):
    """Constructor generates a valid UUID4 session_id when none is provided."""
    from smartmemory.observability.events import EventSpooler

    spooler = EventSpooler()
    # Should be a valid UUID
    parsed = uuid.UUID(spooler.session_id, version=4)
    assert str(parsed) == spooler.session_id


@patch("smartmemory.observability.events.get_config", return_value=_make_mock_config())
@patch("redis.Redis")
def test_constructor_preserves_custom_session_id(mock_redis_cls, mock_get_config):
    """Constructor keeps the caller-supplied session_id unchanged."""
    from smartmemory.observability.events import EventSpooler

    custom_id = "my-custom-session-42"
    spooler = EventSpooler(session_id=custom_id)
    assert spooler.session_id == custom_id


@patch("smartmemory.observability.events.get_config", return_value=_make_mock_config())
@patch("redis.Redis")
def test_constructor_default_stream_name(mock_redis_cls, mock_get_config):
    """Constructor defaults stream_name to STREAM_NAME when no namespace is active."""
    from smartmemory.observability.events import EventSpooler, STREAM_NAME

    spooler = EventSpooler()
    assert spooler.stream_name == STREAM_NAME


@patch("smartmemory.observability.events.get_config")
@patch("redis.Redis")
def test_constructor_appends_namespace_to_stream(mock_redis_cls, mock_get_config):
    """Constructor appends active_namespace to stream_name when present."""
    config = _make_mock_config()

    def _side_effect(key, *args, **kwargs):
        if key == "active_namespace":
            return "prod"
        if key == "observability":
            return None
        return None

    config.get.side_effect = _side_effect
    mock_get_config.return_value = config

    from smartmemory.observability.events import EventSpooler

    spooler = EventSpooler()
    assert spooler.stream_name == "smartmemory:events:prod"


# ---------------------------------------------------------------------------
# emit_event
# ---------------------------------------------------------------------------


@patch("smartmemory.observability.events.get_config", return_value=_make_mock_config())
@patch("redis.Redis")
def test_emit_event_calls_xadd_when_enabled(mock_redis_cls, mock_get_config):
    """emit_event calls redis xadd when obs_enabled is True."""
    from smartmemory.observability.events import EventSpooler

    mock_client = MagicMock()
    mock_redis_cls.return_value = mock_client

    spooler = EventSpooler()
    # obs_enabled defaults to True (config.get("observability") returns None -> {}.get("enabled", True) -> True)
    assert spooler.obs_enabled is True

    spooler.emit_event("test_event", "test_component", "test_op", {"key": "val"})
    mock_client.xadd.assert_called_once()

    # Verify the event_type is in the payload
    call_args = mock_client.xadd.call_args
    event_data = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("fields", {})
    assert event_data["event_type"] == "test_event"
    assert event_data["component"] == "test_component"
    assert event_data["operation"] == "test_op"


@patch("smartmemory.observability.events.get_config")
@patch("redis.Redis")
def test_emit_event_noop_when_disabled(mock_redis_cls, mock_get_config):
    """emit_event is a no-op (no xadd call) when obs_enabled is False."""
    config = _make_mock_config()

    def _side_effect(key, *args, **kwargs):
        if key == "observability":
            return {"enabled": False}
        return None

    config.get.side_effect = _side_effect
    mock_get_config.return_value = config

    from smartmemory.observability.events import EventSpooler

    mock_client = MagicMock()
    mock_redis_cls.return_value = mock_client

    spooler = EventSpooler()
    assert spooler.obs_enabled is False

    spooler.emit_event("test_event", "comp", "op", {"k": "v"})
    mock_client.xadd.assert_not_called()


@patch("smartmemory.observability.events.get_config", return_value=_make_mock_config())
@patch("redis.Redis")
def test_emit_event_includes_session_id(mock_redis_cls, mock_get_config):
    """emit_event includes the spooler's session_id in the event payload."""
    from smartmemory.observability.events import EventSpooler

    mock_client = MagicMock()
    mock_redis_cls.return_value = mock_client

    spooler = EventSpooler(session_id="sess-abc")
    spooler.emit_event("ping", "health", "check")

    call_args = mock_client.xadd.call_args
    event_data = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("fields", {})
    assert event_data["session_id"] == "sess-abc"


@patch("smartmemory.observability.events.get_config", return_value=_make_mock_config())
@patch("redis.Redis")
def test_emit_event_applies_maxlen(mock_redis_cls, mock_get_config):
    """emit_event passes maxlen and approximate=True to xadd when maxlen is set."""
    from smartmemory.observability.events import EventSpooler

    mock_client = MagicMock()
    mock_redis_cls.return_value = mock_client

    spooler = EventSpooler()
    # Default maxlen is 100_000
    assert spooler.maxlen == 100_000

    spooler.emit_event("evt", "comp", "op")

    call_kwargs = mock_client.xadd.call_args[1]
    assert call_kwargs["maxlen"] == 100_000
    assert call_kwargs["approximate"] is True
