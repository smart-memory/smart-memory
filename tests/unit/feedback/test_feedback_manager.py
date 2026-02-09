"""Unit tests for FeedbackManager."""

import pytest

pytestmark = pytest.mark.unit

from smartmemory.feedback.manager import FeedbackManager


@pytest.fixture
def manager():
    return FeedbackManager()


class TestRegisterChannel:
    def test_register_and_retrieve(self, manager):
        fn = lambda *a, **kw: "ok"
        manager.register_channel("cli", fn)
        assert manager.get_channel("cli") is fn

    def test_register_multiple_channels(self, manager):
        fn1 = lambda: "cli"
        fn2 = lambda: "slack"
        manager.register_channel("cli", fn1)
        manager.register_channel("slack", fn2)
        assert manager.get_channel("cli") is fn1
        assert manager.get_channel("slack") is fn2

    def test_overwrite_channel(self, manager):
        fn1 = lambda: "old"
        fn2 = lambda: "new"
        manager.register_channel("cli", fn1)
        manager.register_channel("cli", fn2)
        assert manager.get_channel("cli") is fn2


class TestGetChannel:
    def test_returns_none_for_unregistered(self, manager):
        assert manager.get_channel("nonexistent") is None

    def test_returns_callable(self, manager):
        fn = lambda step, ctx, res: "feedback"
        manager.register_channel("test", fn)
        retrieved = manager.get_channel("test")
        assert callable(retrieved)


class TestRequestFeedback:
    def test_calls_registered_channel(self, manager):
        calls = []
        def mock_fn(step, context):
            calls.append((step, context))
            return "good"

        manager.register_channel("cli", mock_fn)
        result = manager.request_feedback("cli", 1, {"query": "test"})
        assert result == "good"
        assert calls == [(1, {"query": "test"})]

    def test_raises_on_unregistered_channel(self, manager):
        with pytest.raises(ValueError, match="Feedback channel 'missing' not registered"):
            manager.request_feedback("missing", 1, {})

    def test_forwards_kwargs(self, manager):
        def mock_fn(*args, **kwargs):
            return kwargs

        manager.register_channel("test", mock_fn)
        result = manager.request_feedback("test", verbose=True, limit=5)
        assert result == {"verbose": True, "limit": 5}

    def test_channel_exception_propagates(self, manager):
        def failing_fn(*args, **kwargs):
            raise RuntimeError("Channel broken")

        manager.register_channel("broken", failing_fn)
        with pytest.raises(RuntimeError, match="Channel broken"):
            manager.request_feedback("broken")
