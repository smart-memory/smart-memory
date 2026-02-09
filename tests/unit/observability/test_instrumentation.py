"""Unit tests for observability instrumentation context management."""

import pytest

pytestmark = pytest.mark.unit

from smartmemory.observability.instrumentation import (
    set_obs_context,
    update_obs_context,
    get_obs_context,
    clear_obs_context,
    with_obs_context,
    make_emitter,
    emit_ctx,
)


@pytest.fixture(autouse=True)
def _clean_context():
    clear_obs_context()
    yield
    clear_obs_context()


class TestObsContext:
    def test_default_empty(self):
        assert get_obs_context() == {}

    def test_set_and_get(self):
        set_obs_context({"run_id": "r1", "stage": "extract"})
        ctx = get_obs_context()
        assert ctx["run_id"] == "r1"
        assert ctx["stage"] == "extract"

    def test_set_replaces(self):
        set_obs_context({"a": 1})
        set_obs_context({"b": 2})
        ctx = get_obs_context()
        assert "a" not in ctx
        assert ctx["b"] == 2

    def test_update_merges(self):
        set_obs_context({"a": 1})
        update_obs_context({"b": 2})
        ctx = get_obs_context()
        assert ctx["a"] == 1
        assert ctx["b"] == 2

    def test_update_overwrites_key(self):
        set_obs_context({"a": 1})
        update_obs_context({"a": 99})
        assert get_obs_context()["a"] == 99

    def test_clear(self):
        set_obs_context({"a": 1})
        clear_obs_context()
        assert get_obs_context() == {}

    def test_get_returns_copy(self):
        set_obs_context({"a": 1})
        ctx = get_obs_context()
        ctx["a"] = 999
        assert get_obs_context()["a"] == 1

    def test_set_ignores_non_dict(self):
        set_obs_context({"a": 1})
        set_obs_context("not a dict")
        assert get_obs_context() == {"a": 1}

    def test_update_ignores_non_dict(self):
        set_obs_context({"a": 1})
        update_obs_context("not a dict")
        assert get_obs_context() == {"a": 1}


class TestWithObsContext:
    def test_decorator_with_dict(self):
        @with_obs_context({"trace_id": "t1"})
        def my_fn():
            return get_obs_context()

        result = my_fn()
        assert result["trace_id"] == "t1"
        # Context restored after call
        assert get_obs_context() == {}

    def test_decorator_with_callable(self):
        @with_obs_context(lambda *a, **kw: {"dynamic": "yes"})
        def my_fn():
            return get_obs_context()

        result = my_fn()
        assert result["dynamic"] == "yes"

    def test_decorator_merge_true(self):
        set_obs_context({"existing": "val"})

        @with_obs_context({"new": "val2"}, merge=True)
        def my_fn():
            return get_obs_context()

        result = my_fn()
        assert result["existing"] == "val"
        assert result["new"] == "val2"

    def test_decorator_merge_false(self):
        set_obs_context({"existing": "val"})

        @with_obs_context({"new": "val2"}, merge=False)
        def my_fn():
            return get_obs_context()

        result = my_fn()
        assert "existing" not in result
        assert result["new"] == "val2"

    def test_restores_context_on_exception(self):
        set_obs_context({"before": True})

        @with_obs_context({"during": True})
        def failing_fn():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            failing_fn()

        ctx = get_obs_context()
        assert ctx["before"] is True
        assert "during" not in ctx

    def test_decorator_with_none(self):
        @with_obs_context(None)
        def my_fn():
            return get_obs_context()

        result = my_fn()
        assert result == {}


class TestMakeEmitter:
    def test_creates_callable(self):
        emitter = make_emitter(component="test")
        assert callable(emitter)

    def test_emitter_does_not_raise(self):
        emitter = make_emitter(component="test", default_type="test_event")
        # Should not raise even though observability is disabled
        emitter("test_event", "op", {"key": "val"})


class TestEmitCtx:
    def test_does_not_raise_when_disabled(self):
        # Observability is disabled by default in tests
        emit_ctx("test_event", component="test", operation="op", data={"k": "v"})

    def test_includes_context_in_payload(self):
        set_obs_context({"run_id": "r1"})
        # We can't easily verify the payload without Redis, but we can verify
        # the function doesn't crash
        emit_ctx("test", component="c", operation="o", data={"x": 1})
