"""Unit tests for the unified trace_span() observability API."""

import threading
import time
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _enable_observability():
    """Enable observability for all tests in this module.

    Since _is_enabled() now reads from os.environ at call time (DIST-LITE-2),
    we set the env var rather than patching a module-level flag.
    """
    import os
    old = os.environ.get("SMARTMEMORY_OBSERVABILITY")
    os.environ["SMARTMEMORY_OBSERVABILITY"] = "true"
    yield
    if old is None:
        os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
    else:
        os.environ["SMARTMEMORY_OBSERVABILITY"] = old


# ---------------------------------------------------------------------------
# SpanContext dataclass
# ---------------------------------------------------------------------------
class TestSpanContext:
    def test_default_fields(self):
        from smartmemory.observability.tracing import SpanContext

        span = SpanContext()
        assert span.trace_id == ""
        assert span.span_id == ""
        assert span.parent_span_id is None
        assert span.name == ""
        assert span.start_time == 0.0
        assert span.attributes == {}

    def test_custom_fields(self):
        from smartmemory.observability.tracing import SpanContext

        span = SpanContext(
            trace_id="abc",
            span_id="def",
            parent_span_id="ghi",
            name="test.op",
            start_time=1.0,
            attributes={"key": "val"},
        )
        assert span.trace_id == "abc"
        assert span.span_id == "def"
        assert span.parent_span_id == "ghi"
        assert span.name == "test.op"
        assert span.start_time == 1.0
        assert span.attributes == {"key": "val"}

    def test_attributes_default_is_independent(self):
        """Each SpanContext instance gets its own dict."""
        from smartmemory.observability.tracing import SpanContext

        s1 = SpanContext()
        s2 = SpanContext()
        s1.attributes["a"] = 1
        assert "a" not in s2.attributes


# ---------------------------------------------------------------------------
# trace_span() — basic behavior
# ---------------------------------------------------------------------------
class TestTraceSpan:
    def test_root_span_creates_trace_id(self):
        from smartmemory.observability.tracing import trace_span

        with patch("smartmemory.observability.tracing._emit_span"):
            with trace_span("test.op") as span:
                assert span.trace_id != ""
                assert len(span.trace_id) == 16

    def test_unique_trace_ids_per_root(self):
        from smartmemory.observability.tracing import trace_span

        with patch("smartmemory.observability.tracing._emit_span"):
            with trace_span("op1") as s1:
                tid1 = s1.trace_id
            with trace_span("op2") as s2:
                tid2 = s2.trace_id
        assert tid1 != tid2

    def test_child_inherits_trace_id(self):
        from smartmemory.observability.tracing import trace_span

        with patch("smartmemory.observability.tracing._emit_span"):
            with trace_span("parent") as parent:
                with trace_span("child") as child:
                    assert child.trace_id == parent.trace_id
                    assert child.parent_span_id == parent.span_id

    def test_deep_nesting_three_levels(self):
        from smartmemory.observability.tracing import trace_span

        with patch("smartmemory.observability.tracing._emit_span"):
            with trace_span("level0") as s0:
                with trace_span("level1") as s1:
                    with trace_span("level2") as s2:
                        assert s2.trace_id == s0.trace_id
                        assert s2.parent_span_id == s1.span_id
                        assert s1.parent_span_id == s0.span_id
                        assert s0.parent_span_id is None

    def test_sibling_spans_share_parent(self):
        from smartmemory.observability.tracing import trace_span

        with patch("smartmemory.observability.tracing._emit_span"):
            with trace_span("parent") as parent:
                with trace_span("child_a") as a:
                    pass
                with trace_span("child_b") as b:
                    pass
        assert a.parent_span_id == parent.span_id
        assert b.parent_span_id == parent.span_id
        assert a.span_id != b.span_id
        assert a.trace_id == b.trace_id

    def test_span_id_uniqueness(self):
        from smartmemory.observability.tracing import trace_span

        with patch("smartmemory.observability.tracing._emit_span"):
            with trace_span("a") as s1:
                pass
            with trace_span("b") as s2:
                pass
        assert s1.span_id != s2.span_id

    def test_attributes_initial(self):
        from smartmemory.observability.tracing import trace_span

        with patch("smartmemory.observability.tracing._emit_span"):
            with trace_span("op", attributes={"model": "gpt-4"}) as span:
                assert span.attributes["model"] == "gpt-4"

    def test_attributes_mutable(self):
        from smartmemory.observability.tracing import trace_span

        emitted = []
        with patch("smartmemory.observability.tracing._emit_span", side_effect=lambda d: emitted.append(d)):
            with trace_span("op") as span:
                span.attributes["added_later"] = 42

        assert emitted[0]["added_later"] == 42

    def test_duration_ms_calculated(self):
        from smartmemory.observability.tracing import trace_span

        emitted = []
        with patch("smartmemory.observability.tracing._emit_span", side_effect=lambda d: emitted.append(d)):
            with trace_span("op"):
                time.sleep(0.01)  # 10ms

        assert emitted[0]["duration_ms"] >= 5  # allow some slack

    def test_component_derived_from_dot_notation(self):
        from smartmemory.observability.tracing import trace_span

        emitted = []
        with patch("smartmemory.observability.tracing._emit_span", side_effect=lambda d: emitted.append(d)):
            with trace_span("pipeline.classify"):
                pass

        assert emitted[0]["component"] == "pipeline"
        assert emitted[0]["operation"] == "classify"

    def test_component_override(self):
        from smartmemory.observability.tracing import trace_span

        emitted = []
        with patch("smartmemory.observability.tracing._emit_span", side_effect=lambda d: emitted.append(d)):
            with trace_span("pipeline.classify", component="custom"):
                pass

        assert emitted[0]["component"] == "custom"

    def test_emits_on_exit_not_during(self):
        from smartmemory.observability.tracing import trace_span

        emitted = []
        with patch("smartmemory.observability.tracing._emit_span", side_effect=lambda d: emitted.append(d)):
            with trace_span("op"):
                assert len(emitted) == 0  # not yet emitted
            assert len(emitted) == 1  # emitted on exit

    def test_correct_envelope_format(self):
        from smartmemory.observability.tracing import trace_span

        emitted = []
        with patch("smartmemory.observability.tracing._emit_span", side_effect=lambda d: emitted.append(d)):
            with trace_span("graph.add_node", attributes={"count": 3}) as span:
                pass

        evt = emitted[0]
        assert evt["event_type"] == "span"
        assert evt["component"] == "graph"
        assert evt["operation"] == "add_node"
        assert evt["name"] == "graph.add_node"
        assert evt["trace_id"] == span.trace_id
        assert evt["span_id"] == span.span_id
        assert evt["parent_span_id"] is None
        assert isinstance(evt["duration_ms"], float)
        assert evt["count"] == 3

    def test_emits_on_exception_with_error_fields(self):
        from smartmemory.observability.tracing import trace_span

        emitted = []
        with patch("smartmemory.observability.tracing._emit_span", side_effect=lambda d: emitted.append(d)):
            with pytest.raises(ValueError, match="boom"):
                with trace_span("failing.op"):
                    raise ValueError("boom")

        assert len(emitted) == 1
        evt = emitted[0]
        assert evt["error"] == "boom"
        assert evt["status"] == "error"

    def test_parent_restored_after_exception(self):
        from smartmemory.observability.tracing import trace_span, current_span

        with patch("smartmemory.observability.tracing._emit_span"):
            with trace_span("parent") as parent:
                try:
                    with trace_span("child"):
                        raise RuntimeError("fail")
                except RuntimeError:
                    pass
                # parent should be current again
                assert current_span() is parent

    def test_name_without_dot(self):
        """When name has no dot, component = name, operation = name."""
        from smartmemory.observability.tracing import trace_span

        emitted = []
        with patch("smartmemory.observability.tracing._emit_span", side_effect=lambda d: emitted.append(d)):
            with trace_span("search"):
                pass

        assert emitted[0]["component"] == "search"
        assert emitted[0]["operation"] == "search"

    def test_user_attributes_cannot_clobber_structural_fields(self):
        """User-provided attributes must not overwrite trace_id, span_id, etc."""
        from smartmemory.observability.tracing import trace_span

        emitted = []
        with patch("smartmemory.observability.tracing._emit_span", side_effect=lambda d: emitted.append(d)):
            with trace_span(
                "op",
                attributes={
                    "trace_id": "INJECTED",
                    "span_id": "INJECTED",
                    "event_type": "INJECTED",
                    "duration_ms": -999,
                },
            ) as span:
                pass

        evt = emitted[0]
        # Structural fields must win over user attributes
        assert evt["trace_id"] == span.trace_id
        assert evt["span_id"] == span.span_id
        assert evt["event_type"] == "span"
        assert evt["duration_ms"] >= 0


# ---------------------------------------------------------------------------
# current_trace_id() and current_span()
# ---------------------------------------------------------------------------
class TestCurrentTraceId:
    def test_returns_none_outside_span(self):
        from smartmemory.observability.tracing import current_trace_id

        assert current_trace_id() is None

    def test_returns_trace_id_inside_span(self):
        from smartmemory.observability.tracing import trace_span, current_trace_id

        with patch("smartmemory.observability.tracing._emit_span"):
            with trace_span("op") as span:
                assert current_trace_id() == span.trace_id


class TestCurrentSpan:
    def test_returns_none_outside(self):
        from smartmemory.observability.tracing import current_span

        assert current_span() is None

    def test_returns_span_inside(self):
        from smartmemory.observability.tracing import trace_span, current_span

        with patch("smartmemory.observability.tracing._emit_span"):
            with trace_span("op") as span:
                assert current_span() is span


# ---------------------------------------------------------------------------
# Disabled mode
# ---------------------------------------------------------------------------
class TestDisabledMode:
    def test_is_enabled_returns_false(self):
        with patch("smartmemory.observability.tracing._is_enabled", return_value=False):
            from smartmemory.observability.tracing import trace_span

            with trace_span("op") as span:
                assert span.trace_id == ""
                assert span.span_id == ""
                assert span.name == ""

    def test_does_not_call_emit_span(self):
        with patch("smartmemory.observability.tracing._is_enabled", return_value=False):
            with patch("smartmemory.observability.tracing._emit_span") as mock_emit:
                from smartmemory.observability.tracing import trace_span

                with trace_span("op"):
                    pass

                mock_emit.assert_not_called()

    def test_disabled_spans_are_independent(self):
        """Each disabled span is a fresh SpanContext — no shared mutable state."""
        with patch("smartmemory.observability.tracing._is_enabled", return_value=False):
            from smartmemory.observability.tracing import trace_span, SpanContext

            with trace_span("a") as s1:
                s1.attributes["key"] = "val"
            with trace_span("b") as s2:
                pass
            assert isinstance(s1, SpanContext)
            assert isinstance(s2, SpanContext)
            assert s1 is not s2  # Fresh instance each time
            assert "key" not in s2.attributes  # No cross-bleed


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------
class TestThreadSafety:
    def test_different_threads_get_independent_trace_ids(self):
        from smartmemory.observability.tracing import trace_span

        results = {}
        barrier = threading.Barrier(2)

        def worker(name):
            with patch("smartmemory.observability.tracing._emit_span"):
                with trace_span(f"thread.{name}") as span:
                    barrier.wait(timeout=2)
                    results[name] = span.trace_id

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert "a" in results
        assert "b" in results
        assert results["a"] != results["b"]

    def test_main_thread_unaffected_by_child_threads(self):
        from smartmemory.observability.tracing import trace_span, current_trace_id

        child_tid = {}

        def child_worker():
            with patch("smartmemory.observability.tracing._emit_span"):
                with trace_span("child.op") as span:
                    child_tid["value"] = span.trace_id

        with patch("smartmemory.observability.tracing._emit_span"):
            with trace_span("main.op") as main_span:
                t = threading.Thread(target=child_worker)
                t.start()
                t.join(timeout=5)

                # Main thread still has its own trace_id
                assert current_trace_id() == main_span.trace_id
                # Child had a different trace_id
                assert child_tid["value"] != main_span.trace_id
