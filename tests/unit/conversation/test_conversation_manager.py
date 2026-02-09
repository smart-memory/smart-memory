"""Unit tests for ConversationManager, ConversationContext, and ConversationSession."""

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.unit

from smartmemory.conversation.context import ConversationContext
from smartmemory.conversation.session import ConversationSession
from smartmemory.conversation.manager import ConversationManager


# ---------------------------------------------------------------------------
# ConversationContext
# ---------------------------------------------------------------------------
class TestConversationContext:
    def test_defaults(self):
        ctx = ConversationContext()
        assert ctx.conversation_id is None
        assert ctx.participant_id is None
        assert ctx.topics == []
        assert ctx.entities == []
        assert ctx.sentiment is None
        assert ctx.turn_history == []
        assert ctx.active_threads == []
        assert ctx.coreference_chains == []
        assert ctx.extra == {}

    def test_to_dict_round_trip(self):
        ctx = ConversationContext(
            conversation_id="conv_1",
            participant_id="user_1",
            topics=["python", "ml"],
            entities=[{"name": "Python", "type": "language"}],
            sentiment="positive",
            turn_history=[{"role": "user", "content": "hello"}],
            active_threads=["thread_1"],
            coreference_chains=[{"mentions": ["it"], "head": "Python"}],
            extra={"key": "value"},
        )
        d = ctx.to_dict()
        assert d["conversation_id"] == "conv_1"
        assert d["topics"] == ["python", "ml"]
        assert d["sentiment"] == "positive"
        assert d["extra"] == {"key": "value"}

        restored = ConversationContext.from_dict(d)
        assert restored.conversation_id == "conv_1"
        assert restored.topics == ["python", "ml"]
        assert restored.entities == [{"name": "Python", "type": "language"}]

    def test_from_dict_none(self):
        ctx = ConversationContext.from_dict(None)
        assert ctx.conversation_id is None
        assert ctx.topics == []

    def test_from_dict_empty(self):
        ctx = ConversationContext.from_dict({})
        assert ctx.conversation_id is None
        assert ctx.topics == []

    def test_created_at_is_utc(self):
        ctx = ConversationContext()
        assert ctx.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# ConversationSession
# ---------------------------------------------------------------------------
class TestConversationSession:
    def test_defaults(self):
        sess = ConversationSession(session_id="s1")
        assert sess.session_id == "s1"
        assert sess.participant_id is None
        assert sess.conversation_type == "default"
        assert sess.is_active is True
        assert isinstance(sess.context, ConversationContext)

    def test_touch_updates_last_activity(self):
        sess = ConversationSession(session_id="s1")
        before = sess.last_activity
        time.sleep(0.01)
        sess.touch()
        assert sess.last_activity > before

    def test_to_dict(self):
        sess = ConversationSession(session_id="s1", participant_id="u1")
        d = sess.to_dict()
        assert d["session_id"] == "s1"
        assert d["participant_id"] == "u1"
        assert d["is_active"] is True
        assert "context" in d


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------
class TestConversationManager:
    @pytest.fixture
    def mgr(self):
        return ConversationManager()

    def test_create_session_default(self, mgr):
        sess = mgr.create_session()
        assert sess.is_active is True
        assert sess.session_id is not None
        assert mgr.get_session(sess.session_id) is sess

    def test_create_session_custom_id(self, mgr):
        sess = mgr.create_session(session_id="custom_id", participant_id="user_1")
        assert sess.session_id == "custom_id"
        assert sess.participant_id == "user_1"

    def test_create_session_with_context(self, mgr):
        ctx = ConversationContext(conversation_id="c1", topics=["ai"])
        sess = mgr.create_session(context=ctx)
        assert sess.context.topics == ["ai"]

    def test_get_session_not_found(self, mgr):
        assert mgr.get_session("nonexistent") is None

    def test_get_session_empty_id_raises(self, mgr):
        with pytest.raises(ValueError, match="session_id is required"):
            mgr.get_session("")

    def test_end_session(self, mgr):
        sess = mgr.create_session(session_id="s1")
        assert mgr.end_session("s1") is True
        assert sess.is_active is False
        # Session still exists, just inactive
        assert mgr.get_session("s1") is not None

    def test_end_session_not_found(self, mgr):
        assert mgr.end_session("nonexistent") is False

    def test_end_session_empty_id_raises(self, mgr):
        with pytest.raises(ValueError):
            mgr.end_session("")

    def test_delete_session(self, mgr):
        mgr.create_session(session_id="s1")
        assert mgr.delete_session("s1") is True
        assert mgr.get_session("s1") is None

    def test_delete_session_not_found(self, mgr):
        assert mgr.delete_session("nonexistent") is False

    def test_delete_session_empty_id_raises(self, mgr):
        with pytest.raises(ValueError):
            mgr.delete_session("")

    def test_list_active_sessions(self, mgr):
        mgr.create_session(session_id="s1")
        mgr.create_session(session_id="s2")
        mgr.create_session(session_id="s3")
        mgr.end_session("s2")
        active = mgr.list_active_sessions()
        assert "s1" in active
        assert "s3" in active
        assert "s2" not in active

    def test_touch(self, mgr):
        sess = mgr.create_session(session_id="s1")
        before = sess.last_activity
        time.sleep(0.01)
        mgr.touch("s1")
        assert sess.last_activity > before

    def test_touch_not_found_raises(self, mgr):
        with pytest.raises(KeyError, match="session not found"):
            mgr.touch("nonexistent")

    def test_touch_empty_id_raises(self, mgr):
        with pytest.raises(ValueError):
            mgr.touch("")

    def test_cleanup_inactive_sessions(self, mgr):
        sess = mgr.create_session(session_id="old")
        # Backdate last_activity
        sess.last_activity = datetime.now(timezone.utc) - timedelta(hours=48)
        mgr.create_session(session_id="recent")

        cleaned = mgr.cleanup_inactive_sessions(timeout_hours=24)
        assert "old" in cleaned
        assert "recent" not in cleaned
        assert mgr.get_session("old").is_active is False
        assert mgr.get_session("recent").is_active is True

    def test_cleanup_invalid_timeout_raises(self, mgr):
        with pytest.raises(ValueError, match="timeout_hours must be positive"):
            mgr.cleanup_inactive_sessions(timeout_hours=0)

    def test_thread_safety(self, mgr):
        errors = []

        def create_sessions(start, count):
            try:
                for i in range(count):
                    mgr.create_session(session_id=f"thread_{start}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_sessions, args=(t, 10)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(mgr.list_active_sessions()) == 50
