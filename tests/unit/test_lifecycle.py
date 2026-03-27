"""Tests for DIST-AGENT-HOOKS-1 lifecycle engine."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from smartmemory_app.lifecycle import MemoryLifecycle, _estimate_tokens
from smartmemory_app.lifecycle_config import LifecycleConfig, RecallStrategy


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Use a temp dir for session state files."""
    with patch.dict(os.environ, {"SMARTMEMORY_DATA_DIR": str(tmp_path)}):
        yield tmp_path


@pytest.fixture
def config():
    return LifecycleConfig()


@pytest.fixture
def disabled_config():
    return LifecycleConfig(enabled=False)


class TestLifecycleConfig:
    def test_defaults(self):
        cfg = LifecycleConfig()
        assert cfg.enabled is True
        assert cfg.recall_strategy == RecallStrategy.TOPIC_CHANGE
        assert cfg.orient_budget == 1500
        assert cfg.recall_budget == 500

    def test_from_config_empty(self):
        cfg = LifecycleConfig.from_config({})
        assert cfg.enabled is True

    def test_from_config_none(self):
        cfg = LifecycleConfig.from_config(None)
        assert cfg.enabled is True

    def test_from_config_full(self):
        raw = {
            "enabled": False,
            "recall_strategy": "every_prompt",
            "orient_budget": 2000,
            "recall_budget": 800,
            "topic_threshold": 0.5,
            "observe_tool_calls": False,
            "distill_turns": False,
            "learn_from_errors": False,
        }
        cfg = LifecycleConfig.from_config(raw)
        assert cfg.enabled is False
        assert cfg.recall_strategy == RecallStrategy.EVERY_PROMPT
        assert cfg.orient_budget == 2000
        assert cfg.observe_tool_calls is False

    def test_from_config_invalid_strategy_falls_back(self):
        cfg = LifecycleConfig.from_config({"recall_strategy": "bogus"})
        assert cfg.recall_strategy == RecallStrategy.TOPIC_CHANGE

    def test_apply_overrides(self):
        cfg = LifecycleConfig()
        overridden = cfg.apply_overrides({"recall_strategy": "every_prompt", "orient_budget": 3000})
        assert overridden.recall_strategy == RecallStrategy.EVERY_PROMPT
        assert overridden.orient_budget == 3000
        # Original unchanged
        assert cfg.recall_strategy == RecallStrategy.TOPIC_CHANGE


class TestOrient:
    @patch("smartmemory_app.storage.search", return_value=[])
    @patch("smartmemory_app.storage.recall", return_value="## Context\n- item 1")
    def test_orient_returns_context(self, mock_recall, mock_search, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        result = lc.orient("/some/path")
        assert "Context" in result
        mock_recall.assert_called_once()

    def test_orient_disabled_returns_empty(self, tmp_data_dir, disabled_config):
        lc = MemoryLifecycle("test-session", disabled_config)
        assert lc.orient() == ""

    @patch("smartmemory_app.storage.recall", return_value="")
    def test_orient_clears_state(self, mock_recall, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        lc._current_user_turn = "old prompt"
        lc._turn_count = 5
        lc.orient()
        assert lc._current_user_turn is None
        assert lc._turn_count == 0


class TestRecall:
    @patch("smartmemory_app.storage.search", return_value=[
        {"content": "relevant memory", "memory_type": "semantic"},
    ])
    def test_recall_returns_context(self, mock_search, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        # First call with no prior injection → always fires
        result = lc.recall("how does auth work?")
        assert "relevant memory" in result
        assert lc._current_user_turn == "how does auth work?"

    def test_recall_always_captures_prompt(self, tmp_data_dir):
        cfg = LifecycleConfig(recall_strategy=RecallStrategy.SESSION_ONLY)
        lc = MemoryLifecycle("test-session", cfg)
        result = lc.recall("some prompt")
        assert result == ""  # session_only skips recall
        assert lc._current_user_turn == "some prompt"  # but prompt captured

    def test_recall_disabled_returns_empty(self, tmp_data_dir, disabled_config):
        lc = MemoryLifecycle("test-session", disabled_config)
        assert lc.recall("hello") == ""


class TestShouldRecall:
    def test_session_only_always_skips(self, tmp_data_dir):
        cfg = LifecycleConfig(recall_strategy=RecallStrategy.SESSION_ONLY)
        lc = MemoryLifecycle("test-session", cfg)
        assert lc._should_recall("any prompt") is False

    def test_empty_prompt_skips(self, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        assert lc._should_recall("") is False
        assert lc._should_recall("   ") is False

    def test_confirmation_skips(self, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        assert lc._should_recall("yes") is False
        assert lc._should_recall("no") is False
        assert lc._should_recall("ok") is False
        assert lc._should_recall("y") is False

    def test_slash_command_skips(self, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        assert lc._should_recall("/commit") is False
        assert lc._should_recall("/help") is False

    def test_dedup_skips(self, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        lc._last_recalled_prompt = "same prompt"
        assert lc._should_recall("same prompt") is False

    def test_every_prompt_fires_after_gate(self, tmp_data_dir):
        cfg = LifecycleConfig(recall_strategy=RecallStrategy.EVERY_PROMPT)
        lc = MemoryLifecycle("test-session", cfg)
        assert lc._should_recall("how does auth work?") is True

    def test_topic_change_fires_on_first_prompt(self, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        # No prior embedding → should recall
        assert lc._should_recall("how does auth work?") is True


class TestObserve:
    @patch("smartmemory_app.storage.ingest", return_value="item-123")
    def test_observe_ingests(self, mock_ingest, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        lc.observe("Bash", {"command": "ls"}, "file1.py\nfile2.py")
        mock_ingest.assert_called_once()
        call_kwargs = mock_ingest.call_args
        assert "hook:observe" in str(call_kwargs)
        assert lc._observation_count == 1

    def test_observe_disabled_skips(self, tmp_data_dir):
        cfg = LifecycleConfig(observe_tool_calls=False)
        lc = MemoryLifecycle("test-session", cfg)
        lc.observe("Bash", {}, "")
        assert lc._observation_count == 0


class TestDistill:
    @patch("smartmemory_app.storage.ingest", return_value="item-456")
    def test_distill_pairs_turn(self, mock_ingest, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        lc._current_user_turn = "how does auth work?"
        lc.distill("Auth uses JWT tokens stored in cookies.")
        mock_ingest.assert_called_once()
        call_args = str(mock_ingest.call_args)
        assert "lifecycle:distill" in call_args
        assert "how does auth work?" in call_args
        assert lc._current_user_turn is None  # consumed

    def test_distill_disabled_skips(self, tmp_data_dir):
        cfg = LifecycleConfig(distill_turns=False)
        lc = MemoryLifecycle("test-session", cfg)
        lc._current_user_turn = "prompt"
        lc.distill("response")
        # current_user_turn NOT consumed (distill didn't run)
        assert lc._current_user_turn == "prompt"


class TestLearn:
    @patch("smartmemory_app.storage.ingest", return_value="item-789")
    def test_learn_ingests_error(self, mock_ingest, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        lc.learn("Bash", "command not found: foobar")
        mock_ingest.assert_called_once()
        assert "hook:learn" in str(mock_ingest.call_args)


class TestPersist:
    @patch("smartmemory_app.storage.ingest", return_value="item-999")
    def test_persist_saves_summary(self, mock_ingest, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        lc._last_assistant_message = "Fixed the auth bug by updating JWT validation."
        lc._turn_count = 5
        lc._observation_count = 12
        lc.persist()
        mock_ingest.assert_called_once()
        assert "hook:persist" in str(mock_ingest.call_args)
        # State file should be deleted
        assert not lc._state_path().exists()

    def test_persist_no_message_just_cleans_up(self, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        lc._save_state()  # create state file
        assert lc._state_path().exists()
        lc.persist()
        assert not lc._state_path().exists()


class TestSessionState:
    def test_state_round_trips(self, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        lc._current_user_turn = "hello"
        lc._turn_count = 3
        lc._observation_count = 7
        lc._config_overrides = {"recall_strategy": "every_prompt"}
        lc._save_state()

        lc2 = MemoryLifecycle("test-session", config)
        assert lc2._current_user_turn == "hello"
        assert lc2._turn_count == 3
        assert lc2._observation_count == 7
        assert lc2._config_overrides == {"recall_strategy": "every_prompt"}

    def test_different_sessions_isolated(self, tmp_data_dir, config):
        lc1 = MemoryLifecycle("session-a", config)
        lc1._current_user_turn = "prompt A"
        lc1._save_state()

        lc2 = MemoryLifecycle("session-b", config)
        assert lc2._current_user_turn is None  # not contaminated

    def test_cleanup_stale_sessions(self, tmp_data_dir, config):
        lc = MemoryLifecycle("old-session", config)
        lc._save_state()
        # Backdate the file
        path = lc._state_path()
        old_time = path.stat().st_mtime - (25 * 3600)
        os.utime(path, (old_time, old_time))

        deleted = MemoryLifecycle.cleanup_stale_sessions(max_age_hours=24)
        assert deleted == 1
        assert not path.exists()


class TestFormatting:
    def test_estimate_tokens(self):
        assert _estimate_tokens("hello") == 1  # 5 chars / 4 = 1
        assert _estimate_tokens("a" * 100) == 25

    def test_orient_block_respects_budget(self, tmp_data_dir):
        cfg = LifecycleConfig(orient_budget=10)  # very small budget
        lc = MemoryLifecycle("test-session", cfg)
        # Long context should be included (budget check is on token estimate)
        result = lc._format_orient_block("short", [])
        assert result == "short"

    def test_recall_block_respects_budget(self, tmp_data_dir, config):
        lc = MemoryLifecycle("test-session", config)
        results = [{"content": f"memory {i}", "memory_type": "semantic"} for i in range(100)]
        result = lc._format_recall_block(results)
        # Should not include all 100 — budget limits it
        assert result.count("memory") < 100
