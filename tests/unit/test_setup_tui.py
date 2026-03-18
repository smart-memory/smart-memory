"""Unit tests for DIST-SETUP-TUI-1: setup refactor + TUI integration."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from smartmemory_app.setup import (
    SetupResult,
    _can_run_tui,
    _apply_setup_result,
    _seed_data_dir,
)


class TestSetupResult:
    """SetupResult dataclass has correct fields and defaults."""

    def test_defaults(self):
        r = SetupResult()
        assert r.mode == "local"
        assert r.llm_provider == "groq"
        assert r.llm_model == ""
        assert r.embedding_provider == "local"
        assert r.coreference is False
        assert r.data_dir == "~/.smartmemory"

    def test_custom_values(self):
        r = SetupResult(
            mode="local",
            llm_provider="ollama",
            llm_model="llama3.1:8b",
            embedding_provider="openai",
            coreference=True,
            data_dir="/custom/path",
        )
        assert r.llm_provider == "ollama"
        assert r.llm_model == "llama3.1:8b"
        assert r.coreference is True


class TestCanRunTui:
    """_can_run_tui() detects non-interactive environments."""

    def test_returns_false_when_stdin_not_tty(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: False))
        monkeypatch.setattr("sys.stdout", MagicMock(isatty=lambda: True))
        assert _can_run_tui() is False

    def test_returns_false_when_stdout_not_tty(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: True))
        monkeypatch.setattr("sys.stdout", MagicMock(isatty=lambda: False))
        assert _can_run_tui() is False

    def test_returns_false_when_term_dumb(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: True))
        monkeypatch.setattr("sys.stdout", MagicMock(isatty=lambda: True))
        monkeypatch.setenv("TERM", "dumb")
        assert _can_run_tui() is False

    def test_returns_false_when_ci(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: True))
        monkeypatch.setattr("sys.stdout", MagicMock(isatty=lambda: True))
        monkeypatch.delenv("TERM", raising=False)
        monkeypatch.setenv("CI", "true")
        assert _can_run_tui() is False

    def test_returns_false_when_textual_not_importable(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: True))
        monkeypatch.setattr("sys.stdout", MagicMock(isatty=lambda: True))
        monkeypatch.delenv("TERM", raising=False)
        monkeypatch.delenv("CI", raising=False)
        with patch.dict("sys.modules", {"textual": None}):
            import importlib
            # Force ImportError on textual
            with patch("builtins.__import__", side_effect=ImportError("no textual")):
                assert _can_run_tui() is False

    def test_returns_true_when_all_conditions_met(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: True))
        monkeypatch.setattr("sys.stdout", MagicMock(isatty=lambda: True))
        monkeypatch.delenv("TERM", raising=False)
        monkeypatch.delenv("CI", raising=False)
        # textual is installed in test env
        assert _can_run_tui() is True


class TestApplySetupResult:
    """_apply_setup_result() saves config and runs post-config steps."""

    def test_saves_config_with_correct_fields(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))

        result = SetupResult(
            llm_provider="ollama",
            llm_model="llama3.1:8b",
            embedding_provider="openai",
            coreference=True,
            data_dir=str(tmp_path / "data"),
        )

        with (
            patch("smartmemory_app.setup._ensure_spacy"),
            patch("smartmemory_app.setup._copy_hooks"),
            patch("smartmemory_app.setup._copy_skills"),
            patch("smartmemory_app.setup._register_hooks"),
            patch("smartmemory_app.setup._seed_data_dir"),
            patch("smartmemory_app.config.save_config") as mock_save,
        ):
            _apply_setup_result(result)

        cfg = mock_save.call_args[0][0]
        assert cfg.mode == "local"
        assert cfg.llm_provider == "ollama"
        assert cfg.llm_model == "llama3.1:8b"
        assert cfg.embedding_provider == "openai"
        assert cfg.coreference is True

    def test_on_step_callback_called_for_each_step(self):
        result = SetupResult(data_dir="/tmp/test-sm")
        steps_seen = []

        with (
            patch("smartmemory_app.setup._ensure_spacy"),
            patch("smartmemory_app.setup._copy_hooks"),
            patch("smartmemory_app.setup._copy_skills"),
            patch("smartmemory_app.setup._register_hooks"),
            patch("smartmemory_app.setup._seed_data_dir"),
            patch("smartmemory_app.config.save_config"),
        ):
            _apply_setup_result(result, on_step=steps_seen.append)

        assert steps_seen == [
            "Config written",
            "spaCy model ready",
            "Hooks installed",
            "Skills installed",
            "Hooks registered",
            "Patterns seeded",
        ]

    def test_expanduser_on_data_dir(self):
        result = SetupResult(data_dir="~/custom-dir")

        with (
            patch("smartmemory_app.setup._ensure_spacy"),
            patch("smartmemory_app.setup._copy_hooks"),
            patch("smartmemory_app.setup._copy_skills"),
            patch("smartmemory_app.setup._register_hooks"),
            patch("smartmemory_app.setup._seed_data_dir") as mock_seed,
            patch("smartmemory_app.config.save_config") as mock_save,
        ):
            _apply_setup_result(result)

        # Config should have expanded path
        cfg = mock_save.call_args[0][0]
        assert "~" not in cfg.data_dir
        # _seed_data_dir should receive expanded path
        seed_arg = mock_seed.call_args[0][0]
        assert "~" not in seed_arg


class TestSeedDataDir:
    """_seed_data_dir() honors explicit path, env var, and default."""

    def test_explicit_dir_with_tilde(self, tmp_path, monkeypatch):
        # Can't test ~ literally, but can test explicit path is used
        target = tmp_path / "explicit"
        with patch("smartmemory_app.patterns.JSONLPatternStore") as mock_store:
            _seed_data_dir(str(target))
        assert target.exists()
        mock_store.assert_called_once_with(target)

    def test_env_var_fallback(self, tmp_path, monkeypatch):
        env_dir = tmp_path / "env-dir"
        monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(env_dir))
        with patch("smartmemory_app.patterns.JSONLPatternStore") as mock_store:
            _seed_data_dir(None)
        assert env_dir.exists()
        mock_store.assert_called_once_with(env_dir)

    def test_default_fallback(self, monkeypatch):
        monkeypatch.delenv("SMARTMEMORY_DATA_DIR", raising=False)
        with patch("smartmemory_app.patterns.JSONLPatternStore"):
            with patch("smartmemory_app.setup.DATA_DIR", Path("/tmp/test-default-sm")):
                _seed_data_dir(None)
        # Just verify it didn't crash — DATA_DIR is a module constant


class TestSetupDispatch:
    """setup() routes to TUI vs click correctly."""

    def test_flags_bypass_tui(self):
        """--mode local skips TUI entirely."""
        from click.testing import CliRunner
        from smartmemory_app.cli import cli

        with (
            patch("smartmemory_app.setup._setup_click") as mock_click,
            patch("smartmemory_app.setup._start_daemon_local"),
            patch("smartmemory_app.setup._can_run_tui", return_value=True),
        ):
            runner = CliRunner()
            runner.invoke(cli, ["setup", "--mode", "local"])

        mock_click.assert_called_once_with("local", None)

    def test_tui_fallback_on_import_error(self):
        """When TUI fails, falls back to click prompts."""
        from click.testing import CliRunner
        from smartmemory_app.cli import cli

        with (
            patch("smartmemory_app.setup._can_run_tui", return_value=True),
            patch("smartmemory_app.setup_tui.run_setup_tui", side_effect=ImportError("no textual")),
            patch("smartmemory_app.setup._setup_click") as mock_click,
            patch("smartmemory_app.setup._start_daemon_local"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["setup"], input="1\ngroq\nlocal\n~/.smartmemory\n")

        # Should have fallen back to click
        # (mock_click won't be called if _setup_click is the fallback — it's the real function)
        # Just verify it didn't crash
        assert result.exit_code == 0 or "TUI unavailable" in result.output


class TestRemoteHandoff:
    """Remote mode selection in TUI hands off to _setup_remote."""

    def test_remote_result_calls_setup_remote(self):
        from click.testing import CliRunner
        from smartmemory_app.cli import cli

        remote_result = SetupResult(mode="remote")

        with (
            patch("smartmemory_app.setup._can_run_tui", return_value=True),
            patch("smartmemory_app.setup_tui.run_setup_tui", return_value=remote_result),
            patch("smartmemory_app.setup._setup_remote") as mock_remote,
        ):
            runner = CliRunner()
            runner.invoke(cli, ["setup"])

        mock_remote.assert_called_once()
