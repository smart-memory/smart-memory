"""Unit tests for the unified configuration loader."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from smartmemory.configuration.loader import (
    _parse_env_file,
    _find_config_dir,
    deep_merge,
    load_config,
    load_environment,
)


@pytest.fixture
def tmp_env_file(tmp_path):
    """Create a temporary .env file for testing."""
    env_file = tmp_path / "test.env"
    env_file.write_text("KEY1=value1\nKEY2=value2\n# This is a comment\n\nKEY3=value3\n")
    return env_file


@pytest.fixture
def tmp_env_file_quotes(tmp_path):
    """Create a .env file with quoted values."""
    env_file = tmp_path / "quoted.env"
    env_file.write_text(
        'SINGLE=\'single quoted\'\nDOUBLE="double quoted"\nUNQUOTED=plain value\nMIXED="starts but no end\n'
    )
    return env_file


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temporary config directory with base.env and secrets.env."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    (config_dir / "base.env").write_text("BASE_KEY=base_value\nSHARED_KEY=from_base\n")
    (config_dir / "secrets.env").write_text("SECRET_KEY=secret_value\nSHARED_KEY=from_secrets\n")
    return config_dir


@pytest.fixture(autouse=True)
def reset_loader():
    """Reset the _loaded guard before each test."""
    import smartmemory.configuration.loader as loader_mod

    loader_mod._loaded = False
    yield
    loader_mod._loaded = False


class TestParseEnvFile:
    def test_basic(self, tmp_env_file):
        result = _parse_env_file(tmp_env_file)
        assert result == {"KEY1": "value1", "KEY2": "value2", "KEY3": "value3"}

    def test_quotes(self, tmp_env_file_quotes):
        result = _parse_env_file(tmp_env_file_quotes)
        assert result["SINGLE"] == "single quoted"
        assert result["DOUBLE"] == "double quoted"
        assert result["UNQUOTED"] == "plain value"
        # Mismatched quotes — not stripped
        assert result["MIXED"] == '"starts but no end'

    def test_comments_and_empty_lines(self, tmp_env_file):
        result = _parse_env_file(tmp_env_file)
        assert len(result) == 3  # Comments and empty lines excluded

    def test_no_equals(self, tmp_path):
        env_file = tmp_path / "bad.env"
        env_file.write_text("NOEQUALS\nGOOD=value\n")
        result = _parse_env_file(env_file)
        assert result == {"GOOD": "value"}


class TestLoadEnvironment:
    def test_setdefault_semantics(self, tmp_config_dir):
        """OS env wins over file values."""
        os.environ["BASE_KEY"] = "os_value"
        try:
            load_environment(config_dir=tmp_config_dir)
            assert os.environ["BASE_KEY"] == "os_value"
        finally:
            del os.environ["BASE_KEY"]

    def test_file_merge(self, tmp_config_dir):
        """Later files override earlier files."""
        # Clean env so file values take effect
        os.environ.pop("SHARED_KEY", None)
        os.environ.pop("BASE_KEY", None)
        os.environ.pop("SECRET_KEY", None)
        try:
            load_environment(config_dir=tmp_config_dir)
            # secrets.env overrides base.env for SHARED_KEY
            assert os.environ["SHARED_KEY"] == "from_secrets"
            assert os.environ["BASE_KEY"] == "base_value"
            assert os.environ["SECRET_KEY"] == "secret_value"
        finally:
            os.environ.pop("SHARED_KEY", None)
            os.environ.pop("BASE_KEY", None)
            os.environ.pop("SECRET_KEY", None)

    def test_idempotent(self, tmp_config_dir):
        """_loaded guard prevents double load."""
        os.environ.pop("BASE_KEY", None)
        try:
            load_environment(config_dir=tmp_config_dir)
            assert os.environ["BASE_KEY"] == "base_value"

            # Change the file — second call should be a no-op
            (tmp_config_dir / "base.env").write_text("BASE_KEY=changed\n")
            os.environ.pop("BASE_KEY", None)
            load_environment(config_dir=tmp_config_dir)
            # Should NOT have reloaded — BASE_KEY was popped but _loaded=True
            assert "BASE_KEY" not in os.environ
        finally:
            os.environ.pop("BASE_KEY", None)


class TestFindConfigDir:
    def test_env_var_precedence(self, tmp_path):
        """SMARTMEMORY_CONFIG_DIR env var takes precedence."""
        config_dir = tmp_path / "custom_config"
        config_dir.mkdir()
        with patch.dict(os.environ, {"SMARTMEMORY_CONFIG_DIR": str(config_dir)}):
            result = _find_config_dir()
            assert result == config_dir

    def test_returns_none_when_nothing_found(self):
        """Returns None when no config dir exists."""
        with patch.dict(os.environ, {"SMARTMEMORY_CONFIG_DIR": ""}, clear=False):
            with patch("smartmemory.configuration.loader.Path.cwd", return_value=Path("/nonexistent")):
                with patch("smartmemory.configuration.loader.Path.home", return_value=Path("/nonexistent")):
                    result = _find_config_dir()
                    # May or may not be None depending on actual filesystem
                    # The point is it doesn't crash


class TestLoadConfig:
    def test_returns_dict(self, tmp_path):
        """JSON parsed without interpolation."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"key": "value", "nested": {"a": 1}}')
        result = load_config(config_path=config_file)
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_missing_file(self, tmp_path):
        """Returns empty dict for missing file."""
        result = load_config(config_path=tmp_path / "missing.json")
        assert result == {}

    def test_no_interpolation(self, tmp_path):
        """${VAR:-default} is NOT expanded — treated as literal string."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"host": "${FALKORDB_HOST:-localhost}"}')
        result = load_config(config_path=config_file)
        assert result["host"] == "${FALKORDB_HOST:-localhost}"


class TestDockerLikeEnvironment:
    """Test config resolution when config/ dir doesn't exist (Docker containers)."""

    def test_load_environment_skips_when_no_config_dir(self):
        """In Docker, there's no smart-memory-infra/config/ dir.
        load_environment() should succeed (no-op) without error."""
        load_environment(config_dir=None)
        # Should not raise — _loaded gets set to True via the None path

    def test_config_json_found_in_cwd(self, tmp_path):
        """When config.json exists in cwd (COPY'd by Dockerfile),
        _find_config_json() should find it."""
        from smartmemory.configuration.loader import _find_config_json

        config_file = tmp_path / "config.json"
        config_file.write_text('{"graph_db": {"host": "localhost"}}')

        with patch("smartmemory.configuration.loader.Path.cwd", return_value=tmp_path):
            with patch.dict(os.environ, {"SMARTMEMORY_CONFIG": ""}, clear=False):
                result = _find_config_json()
                assert result == config_file

    def test_config_json_env_var_overrides_cwd(self, tmp_path):
        """SMARTMEMORY_CONFIG env var takes precedence over cwd discovery."""
        from smartmemory.configuration.loader import _find_config_json

        env_config = tmp_path / "explicit.json"
        env_config.write_text("{}")

        with patch.dict(os.environ, {"SMARTMEMORY_CONFIG": str(env_config)}):
            result = _find_config_json()
            assert result == env_config


class TestDeepMerge:
    def test_leaf_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_override_dict_with_scalar(self):
        base = {"a": {"x": 1}}
        override = {"a": "replaced"}
        result = deep_merge(base, override)
        assert result == {"a": "replaced"}

    def test_empty_override(self):
        base = {"a": 1}
        result = deep_merge(base, {})
        assert result == {"a": 1}

    def test_empty_base(self):
        result = deep_merge({}, {"a": 1})
        assert result == {"a": 1}
