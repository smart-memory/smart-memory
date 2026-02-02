"""
Tests for Claude CLI Provider

These tests mock subprocess calls since actual CLI execution requires authentication.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from smartmemory.integration.llm.claude_cli_provider import (
    ClaudeCLIProvider,
    create_claude_cli_provider,
)


class TestClaudeCLIProviderInitialization:
    """Test provider initialization."""

    @patch("subprocess.run")
    def test_init_verifies_cli_available(self, mock_run):
        """Provider should verify CLI is available on init."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="claude-code version 1.0.0",
            stderr=""
        )

        ClaudeCLIProvider()

        mock_run.assert_called_once()
        assert "--version" in mock_run.call_args[0][0]

    @patch("subprocess.run")
    def test_init_raises_if_cli_not_found(self, mock_run):
        """Provider should raise if CLI command not found."""
        mock_run.side_effect = FileNotFoundError("claude not found")

        with pytest.raises(RuntimeError, match="Claude CLI not found"):
            ClaudeCLIProvider()

    @patch("subprocess.run")
    def test_init_custom_command(self, mock_run):
        """Provider should support custom command name."""
        mock_run.return_value = MagicMock(returncode=0, stdout="v1.0.0", stderr="")

        provider = ClaudeCLIProvider(command="my-claude")

        assert provider.command == "my-claude"
        assert mock_run.call_args[0][0][0] == "my-claude"


class TestModelNormalization:
    """Test model name normalization."""

    @patch("subprocess.run")
    def test_normalize_standard_models(self, mock_run):
        """Should normalize standard model names."""
        mock_run.return_value = MagicMock(returncode=0, stdout="v1.0.0", stderr="")
        provider = ClaudeCLIProvider()

        assert provider._normalize_model("opus") == "opus"
        assert provider._normalize_model("sonnet") == "sonnet"
        assert provider._normalize_model("haiku") == "haiku"

    @patch("subprocess.run")
    def test_normalize_versioned_models(self, mock_run):
        """Should normalize versioned model names."""
        mock_run.return_value = MagicMock(returncode=0, stdout="v1.0.0", stderr="")
        provider = ClaudeCLIProvider()

        assert provider._normalize_model("opus-4.5") == "opus"
        assert provider._normalize_model("sonnet-4.1") == "sonnet"
        assert provider._normalize_model("claude-opus-4-5") == "opus"

    @patch("subprocess.run")
    def test_normalize_aliases(self, mock_run):
        """Should normalize alias names."""
        mock_run.return_value = MagicMock(returncode=0, stdout="v1.0.0", stderr="")
        provider = ClaudeCLIProvider()

        assert provider._normalize_model("default") == "sonnet"
        assert provider._normalize_model("fast") == "haiku"
        assert provider._normalize_model("best") == "opus"

    @patch("subprocess.run")
    def test_normalize_unknown_passthrough(self, mock_run):
        """Unknown models should pass through unchanged."""
        mock_run.return_value = MagicMock(returncode=0, stdout="v1.0.0", stderr="")
        provider = ClaudeCLIProvider()

        assert provider._normalize_model("custom-model") == "custom-model"


class TestEnvironmentCleaning:
    """Test environment variable handling."""

    @patch("subprocess.run")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test", "PATH": "/usr/bin"})
    def test_clean_env_removes_api_keys(self, mock_run):
        """Should remove API keys from environment."""
        mock_run.return_value = MagicMock(returncode=0, stdout="v1.0.0", stderr="")
        provider = ClaudeCLIProvider()

        clean_env = provider._get_clean_env()

        assert "ANTHROPIC_API_KEY" not in clean_env
        assert "PATH" in clean_env  # Other vars preserved


class TestChatCompletion:
    """Test chat completion functionality."""

    @patch("subprocess.run")
    def test_chat_completion_basic(self, mock_run):
        """Should execute chat completion via CLI."""
        # First call: version check
        # Second call: actual completion
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0.0", stderr=""),
            MagicMock(
                returncode=0,
                stdout=json.dumps({"result": "Hello! How can I help?"}),
                stderr=""
            ),
        ]

        provider = ClaudeCLIProvider()
        result = provider.chat_completion([
            {"role": "user", "content": "Hello"}
        ])

        assert "content" in result
        assert "Hello" in result["content"] or "How can I help" in result["content"]

    @patch("subprocess.run")
    def test_chat_completion_with_system_prompt(self, mock_run):
        """Should handle system prompts."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0.0", stderr=""),
            MagicMock(
                returncode=0,
                stdout=json.dumps({"result": "I am a helpful assistant"}),
                stderr=""
            ),
        ]

        provider = ClaudeCLIProvider()
        result = provider.chat_completion([
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Who are you?"}
        ])

        # Verify system prompt was passed
        call_args = mock_run.call_args[0][0]
        assert "--append-system-prompt" in call_args

    @patch("subprocess.run")
    def test_chat_completion_with_model_selection(self, mock_run):
        """Should pass model selection to CLI."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0.0", stderr=""),
            MagicMock(returncode=0, stdout='{"result": "test"}', stderr=""),
        ]

        provider = ClaudeCLIProvider()
        provider.chat_completion(
            [{"role": "user", "content": "test"}],
            model="opus"
        )

        call_args = mock_run.call_args[0][0]
        assert "--model" in call_args
        model_idx = call_args.index("--model")
        assert call_args[model_idx + 1] == "opus"

    @patch("subprocess.run")
    def test_chat_completion_timeout(self, mock_run):
        """Should handle CLI timeout."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0.0", stderr=""),
            subprocess.TimeoutExpired(cmd="claude", timeout=120),
        ]

        provider = ClaudeCLIProvider()

        with pytest.raises(RuntimeError, match="timed out"):
            provider.chat_completion([{"role": "user", "content": "test"}])

    @patch("subprocess.run")
    def test_chat_completion_cli_error(self, mock_run):
        """Should handle CLI errors."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0.0", stderr=""),
            MagicMock(returncode=1, stdout="", stderr="Rate limit exceeded"),
        ]

        provider = ClaudeCLIProvider()

        with pytest.raises(RuntimeError, match="Rate limit"):
            provider.chat_completion([{"role": "user", "content": "test"}])


class TestStructuredCompletion:
    """Test structured completion functionality."""

    @patch("subprocess.run")
    def test_structured_completion_with_pydantic(self, mock_run):
        """Should request JSON and parse Pydantic-style response."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            count: int

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0.0", stderr=""),
            MagicMock(
                returncode=0,
                stdout=json.dumps({"result": '{"name": "test", "count": 42}'}),
                stderr=""
            ),
        ]

        provider = ClaudeCLIProvider()
        result = provider.structured_completion(
            [{"role": "user", "content": "Give me a name and count"}],
            TestModel
        )

        assert result is not None
        assert result["parsed_data"]["name"] == "test"
        assert result["parsed_data"]["count"] == 42
        assert "models" in result

    @patch("subprocess.run")
    def test_structured_completion_parse_failure(self, mock_run):
        """Should return None if JSON parsing fails."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0.0", stderr=""),
            MagicMock(
                returncode=0,
                stdout=json.dumps({"result": "This is not JSON"}),
                stderr=""
            ),
        ]

        provider = ClaudeCLIProvider()
        result = provider.structured_completion(
            [{"role": "user", "content": "test"}],
            dict  # Simple type
        )

        assert result is None


class TestFactoryFunction:
    """Test factory function."""

    @patch("subprocess.run")
    def test_create_claude_cli_provider(self, mock_run):
        """Factory should create provider with options."""
        mock_run.return_value = MagicMock(returncode=0, stdout="v1.0.0", stderr="")

        provider = create_claude_cli_provider(
            timeout_seconds=60,
            default_model="opus"
        )

        assert provider.timeout_seconds == 60
        assert provider.default_model == "opus"


class TestJSONParsing:
    """Test JSON response parsing."""

    @patch("subprocess.run")
    def test_parse_direct_json(self, mock_run):
        """Should parse direct JSON response."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0.0", stderr=""),
            MagicMock(returncode=0, stdout='{"result": "test"}', stderr=""),
        ]

        provider = ClaudeCLIProvider()
        result = provider._parse_json_response('{"result": "test"}')

        assert result["result"] == "test"

    @patch("subprocess.run")
    def test_parse_json_with_prefix(self, mock_run):
        """Should find JSON in mixed output."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0.0", stderr=""),
        ]

        provider = ClaudeCLIProvider()
        result = provider._parse_json_response('Some prefix\n{"result": "test"}')

        assert result["result"] == "test"

    @patch("subprocess.run")
    def test_parse_non_json_fallback(self, mock_run):
        """Should return raw text if no JSON found."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0.0", stderr=""),
        ]

        provider = ClaudeCLIProvider()
        result = provider._parse_json_response("Just plain text")

        assert result["result"] == "Just plain text"


class TestSupportedFeatures:
    """Test feature reporting."""

    @patch("subprocess.run")
    def test_get_supported_features(self, mock_run):
        """Should report supported features."""
        mock_run.return_value = MagicMock(returncode=0, stdout="v1.0.0", stderr="")

        provider = ClaudeCLIProvider()
        features = provider.get_supported_features()

        assert "chat_completion" in features


class TestValidateConnection:
    """Test connection validation."""

    @patch("subprocess.run")
    def test_validate_connection_success(self, mock_run):
        """Should return True if CLI works."""
        mock_run.return_value = MagicMock(returncode=0, stdout="v1.0.0", stderr="")

        provider = ClaudeCLIProvider()
        assert provider.validate_connection() is True

    @patch("subprocess.run")
    def test_validate_connection_failure(self, mock_run):
        """Should return False if CLI fails."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0.0", stderr=""),  # Init
            FileNotFoundError("not found"),  # Validation
        ]

        provider = ClaudeCLIProvider()
        assert provider.validate_connection() is False
