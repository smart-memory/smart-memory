"""Tests for Claude CLI interface."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

# Skip all tests if claude-cli not installed
pytest.importorskip("claude_cli")

from claude_cli import (
    Claude,
    CLIError,
    CLINotFoundError,
    CLITimeoutError,
    Response,
)


class TestClaude:
    """Test main Claude class."""

    @patch("claude_cli.provider.subprocess.run")
    def test_simple_call(self, mock_run):
        """Test simple __call__ interface."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0", stderr=""),  # verify
            MagicMock(returncode=0, stdout='{"result": "4"}', stderr=""),
        ]

        claude = Claude()
        answer = claude("What is 2+2?")

        assert answer == "4"

    @patch("claude_cli.provider.subprocess.run")
    def test_simple_call_with_model(self, mock_run):
        """Test simple call with model override."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0", stderr=""),
            MagicMock(returncode=0, stdout='{"result": "test"}', stderr=""),
        ]

        claude = Claude(model="haiku")
        claude("test", model="opus")

        # Check that opus was passed in args
        call_args = mock_run.call_args[0][0]
        model_idx = call_args.index("--model")
        assert call_args[model_idx + 1] == "opus"

    @patch("claude_cli.provider.subprocess.run")
    def test_chat(self, mock_run):
        """Test chat interface."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0", stderr=""),
            MagicMock(
                returncode=0,
                stdout='{"result": "Hello!", "usage": {"input_tokens": 10, "output_tokens": 5}}',
                stderr="",
            ),
        ]

        claude = Claude()
        response = claude.chat([{"role": "user", "content": "Hi"}])

        assert isinstance(response, Response)
        assert response.content == "Hello!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5

    @patch("claude_cli.provider.subprocess.run")
    def test_chat_with_system(self, mock_run):
        """Test chat with system prompt."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0", stderr=""),
            MagicMock(returncode=0, stdout='{"result": "Hi"}', stderr=""),
        ]

        claude = Claude()
        claude.chat([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ])

        call_args = mock_run.call_args[0][0]
        assert "--append-system-prompt" in call_args

    @patch("claude_cli.provider.subprocess.run")
    def test_structured(self, mock_run):
        """Test structured output."""
        class Person(BaseModel):
            name: str
            age: int

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0", stderr=""),
            MagicMock(
                returncode=0,
                stdout='{"result": "{\\"name\\": \\"John\\", \\"age\\": 30}"}',
                stderr="",
            ),
        ]

        claude = Claude()
        person = claude.structured("John is 30", schema=Person)

        assert person.name == "John"
        assert person.age == 30

    @patch("claude_cli.provider.subprocess.run")
    def test_structured_with_markdown_fence(self, mock_run):
        """Test structured output strips markdown fences."""
        class Data(BaseModel):
            value: int

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0", stderr=""),
            MagicMock(
                returncode=0,
                stdout='{"result": "```json\\n{\\"value\\": 42}\\n```"}',
                stderr="",
            ),
        ]

        claude = Claude()
        data = claude.structured("test", schema=Data)

        assert data.value == 42

    @patch("claude_cli.provider.subprocess.run")
    def test_structured_safe_success(self, mock_run):
        """Test structured_safe returns StructuredResponse on success."""
        class Item(BaseModel):
            name: str

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0", stderr=""),
            MagicMock(returncode=0, stdout='{"result": "{\\"name\\": \\"test\\"}"}', stderr=""),
        ]

        claude = Claude()
        result = claude.structured_safe("test", schema=Item)

        assert result.success is True
        assert result.data.name == "test"

    @patch("claude_cli.provider.subprocess.run")
    def test_structured_safe_failure(self, mock_run):
        """Test structured_safe returns error instead of raising."""
        class Item(BaseModel):
            name: str

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0", stderr=""),
            MagicMock(returncode=0, stdout='{"result": "not json at all"}', stderr=""),
        ]

        claude = Claude()
        result = claude.structured_safe("test", schema=Item)

        assert result.success is False
        assert result.error is not None

    @patch("claude_cli.provider.subprocess.run")
    def test_is_available_true(self, mock_run):
        """Test is_available returns True when CLI works."""
        mock_run.return_value = MagicMock(returncode=0, stdout="v1.0", stderr="")

        claude = Claude()
        assert claude.is_available is True

    @patch("claude_cli.provider.subprocess.run")
    def test_is_available_false(self, mock_run):
        """Test is_available returns False when CLI not found."""
        mock_run.side_effect = FileNotFoundError()

        claude = Claude()
        assert claude.is_available is False


class TestModelAliases:
    """Test model alias resolution."""

    @patch("claude_cli.provider.subprocess.run")
    def test_fast_alias(self, mock_run):
        """Test 'fast' resolves to haiku."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0", stderr=""),
            MagicMock(returncode=0, stdout='{"result": "ok"}', stderr=""),
        ]

        claude = Claude()
        claude("test", model="fast")

        call_args = mock_run.call_args[0][0]
        model_idx = call_args.index("--model")
        assert call_args[model_idx + 1] == "haiku"

    @patch("claude_cli.provider.subprocess.run")
    def test_best_alias(self, mock_run):
        """Test 'best' resolves to opus."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0", stderr=""),
            MagicMock(returncode=0, stdout='{"result": "ok"}', stderr=""),
        ]

        claude = Claude()
        claude("test", model="best")

        call_args = mock_run.call_args[0][0]
        model_idx = call_args.index("--model")
        assert call_args[model_idx + 1] == "opus"


class TestErrors:
    """Test error handling."""

    @patch("claude_cli.provider.subprocess.run")
    def test_cli_not_found(self, mock_run):
        """Test CLINotFoundError when CLI not installed."""
        mock_run.side_effect = FileNotFoundError()

        claude = Claude()
        with pytest.raises(CLINotFoundError):
            claude("test")

    @patch("claude_cli.provider.subprocess.run")
    def test_cli_timeout(self, mock_run):
        """Test CLITimeoutError on timeout."""
        import subprocess
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0", stderr=""),
            subprocess.TimeoutExpired(cmd="claude", timeout=120),
        ]

        claude = Claude()
        with pytest.raises(CLITimeoutError):
            claude("test")

    @patch("claude_cli.provider.subprocess.run")
    def test_cli_error(self, mock_run):
        """Test CLIError on non-zero exit."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="v1.0", stderr=""),
            MagicMock(returncode=1, stdout="", stderr="Rate limit exceeded"),
        ]

        claude = Claude()
        with pytest.raises(CLIError, match="Rate limit"):
            claude("test")


class TestResponseModel:
    """Test Response model."""

    def test_response_str(self):
        """Test Response string conversion."""
        response = Response(content="Hello", model="haiku")
        assert str(response) == "Hello"

    def test_usage_total(self):
        """Test Usage total_tokens property."""
        from claude_cli import Usage

        usage = Usage(prompt_tokens=10, completion_tokens=20)
        assert usage.total_tokens == 30
