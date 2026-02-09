"""Unit tests for CLI feedback function."""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

from smartmemory.feedback.cli import human_feedback_fn


class TestHumanFeedbackFn:
    @patch("builtins.input", return_value="good")
    def test_returns_user_input(self, mock_input):
        result = human_feedback_fn(0, {}, [{"key": "a"}, {"key": "b"}])
        assert result == "good"

    @patch("builtins.input", return_value="stop")
    def test_stop_input(self, mock_input):
        result = human_feedback_fn(1, {}, [])
        assert result == "stop"

    @patch("builtins.input", return_value="good")
    def test_prints_step_info(self, mock_input, capsys):
        human_feedback_fn(3, {}, [1, 2, 3])
        captured = capsys.readouterr()
        assert "Step 3" in captured.out
        assert "3 results" in captured.out
