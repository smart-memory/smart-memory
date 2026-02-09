"""Unit tests for SlackFeedbackAgent and slack_feedback_fn."""

from unittest.mock import MagicMock, patch

import pytest

slack_sdk = pytest.importorskip("slack_sdk", reason="slack_sdk not installed")

pytestmark = pytest.mark.unit

from smartmemory.feedback.slack import SlackFeedbackAgent, slack_feedback_fn


@pytest.fixture
def mock_webclient():
    with patch("smartmemory.feedback.slack.WebClient") as MockWC:
        client = MagicMock()
        MockWC.return_value = client
        yield client


@pytest.fixture
def agent(mock_webclient):
    return SlackFeedbackAgent(token="xoxb-fake", channel_id="C123")


class TestSlackFeedbackAgent:
    def test_init_creates_client(self, mock_webclient):
        agent = SlackFeedbackAgent(token="xoxb-test", channel_id="C456")
        assert agent.channel_id == "C456"

    def test_post_message_success(self, agent, mock_webclient):
        mock_webclient.chat_postMessage.return_value = {
            "ok": True,
            "message": {"ts": "1234567890.123456", "text": "hello"},
        }
        result = agent._post_message("hello")
        assert result["ts"] == "1234567890.123456"
        mock_webclient.chat_postMessage.assert_called_once_with(
            channel="C123", text="hello", thread_ts=None
        )

    def test_post_message_with_thread(self, agent, mock_webclient):
        mock_webclient.chat_postMessage.return_value = {
            "ok": True,
            "message": {"ts": "111.222"},
        }
        agent._post_message("reply", thread_ts="000.111")
        mock_webclient.chat_postMessage.assert_called_once_with(
            channel="C123", text="reply", thread_ts="000.111"
        )

    def test_post_message_api_error(self, agent, mock_webclient):
        from slack_sdk.errors import SlackApiError

        err_response = MagicMock()
        err_response.__getitem__ = lambda self, key: "channel_not_found"
        mock_webclient.chat_postMessage.side_effect = SlackApiError(
            message="error", response=err_response
        )
        result = agent._post_message("fail")
        assert result == {"ts": None}

    def test_wait_for_reply_gets_first_reply(self, agent, mock_webclient):
        mock_webclient.conversations_replies.return_value = {
            "messages": [
                {"text": "original"},
                {"text": "user reply"},
            ]
        }
        result = agent._wait_for_reply("1234.5678")
        assert result == "user reply"

    def test_wait_for_reply_timeout(self, agent, mock_webclient):
        mock_webclient.conversations_replies.return_value = {
            "messages": [{"text": "original"}]
        }
        result = agent._wait_for_reply("1234.5678", poll_interval=1, timeout=0)
        assert result == "No feedback received (timeout)"

    def test_request_feedback_end_to_end(self, agent, mock_webclient):
        mock_webclient.chat_postMessage.return_value = {
            "ok": True,
            "message": {"ts": "111.222"},
        }
        mock_webclient.conversations_replies.return_value = {
            "messages": [
                {"text": "bot message"},
                {"text": "good job"},
            ]
        }
        result = agent.request_feedback("How did I do?")
        assert result == "good job"


class TestSlackFeedbackFn:
    def test_missing_env_vars(self):
        with patch.dict("os.environ", {}, clear=True):
            result = slack_feedback_fn(0, {}, [])
            assert result == ""

    @patch("smartmemory.feedback.slack.SlackFeedbackAgent")
    def test_with_env_vars(self, MockAgent):
        mock_agent = MagicMock()
        mock_agent.request_feedback.return_value = "looks good"
        MockAgent.return_value = mock_agent

        with patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb-t", "SLACK_CHANNEL_ID": "C1"}):
            result = slack_feedback_fn(1, {}, [{"key": "a"}])
            assert result == "looks good"
            MockAgent.assert_called_once_with("xoxb-t", "C1")

    def test_uses_provided_agent(self):
        mock_agent = MagicMock()
        mock_agent.request_feedback.return_value = "stop"

        with patch.dict("os.environ", {"SLACK_BOT_TOKEN": "t", "SLACK_CHANNEL_ID": "c"}):
            result = slack_feedback_fn(2, {}, [{"key": "b"}], agent=mock_agent)
            assert result == "stop"
            mock_agent.request_feedback.assert_called_once()
