import os

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


class SlackFeedbackAgent:
    """
    Interactive Slack feedback agent for agentic planning/learning.
    Usage:
        agent = SlackFeedbackAgent(token, channel_id)
        feedback = agent.request_feedback(message)
    """

    def __init__(self, token: str, channel_id: str):
        self.client = WebClient(token=token)
        self.channel_id = channel_id

    def request_feedback(self, message: str, thread_ts: str = None) -> str:
        # Feedback is handled synchronously.
        # Post message to Slack
        response = self._post_message(message, thread_ts)
        ts = response["ts"] if "ts" in response else response.get("ts")
        # Wait for a reply in the thread
        feedback = self._wait_for_reply(ts)
        return feedback

    def _post_message(self, message, thread_ts=None):
        try:
            response = self.client.chat_postMessage(
                channel=self.channel_id,
                text=message,
                thread_ts=thread_ts
            )
            return response["message"] if "message" in response else response
        except SlackApiError as e:
            print(f"Slack API error: {e.response['error']}")
            return {"ts": None}

    def _wait_for_reply(self, thread_ts, poll_interval=3, timeout=120):
        # Poll for replies in the thread
        waited = 0
        while waited < timeout:
            replies = self.client.conversations_replies(
                channel=self.channel_id, ts=thread_ts
            )
            messages = replies.get("messages", [])
            if len(messages) > 1:
                # Return the first reply after the original message
                return messages[1]["text"]
            # Polling is handled synchronously.
            waited += poll_interval
        return "No feedback received (timeout)"


# Example feedback_fn for agentic routines
def slack_feedback_fn(step, context, results, agent=None):
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel = os.environ.get("SLACK_CHANNEL_ID")
    if not token or not channel:
        print("Slack token/channel not set in environment.")
        return ""
    agent = agent or SlackFeedbackAgent(token, channel)
    msg = f"Agentic Step {step}: Retrieved {len(results)} results. Please provide feedback or say 'stop'."
    return agent.request_feedback(msg)
