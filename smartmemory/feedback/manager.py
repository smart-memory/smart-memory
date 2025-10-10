from typing import Callable, Dict, Any, Optional


class FeedbackManager:
    """
    Unifies multiple feedback channels (CLI, Slack, Web, etc.) under a single interface.
    Usage:
        manager = FeedbackManager()
        manager.register_channel('cli', cli_feedback_fn)
        manager.register_channel('slack', slack_feedback_fn)
        feedback = manager.request_feedback('slack', step, context, results)
    """

    def __init__(self):
        self.channels: Dict[str, Callable] = {}

    def register_channel(self, name: str, fn: Callable):
        self.channels[name] = fn

    def get_channel(self, name: str) -> Optional[Callable]:
        return self.channels.get(name)

    def request_feedback(self, channel: str, *args, **kwargs) -> Any:
        fn = self.get_channel(channel)
        if not fn:
            raise ValueError(f"Feedback channel '{channel}' not registered.")
        return fn(*args, **kwargs)
