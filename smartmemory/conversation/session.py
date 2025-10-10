from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .context import ConversationContext


@dataclass
class ConversationSession:
    """
    Represents a single conversation session and holds its ConversationContext.
    This is an in-memory utility; persistence is out of scope for the first phase.
    """

    session_id: str
    user_id: Optional[str] = None
    conversation_type: str = "default"
    context: ConversationContext = field(default_factory=ConversationContext)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True

    def touch(self) -> None:
        self.last_activity = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "conversation_type": self.conversation_type,
            "context": self.context.to_dict() if hasattr(self.context, "to_dict") else {},
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_active": self.is_active,
        }
