from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .context import ConversationContext
from .session import ConversationSession


class ConversationManager:
    """
    In-memory conversation session manager.

    - Additive and optional; does not persist sessions.
    - Thread-safe access to a simple session registry.
    - All field names are snake_case.
    """

    def __init__(self):
        self._sessions: Dict[str, ConversationSession] = {}
        self._lock = threading.RLock()

    def create_session(
            self,
            user_id: Optional[str] = None,
            conversation_type: str = "default",
            session_id: Optional[str] = None,
            context: Optional[ConversationContext] = None,
    ) -> ConversationSession:
        """
        Create a new conversation session with a fresh ConversationContext (unless provided).
        """
        sid = session_id or str(uuid.uuid4())
        ctx = context or ConversationContext(conversation_id=sid, user_id=user_id)
        session = ConversationSession(
            session_id=sid,
            user_id=user_id,
            conversation_type=conversation_type,
            context=ctx,
        )
        with self._lock:
            self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        if not session_id:
            raise ValueError("session_id is required")
        with self._lock:
            return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> bool:
        if not session_id:
            raise ValueError("session_id is required")
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return False
            sess.is_active = False
            sess.last_activity = datetime.now(timezone.utc)
            # Keep it for inspection; callers may explicitly delete if needed
            return True

    def delete_session(self, session_id: str) -> bool:
        if not session_id:
            raise ValueError("session_id is required")
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def cleanup_inactive_sessions(self, timeout_hours: int = 24) -> List[str]:
        """
        Close sessions that have had no activity for `timeout_hours`.
        Returns a list of session_ids that were marked inactive (not deleted).
        """
        if timeout_hours <= 0:
            raise ValueError("timeout_hours must be positive")
        cutoff = datetime.now(timezone.utc) - timedelta(hours=timeout_hours)
        updated: List[str] = []
        with self._lock:
            for sid, sess in list(self._sessions.items()):
                if sess.is_active and sess.last_activity < cutoff:
                    sess.is_active = False
                    updated.append(sid)
        return updated

    def list_active_sessions(self) -> List[str]:
        with self._lock:
            return [sid for sid, s in self._sessions.items() if s.is_active]

    def touch(self, session_id: str) -> None:
        if not session_id:
            raise ValueError("session_id is required")
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError(f"session not found: {session_id}")
            sess.touch()
