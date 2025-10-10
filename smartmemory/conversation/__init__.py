"""
Conversation package: opt-in session and context utilities for conversation-aware memory.
This package is additive and does not affect existing behavior unless used.
"""

from .context import ConversationContext
from .manager import ConversationManager
from .session import ConversationSession
