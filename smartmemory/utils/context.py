"""
Context utilities for managing user context in memory operations.

Provides thread-local storage for user_id and other contextual information
that needs to be accessible throughout the memory pipeline.
"""

import contextvars
from typing import Optional

# Context variable for storing the current user ID
_user_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'user_id', default=None
)


def set_user_id(user_id: Optional[str]) -> None:
    """
    Set the current user ID in the context.
    
    Args:
        user_id: The user ID to set, or None to clear
    """
    _user_id_context.set(user_id)


def get_user_id() -> Optional[str]:
    """
    Get the current user ID from the context.
    
    Returns:
        The current user ID, or None if not set
    """
    return _user_id_context.get()


def current_user_id() -> Optional[str]:
    """
    Alias for get_user_id() for backward compatibility.
    
    Returns:
        The current user ID, or None if not set
    """
    return get_user_id()


__all__ = ['set_user_id', 'get_user_id', 'current_user_id']
