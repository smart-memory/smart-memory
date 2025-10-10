"""
Context management for multi-tenancy abstraction.

This module provides a context variable for user_id, used to automatically scope all graph/memory operations
for multi-tenancy. Set the user_id at the entry point (API, CLI, background job) with set_user_id(user_id).
All downstream operations will be properly scoped by user_id without passing it as an argument.

Example usage:
    from smartmemory.utils.context import set_user_id
    set_user_id("alice")
    # All graph/memory operations now scoped to user_id="alice"
"""
from contextvars import ContextVar

# Holds the current user_id for multi-tenancy
current_user_id = ContextVar("current_user_id", default=None)
current_workspace_id = ContextVar("current_workspace_id", default=None)


def set_user_id(user_id):
    """Set the current user_id for the active context (request, thread, or task)."""
    current_user_id.set(user_id)


def get_user_id():
    """Get the current user_id for the active context (request, thread, or task)."""
    return current_user_id.get()


def set_workspace_id(workspace_id):
    """Set the current workspace_id for the active context (request, thread, or task)."""
    current_workspace_id.set(workspace_id)


def get_workspace_id():
    """Get the current workspace_id for the active context (request, thread, or task)."""
    return current_workspace_id.get()


def set_scope(*, user_id=None, workspace_id=None):
    """Convenience helper to set both user and workspace scope in one call."""
    if user_id is not None:
        set_user_id(user_id)
    if workspace_id is not None:
        set_workspace_id(workspace_id)
