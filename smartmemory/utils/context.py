"""
Context management for multi-tenancy abstraction.

This module provides context variables for tenant_id, workspace_id, team_id, and user_id,
used to automatically scope all graph/memory operations for multi-tenancy.

Preferred usage is via the `scope` context manager to ensure cleanup:
    with smartmemory.utils.context.scope(tenant_id="org_123"):
        smart_memory.add(...)
"""
from contextvars import ContextVar, Token
import contextlib
from typing import Optional, Dict, Any

# Context variables for multi-tenant scoping
current_tenant_id: ContextVar[Optional[str]] = ContextVar("current_tenant_id", default=None)
current_workspace_id: ContextVar[Optional[str]] = ContextVar("current_workspace_id", default=None)
current_team_id: ContextVar[Optional[str]] = ContextVar("current_team_id", default=None)
current_user_id: ContextVar[Optional[str]] = ContextVar("current_user_id", default=None)


@contextlib.contextmanager
def scope(*, tenant_id: Optional[str] = None, workspace_id: Optional[str] = None, team_id: Optional[str] = None, user_id: Optional[str] = None):
    """
    Context manager to safely set and reset isolation scope.
    Guarantees cleanup on exit, preventing context leakage in tests and thread pools.
    """
    tokens: Dict[ContextVar, Token] = {}
    
    if tenant_id is not None:
        tokens[current_tenant_id] = current_tenant_id.set(tenant_id)
    if workspace_id is not None:
        tokens[current_workspace_id] = current_workspace_id.set(workspace_id)
    if team_id is not None:
        tokens[current_team_id] = current_team_id.set(team_id)
    if user_id is not None:
        tokens[current_user_id] = current_user_id.set(user_id)
        
    try:
        yield
    finally:
        for var, token in tokens.items():
            var.reset(token)


# --- Raw Accessors (Use with caution, prefer `scope` context manager) ---

def set_tenant_id(tenant_id: str) -> None:
    """Set the current tenant_id. Consider using `with scope(tenant_id=...):` instead."""
    current_tenant_id.set(tenant_id)


def get_tenant_id() -> Optional[str]:
    """Get the current tenant_id."""
    return current_tenant_id.get()


def set_workspace_id(workspace_id: str) -> None:
    """Set the current workspace_id. Consider using `with scope(workspace_id=...):` instead."""
    current_workspace_id.set(workspace_id)


def get_workspace_id() -> Optional[str]:
    """Get the current workspace_id."""
    return current_workspace_id.get()


def set_team_id(team_id: str) -> None:
    """Set the current team_id. Consider using `with scope(team_id=...):` instead."""
    current_team_id.set(team_id)


def get_team_id() -> Optional[str]:
    """Get the current team_id."""
    return current_team_id.get()


def set_user_id(user_id: str) -> None:
    """Set the current user_id. Consider using `with scope(user_id=...):` instead."""
    current_user_id.set(user_id)


def get_user_id() -> Optional[str]:
    """Get the current user_id."""
    return current_user_id.get()


def set_scope(*, tenant_id=None, workspace_id=None, team_id=None, user_id=None):
    """
    Legacy helper to set scope variables globally for the current context.
    WARNING: This does not clean up automatically. Use `with scope(...):` for safety.
    """
    if tenant_id is not None:
        set_tenant_id(tenant_id)
    if workspace_id is not None:
        set_workspace_id(workspace_id)
    if team_id is not None:
        set_team_id(team_id)
    if user_id is not None:
        set_user_id(user_id)
