"""
Generic store backend registry (library-level).

Backends self-register by importing this module and calling register_store().
Consumers resolve a backend by key (e.g., 'mongo', 'json') and pass kwargs.
"""
from __future__ import annotations

from typing import Callable, Dict, Any

# Global registry: backend_key -> factory(kwargs) -> BaseHandler[Any]
_STORE_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_store(backend_key: str, factory: Callable[..., Any]) -> None:
    """Register a store backend factory under a string key.
    Safe to call multiple times; last registration wins.
    """
    if not backend_key or not callable(factory):
        raise ValueError("Invalid backend registration")
    _STORE_REGISTRY[backend_key] = factory


def create_store(backend_key: str, **kwargs) -> Any:
    """Create a store instance from a registered backend key."""
    if backend_key not in _STORE_REGISTRY:
        available = ", ".join(sorted(_STORE_REGISTRY.keys()))
        raise ValueError(f"Unknown store backend '{backend_key}'. Available: {available}")
    return _STORE_REGISTRY[backend_key](**kwargs)


def list_store_backends() -> Dict[str, Callable[..., Any]]:
    """Return a copy of the registered backends mapping."""
    return dict(_STORE_REGISTRY)
