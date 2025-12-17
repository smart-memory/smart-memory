from abc import ABC, abstractmethod
from typing import Dict, List, Type, Optional, Union


class VectorBackend(ABC):
    """Backend interface for vector operations."""

    @abstractmethod
    def add(self, *, item_id: str, embedding: List[float], metadata: Dict) -> None:
        ...

    @abstractmethod
    def upsert(self, *, item_id: str, embedding: List[float], metadata: Dict) -> None:
        ...

    @abstractmethod
    def search(self, *, query_embedding: List[float], top_k: int) -> List[Dict]:
        ...

    @abstractmethod
    def search_by_text(self, *, query_text: str, top_k: int) -> List[Dict]:
        """Search using full-text index (BM25 or equivalent)."""
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


# --- Lazy backend registry and factory ---
_BACKENDS: Optional[Dict[str, Type[VectorBackend]]] = None


def _ensure_registry() -> None:
    """Initialize the backend registry with available backends."""
    global _BACKENDS
    if _BACKENDS is not None:
        return
    
    _BACKENDS = {}
    
    # FalkorDB is the only supported vector backend
    try:
        from .falkor import FalkorVectorBackend
        _BACKENDS["falkordb"] = FalkorVectorBackend
    except ImportError as e:
        raise RuntimeError(f"FalkorDB backend is required but could not be loaded: {e}")


def create_backend(name: str, collection_name: str, persist_directory: Optional[str]) -> VectorBackend:
    _ensure_registry()
    key = (name or "falkordb").lower()
    assert _BACKENDS is not None  # for type checkers
    if key not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(
            f"Unknown vector backend: '{name}'. Available backends: {available}."
        )
    cls = _BACKENDS[key]
    return cls(collection_name, persist_directory)
