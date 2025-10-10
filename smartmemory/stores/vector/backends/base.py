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
    
    # FalkorDB is always available (core dependency)
    try:
        from .falkor import FalkorVectorBackend
        _BACKENDS["falkordb"] = FalkorVectorBackend
    except ImportError as e:
        raise RuntimeError(f"FalkorDB backend is required but could not be loaded: {e}")
    
    # ChromaDB is optional
    try:
        from .chroma import ChromaVectorBackend
        _BACKENDS["chromadb"] = ChromaVectorBackend
    except ImportError:
        # ChromaDB not installed - that's okay, it's optional
        pass


def create_backend(name: str, collection_name: str, persist_directory: Optional[str]) -> VectorBackend:
    _ensure_registry()
    key = (name or "falkordb").lower()
    assert _BACKENDS is not None  # for type checkers
    if key not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(
            f"Unknown vector backend: '{name}'. Available backends: {available}. "
            f"If you want to use ChromaDB, install it with: pip install smartmemory[chromadb]"
        )
    cls = _BACKENDS[key]
    return cls(collection_name, persist_directory)
