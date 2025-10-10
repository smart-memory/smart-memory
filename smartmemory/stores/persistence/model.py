import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Type, List

from smartmemory.models.base import MemoryBaseModel
from smartmemory.stores.persistence.base import T, PersistenceBackend
from smartmemory.stores.persistence.json import JSONFileBackend

# Module-level backend registry (avoids Pydantic field conversion)
_BACKENDS: Dict[str, PersistenceBackend] = {}
_DEFAULT_BACKEND: str = "json"


class _PersistentModelQuery:
    """Query object for Beanie-compatible find().delete() pattern."""

    def __init__(self, model_class: Type[T], filters: Dict):
        self.model_class = model_class
        self.filters = filters

    def delete(self):
        """Delete all models matching the query filters."""
        backend = self.model_class.get_backend()
        models = backend.find_many(self.model_class, **self.filters)

        deleted_count = 0
        for model in models:
            if hasattr(model, 'id') and model.id:
                success = backend.delete_one(self.model_class, id=model.id)
                if success:
                    deleted_count += 1

        # Return object with deleted_count attribute for Beanie compatibility
        class DeleteResult:
            def __init__(self, count):
                self.deleted_count = count

        return DeleteResult(deleted_count)


@dataclass
class PersistentModel(MemoryBaseModel):
    """Base dataclass models with pluggable persistence backends."""

    # Common fields for persistence
    id: Optional[str] = field(default=None, metadata={"description": "Unique identifier"})
    created_at: Optional[datetime] = field(default=None, metadata={"description": "Creation timestamp"})
    updated_at: Optional[datetime] = field(default=None, metadata={"description": "Last update timestamp"})

    @classmethod
    def set_backend(cls, backend: PersistenceBackend, name: str = "default") -> None:
        """Set the persistence backend for this models class."""
        global _BACKENDS, _DEFAULT_BACKEND
        _BACKENDS[name] = backend
        if name == "default":
            _DEFAULT_BACKEND = name

    @classmethod
    def get_backend(cls, name: Optional[str] = None) -> PersistenceBackend:
        """Get the persistence backend."""
        global _BACKENDS, _DEFAULT_BACKEND
        backend_name = name or _DEFAULT_BACKEND

        if backend_name not in _BACKENDS:
            # Default to JSON file backend
            data_dir = os.getenv("smartmemory_DATA_DIR", "data")
            _BACKENDS[backend_name] = JSONFileBackend(data_dir)

        return _BACKENDS[backend_name]

    # Sync persistence methods
    @classmethod
    def find_one(cls: Type[T], **filters) -> Optional[T]:
        """Find a single models."""
        backend = cls.get_backend()
        return backend.find_one(cls, **filters)

    @classmethod
    def find_many(cls: Type[T], **filters) -> List[T]:
        """Find multiple models."""
        backend = cls.get_backend()
        return backend.find_many(cls, **filters)

    def save(self: T) -> T:
        """Save this models."""
        backend = self.get_backend()
        return backend.save(self)

    @classmethod
    def delete_one(cls, **filters) -> bool:
        """Delete a single models."""
        backend = cls.get_backend()
        return backend.delete_one(cls, **filters)

    # Beanie-compatible interface methods
    @classmethod
    def get(cls: Type[T], entity_id: str) -> Optional[T]:
        """Get models by ID (Beanie-compatible)."""
        return cls.find_one(id=entity_id)

    @classmethod
    def count(cls, **filters) -> int:
        """Count models matching filters (Beanie-compatible)."""
        models = cls.find_many(**filters)
        return len(models)

    @classmethod
    def find(cls, **filters):
        """Return a query-like object for Beanie compatibility."""
        return _PersistentModelQuery(cls, filters)

    def delete(self) -> None:
        """Delete this models instance (Beanie-compatible)."""
        if self.id:
            self.delete_one(id=self.id)

    # Legacy method names for compatibility
    @classmethod
    def find_one_sync(cls: Type[T], **filters) -> Optional[T]:
        """Legacy alias for find_one (now sync by default)."""
        return cls.find_one(**filters)

    @classmethod
    def find_many_sync(cls: Type[T], **filters) -> List[T]:
        """Legacy alias for find_many (now sync by default)."""
        return cls.find_many(**filters)

    def save_sync(self: T) -> T:
        """Legacy alias for save (now sync by default)."""
        return self.save()
