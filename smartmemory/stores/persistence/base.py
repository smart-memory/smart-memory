"""
Pydantic-based persistence abstraction for smartmemory.

Simple pluggable persistence interface using Pydantic BaseModel as the foundation.
Supports dependency injection of different backends:
- JSON files (default for core library)
- MongoDB (available in platform layer)  
- PostgreSQL (future)

Usage:
```python
from pydantic import BaseModel
from smartmemory.persistence import PersistentModel

class MyModel(PersistentModel):
    name: str
    value: int

# Use with default JSON backend
models = MyModel(name="test", value=42)
models.save()

found = MyModel.find_one(name="test")
```
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Type, TypeVar

T = TypeVar('T', bound='PersistentModel')


class PersistenceBackend(ABC):
    """Abstract persistence backend interface for Pydantic models."""

    @abstractmethod
    def find_one(self, model_class: Type[T], **filters) -> Optional[T]:
        """Find a single document matching the filters."""
        pass

    @abstractmethod
    def find_many(self, model_class: Type[T], **filters) -> List[T]:
        """Find multiple documents matching the filters."""
        pass

    @abstractmethod
    def save(self, model: T) -> T:
        """Save a Pydantic models (insert or update)."""
        pass

    @abstractmethod
    def delete_one(self, model_class: Type[T], **filters) -> bool:
        """Delete a single document matching the filters."""
        pass
