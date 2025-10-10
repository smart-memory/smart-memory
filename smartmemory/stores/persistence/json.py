import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Type, List, Dict, Any, Optional

import json
from smartmemory.stores.persistence.base import T, PersistenceBackend


class JSONFileBackend(PersistenceBackend):
    """JSON file-based persistence backend for Pydantic models."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def _get_file_path(self, model_class: Type[T]) -> Path:
        """Get file path for a models class."""
        collection_name = getattr(model_class, '__collection__', model_class.__name__.lower())
        return self.data_dir / f"{collection_name}.json"

    def _load_models(self, model_class: Type[T]) -> List[T]:
        """Load Pydantic models from JSON file."""
        file_path = self._get_file_path(model_class)
        if not file_path.exists():
            return []

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return [model_class.from_dict(item) for item in data]  # All models are AgenticBaseModel
        except (json.JSONDecodeError, IOError):
            return []

    def _save_models(self, model_class: Type[T], models: List[T]) -> None:
        """Save AgenticBaseModel dataclasses to JSON file."""
        file_path = self._get_file_path(model_class)
        data = [model.to_dict() for model in models]  # All models are AgenticBaseModel
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _matches_filters(self, model: T, filters: Dict[str, Any]) -> bool:
        """Check if a models matches the given filters."""
        for key, value in filters.items():
            if not hasattr(model, key) or getattr(model, key) != value:
                return False
        return True

    def find_one(self, model_class: Type[T], **filters) -> Optional[T]:
        """Find a single models matching the filters."""
        models = self._load_models(model_class)
        for model in models:
            if self._matches_filters(model, filters):
                return model
        return None

    def find_many(self, model_class: Type[T], **filters) -> List[T]:
        """Find multiple models matching the filters."""
        models = self._load_models(model_class)
        if not filters:
            return models
        return [model for model in models if self._matches_filters(model, filters)]

    def save(self, model: T) -> T:
        """Save a Pydantic models (insert or update)."""
        model_class = model.__class__
        models = self._load_models(model_class)

        # Auto-generate ID if not present
        if not hasattr(model, 'id') or not model.id:
            model.id = str(uuid.uuid4())

        # Update timestamps
        now = datetime.now(timezone.utc)
        if hasattr(model, 'created_at') and not model.created_at:
            model.created_at = now
        if hasattr(model, 'updated_at'):
            model.updated_at = now

        # Find existing models and update, or append new
        found = False
        for i, existing in enumerate(models):
            if hasattr(existing, 'id') and existing.id == model.id:
                models[i] = model
                found = True
                break

        if not found:
            models.append(model)

        self._save_models(model_class, models)
        return model

    def delete_one(self, model_class: Type[T], **filters) -> bool:
        """Delete a single models matching the filters."""
        models = self._load_models(model_class)

        for i, model in enumerate(models):
            if self._matches_filters(model, filters):
                del models[i]
                self._save_models(model_class, models)
                return True

        return False
