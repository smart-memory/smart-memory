"""
JSON-based store implementing BaseHandler[Dict] with self-configuration.
Demonstrates the store self-configuration pattern for different use cases.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from smartmemory.stores.base import BaseHandler
from smartmemory.stores.registry import register_store


class JSONStore(BaseHandler[Dict]):
    """JSON file-based store implementing BaseHandler[Dict] with self-configuration."""

    def __init__(self, data_dir: str = "data", optimize_for: str = "general"):
        """Initialize JSON store with self-configuration based on use case.
        
        Args:
            data_dir: Directory for JSON files
            optimize_for: Use case to optimize for ('episodic', 'semantic', 'ontology', 'general')
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.optimize_for = optimize_for

        # Self-configure based on intended use case
        self._setup_optimizations()

        # Load existing data
        self._items: Dict[str, Dict] = self._load_all_items()

    def _setup_optimizations(self):
        """Configure store optimizations based on use case."""
        if self.optimize_for == "episodic":
            self._setup_episodic_optimizations()
        elif self.optimize_for == "semantic":
            self._setup_semantic_optimizations()
        elif self.optimize_for == "ontology":
            self._setup_ontology_optimizations()
        else:
            self._setup_general_optimizations()

    def _setup_episodic_optimizations(self):
        """Optimizations for episodic memory: temporal indexing, automatic timestamps."""
        self.auto_timestamp = True
        self.sort_by_time = True
        self.backup_frequency = "daily"  # More frequent backups for episodic
        self._ensure_temporal_index()

    def _setup_semantic_optimizations(self):
        """Optimizations for semantic memory: content indexing, deduplication."""
        self.auto_timestamp = False
        self.enable_deduplication = True
        self.content_indexing = True
        self._ensure_content_index()

    def _setup_ontology_optimizations(self):
        """Optimizations for ontology: relationship tracking, validation."""
        self.auto_timestamp = True
        self.validate_relationships = True
        self.track_versions = True
        self._ensure_relationship_index()

    def _setup_general_optimizations(self):
        """General optimizations: balanced performance."""
        self.auto_timestamp = True
        self.sort_by_time = False
        self.backup_frequency = "weekly"

    def _ensure_temporal_index(self):
        """Create temporal index for episodic queries (JSON implementation uses sorting)."""
        # For JSON, we'll maintain items sorted by timestamp
        # In a real DB, this would create a temporal index
        pass

    def _ensure_content_index(self):
        """Create content index for semantic searches."""
        # For JSON, we'll maintain a content lookup
        # In a real DB, this would create a text search index
        pass

    def _ensure_relationship_index(self):
        """Create relationship index for ontology navigation."""
        # For JSON, we'll maintain relationship mappings
        # In a real DB, this would create graph-style indexes
        pass

    def _get_file_path(self) -> Path:
        """Get the main data file path."""
        return self.data_dir / f"{self.optimize_for}_store.json"

    def _load_all_items(self) -> Dict[str, Dict]:
        """Load all items from JSON file."""
        file_path = self._get_file_path()
        if not file_path.exists():
            return {}

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data.get('items', {})
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_all_items(self):
        """Save all items to JSON file."""
        file_path = self._get_file_path()

        # Prepare data structure
        data = {
            'items': self._items,
            'metadata': {
                'optimize_for': self.optimize_for,
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'item_count': len(self._items)
            }
        }

        # Sort items if temporal optimization is enabled
        if getattr(self, 'sort_by_time', False):
            # Sort items by timestamp for temporal queries
            sorted_items = dict(sorted(
                self._items.items(),
                key=lambda x: x[1].get('timestamp', x[1].get('created_at', '')),
                reverse=True
            ))
            data['items'] = sorted_items

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def add(self, item: Dict, **kwargs) -> Union[str, Dict, None]:
        """Add a dict item to the store."""
        if not item:
            return None

        # Generate ID if not provided
        item_id = item.get('id') or kwargs.get('item_id') or str(uuid.uuid4())
        item_copy = dict(item)
        item_copy['id'] = item_id

        # Apply auto-timestamping if enabled
        if getattr(self, 'auto_timestamp', False):
            now = datetime.now(timezone.utc).isoformat()
            if 'created_at' not in item_copy:
                item_copy['created_at'] = now
            item_copy['updated_at'] = now

        # Apply deduplication if enabled
        if getattr(self, 'enable_deduplication', False):
            existing_id = self._find_duplicate(item_copy)
            if existing_id:
                return existing_id  # Return ID of existing duplicate

        # Store the item
        self._items[item_id] = item_copy
        self._save_all_items()

        return item_id

    def get(self, item_id: str, **kwargs) -> Optional[Dict]:
        """Get a dict item from the store."""
        if not item_id:
            return None
        return self._items.get(item_id)

    def update(self, item: Dict, **kwargs) -> Union[bool, Dict]:
        """Update a dict item in the store."""
        if not item:
            return False

    @staticmethod
    def _match_filters(item: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Match item against simple dict filters with $in support.

        Examples:
          {"workspace_id": "acme"}
          {"tags": {"$in": ["default", "ontology"]}}
        """
        for key, val in (filters or {}).items():
            if isinstance(val, dict) and "$in" in val:
                candidates = set(val.get("$in") or [])
                field_val = item.get(key)
                if isinstance(field_val, list):
                    if not (candidates & set(field_val)):
                        return False
                else:
                    if field_val not in candidates:
                        return False
            else:
                if item.get(key) != val:
                    return False
        return True

        item_id = item.get('id')
        if not item_id:
            return False

        if item_id not in self._items:
            return False

        # Apply auto-timestamping if enabled
        item_copy = dict(item)
        if getattr(self, 'auto_timestamp', False):
            item_copy['updated_at'] = datetime.now(timezone.utc).isoformat()

        self._items[item_id] = item_copy
        self._save_all_items()

        return item_copy

    def delete(self, item_id: str, **kwargs) -> bool:
        """Delete a dict item from the store."""
        if not item_id or item_id not in self._items:
            return False

        del self._items[item_id]
        self._save_all_items()
        return True

    def search(self, query: Any, **kwargs) -> List[Dict]:
        """Search for dict items in the store.

        Supports two modes:
        - Text search (existing behavior) when query is a non-dict
        - Field filter search when query is a dict, with operators (currently supports $in)

        Kwargs supported:
        - limit/top_k: int
        - skip: int
        - sort: tuple[str, int] where int < 0 for desc
        """
        # If dict filters provided, use filter-based search
        if isinstance(query, dict):
            filters: Dict[str, Any] = query
            results = [item for item in self._items.values() if self._match_filters(item, filters)]
        else:
            # Text search (backward compatible)
            if not query:
                results = list(self._items.values())
            else:
                query_str = str(query).lower()
                results = []
                for item in self._items.values():
                    if self._matches_query(item, query_str):
                        results.append(item)

        # Sorting
        sort = kwargs.get('sort')
        if sort and isinstance(sort, (list, tuple)) and len(sort) == 2:
            key, direction = sort
            reverse = bool(direction) and int(direction) < 0
            results.sort(key=lambda x: x.get(key), reverse=reverse)
        elif getattr(self, 'sort_by_time', False):
            # Fallback to temporal sort if enabled
            results.sort(key=lambda x: x.get('timestamp', x.get('created_at', '')), reverse=True)

        # Pagination
        skip = int(kwargs.get('skip', 0) or 0)
        limit = kwargs.get('limit', kwargs.get('top_k'))
        if limit is not None:
            limit = int(limit)
            results = results[skip: skip + limit]
        elif skip:
            results = results[skip:]

        return results

    def clear(self, **kwargs) -> bool:
        """Clear items from the store. If filters provided, clear only matching items.

        Kwargs:
          - filters: dict of equality-based filters; supports $in operator for list fields
        """
        filters: Optional[Dict[str, Any]] = kwargs.get('filters')
        if not filters:
            self._items.clear()
            self._save_all_items()
            return True

        # Filtered clear
        remaining: Dict[str, Dict] = {}
        for item_id, item in self._items.items():
            if not self._match_filters(item, filters):
                remaining[item_id] = item
        self._items = remaining
        self._save_all_items()
        return True

    def _matches_query(self, item: Dict, query_str: str) -> bool:
        """Check if item matches search query."""
        # Search in content field
        content = str(item.get('content', '')).lower()
        if query_str in content:
            return True

        # Enhanced content indexing for semantic optimization
        if getattr(self, 'content_indexing', False):
            # Search in additional text fields
            for field in ['title', 'description', 'tags']:
                field_value = str(item.get(field, '')).lower()
                if query_str in field_value:
                    return True

        return False

    def _find_duplicate(self, item: Dict) -> Optional[str]:
        """Find duplicate item based on content (for deduplication)."""
        content = item.get('content', '')
        if not content:
            return None

        for item_id, existing_item in self._items.items():
            if existing_item.get('content') == content:
                return item_id

        return None

    # Tech-specific methods beyond BaseHandler interface
    def backup_to_file(self, backup_path: str) -> bool:
        """JSON-specific: backup data to another file."""
        try:
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)

            with open(backup_file, 'w') as f:
                json.dump({
                    'items': self._items,
                    'metadata': {
                        'optimize_for': self.optimize_for,
                        'backup_timestamp': datetime.now(timezone.utc).isoformat(),
                        'original_file': str(self._get_file_path())
                    }
                }, f, indent=2, default=str)

            return True
        except Exception:
            return False

    def compact_storage(self) -> bool:
        """JSON-specific: remove deleted items and optimize file size."""
        # For JSON, this just rewrites the file (removes any fragmentation)
        try:
            self._save_all_items()
            return True
        except Exception:
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """JSON-specific: get store statistics."""
        return {
            'item_count': len(self._items),
            'optimize_for': self.optimize_for,
            'file_path': str(self._get_file_path()),
            'file_size_bytes': self._get_file_path().stat().st_size if self._get_file_path().exists() else 0,
            'auto_timestamp': getattr(self, 'auto_timestamp', False),
            'features': {
                'deduplication': getattr(self, 'enable_deduplication', False),
                'content_indexing': getattr(self, 'content_indexing', False),
                'temporal_sorting': getattr(self, 'sort_by_time', False)
            }
        }


# Self-register default JSON backend so consumers can resolve by key "json"
try:
    register_store(
        "json",
        lambda **kwargs: JSONStore(
            data_dir=kwargs.get("data_dir", "data"),
            optimize_for=kwargs.get("optimize_for", "prompts"),
        ),
    )
except Exception:
    # Avoid hard failure if registry isn't available at import time
    pass
