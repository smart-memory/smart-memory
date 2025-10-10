"""
Episodic memory using unified base classes and mixins.

Migrated from original 179-line implementation to simplified version
while maintaining all functionality through unified patterns.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from smartmemory.configuration import MemoryConfig
from smartmemory.graph.types.episodic import EpisodicMemoryGraph
from smartmemory.memory.base import GraphBackedMemory
from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


class EpisodicMemory(GraphBackedMemory):
    """
    Episodic memory using unified patterns.
    
    Migrated from original 179-line implementation to simplified version
    while maintaining all functionality through unified base classes and mixins.
    """

    def __init__(self, config: MemoryConfig = None, *args, **kwargs):
        # Initialize with episodic memory type and store
        super().__init__(
            memory_type="episodic",
            config=config,
            *args, **kwargs
        )

        # Set up graph and store for proper delegation
        self.graph = EpisodicMemoryGraph()
        # Set store to graph for unified base class delegation
        self.store = self.graph

    def _add_impl(self, item: MemoryItem, **kwargs) -> Optional[MemoryItem]:
        """Episodic-specific add logic."""
        episode_data = {
            "item_id": item.item_id,
            "name": item.metadata.get("title") or item.metadata.get("description"),
            "content": item.content,
            "description": item.metadata.get("description", "Episodic event"),
            "reference_time": getattr(item, 'transaction_time', None),
            "source": "message",
            "group_id": getattr(item, "group_id", ""),
        }

        try:
            # add_episode returns the episode node directly
            episode_node = self.graph.add_episode(episode_data, **kwargs)

            # Update item with episode data
            if episode_node:
                # Ensure item_id is preserved
                if not item.item_id:
                    item.item_id = episode_node.get("item_id")
                item.group_id = episode_node.get("group_id", "")
                item.metadata["group_id"] = item.group_id
                item.metadata["reference_time"] = episode_node.get("reference_time", "")
                item.metadata["_node"] = episode_node

            return item
        except Exception as e:
            logger.error(f"Failed to add episodic item: {e}")
            return None

    def _get_impl(self, key: str) -> Optional[MemoryItem]:
        """Episodic-specific get logic."""
        try:
            node = self.graph.get_episode(key)
            if node is None:
                return None

            # Ensure we have a valid item_id
            item_id = getattr(node, "item_id", None) or key

            return MemoryItem(
                content=getattr(node, "content", ""),
                metadata={
                    "title": getattr(node, "name", ""),
                    "description": getattr(node, "description", ""),
                    "actions": getattr(node, "actions", []),
                    "entities": getattr(node, "entities", []),
                    "relations": getattr(node, "relations", []),
                    "_node": node,
                },
                item_id=item_id
            )
        except Exception as e:
            logger.error(f"Failed to get episodic item {key}: {e}")
            return None

    def _search_impl(self, query: str, top_k: int = 5, **kwargs) -> List[MemoryItem]:
        """Episodic-specific search logic."""
        try:
            episodes = self.graph.search(query, top_k=top_k, **kwargs)
            return [
                MemoryItem(
                    content=getattr(node, "content", ""),
                    metadata={
                        "title": getattr(node, "name", ""),
                        "description": getattr(node, "description", ""),
                        "actions": getattr(node, "actions", []),
                        "entities": getattr(node, "entities", []),
                        "relations": getattr(node, "relations", []),
                        "_node": node,
                    },
                    item_id=getattr(node, "item_id", None)
                )
                for node in episodes
            ]
        except Exception as e:
            logger.error(f"Failed to search episodic items: {e}")
            return []

    def _remove_impl(self, key: str) -> bool:
        """Episodic-specific remove logic."""
        try:
            result = self.graph.remove(key)
            if result:
                logger.info(f"Removed episodic item {key}")
            return result
        except Exception as e:
            logger.error(f"Failed to remove episodic item {key}: {e}")
            return False

    def get_stable_events(self, confidence: float = 0.9, min_days: int = 3) -> List[MemoryItem]:
        """Get stable events for promotion to semantic memory."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=min_days)

        try:
            all_items = self._search_impl("*", top_k=1000)
            return [
                item for item in all_items
                if (item.metadata.get('confidence', 0.0) >= confidence and
                    self._get_item_date(item) <= cutoff_date)
            ]
        except Exception as e:
            logger.error(f"Failed to get stable events: {e}")
            return []

    def get_stale_events(self, half_life: int = 30) -> List[MemoryItem]:
        """Return events with age >= half_life days."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=half_life)

        try:
            all_items = self._search_impl("*", top_k=1000)
            return [
                item for item in all_items
                if self._get_item_date(item) <= cutoff_date
            ]
        except Exception as e:
            logger.error(f"Failed to get stale events: {e}")
            return []

    def get_events_since(self, days: int = 1) -> List[MemoryItem]:
        """Return events with age <= days."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            all_items = self._search_impl("*", top_k=1000)
            return [
                item for item in all_items
                if self._get_item_date(item) >= cutoff_date
            ]
        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []

    def get_recent(self, top_k: int = 5) -> List[MemoryItem]:
        """Get most recent episodes."""
        try:
            if hasattr(self.graph, 'get_recent'):
                return self.graph.get_recent(top_k=top_k)
            else:
                # Fallback to search with sorting
                return self._search_impl("*", top_k=top_k)
        except Exception as e:
            logger.error(f"Failed to get recent episodes: {e}")
            return []

    def _get_item_date(self, item: MemoryItem) -> datetime:
        """Helper to extract datetime from item metadata."""
        created = item.metadata.get('created_at') or item.metadata.get('reference_time')
        if created:
            try:
                if isinstance(created, str):
                    return datetime.fromisoformat(created)
                return created
            except Exception:
                pass
        return datetime.now(timezone.utc)  # Default to now if no date found
