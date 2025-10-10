"""
Procedural memory using unified base classes and mixins.

Migrated from original 130-line implementation to simplified version
while maintaining all functionality through unified patterns.

Procedural Memory Graph Schema (with Conditional Nodes):
- Node Types: Procedure, Step, ConditionalStep
- Edge Types: HAS_STEP, NEXT_STEP, ON_TRUE, ON_FALSE, HAS_PREREQUISITE
"""

import logging
from typing import Optional, List

from smartmemory.configuration import MemoryConfig
from smartmemory.graph.types.procedural import ProceduralMemoryGraph
from smartmemory.memory.base import GraphBackedMemory
from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


class ProceduralMemory(GraphBackedMemory):
    """
    Procedural memory using unified patterns.
    
    Migrated from original 130-line implementation to simplified version
    while maintaining all functionality through unified base classes and mixins.
    """

    def __init__(self, config: MemoryConfig = None, *args, **kwargs):
        # Initialize with procedural memory type and store
        super().__init__(
            memory_type="procedural",
            config=config,
            *args, **kwargs
        )

        # Set up graph and store for proper delegation
        self.graph = ProceduralMemoryGraph()
        # Set store to graph for unified base class delegation
        self.store = self.graph

    def _add_impl(self, item: MemoryItem, **kwargs) -> Optional[MemoryItem]:
        """Procedural-specific add logic."""
        procedure_data = {
            "item_id": item.item_id,
            "name": item.metadata.get("title") or item.metadata.get("description") or item.item_id,
            "content": item.content,
            "description": item.metadata.get("description", "Procedure step"),
            "procedure_body": item.content,
            "reference_time": getattr(item, 'transaction_time', None),
            "source": "procedure",
            "group_id": getattr(item, "group_id", ""),
            "steps": item.metadata.get("steps", []),
        }

        try:
            # add returns the procedure node directly
            procedure_node = self.graph.add(procedure_data, **kwargs)

            # Update item with procedure data
            if procedure_node:
                # Ensure item_id is preserved
                if not item.item_id:
                    item.item_id = procedure_node.get("item_id")
                item.group_id = procedure_node.get("group_id", "")
                item.metadata["group_id"] = item.group_id
                item.metadata["reference_time"] = procedure_node.get("reference_time", "")
                item.metadata["_node"] = procedure_node

            logger.info(f"Added procedure {item.item_id} to graph DB.")
            return item
        except Exception as e:
            logger.error(f"Failed to add procedure {item.item_id}: {e}")
            return None

    def _get_impl(self, key: str) -> Optional[MemoryItem]:
        """Procedural-specific get logic."""
        try:
            node = self.graph.get(key)
            if node is None:
                return None

            # Ensure we have a valid item_id
            item_id = getattr(node, "item_id", None) or key

            return MemoryItem(
                content=getattr(node, "procedure_body", ""),
                metadata={
                    "title": getattr(node, "name", ""),
                    "description": getattr(node, "description", ""),
                    "steps": getattr(node, "steps", []),
                    "_node": node,
                },
                item_id=item_id
            )
        except Exception as e:
            logger.error(f"Failed to get procedural item {key}: {e}")
            return None

    def _search_impl(self, query: str, top_k: int = 5, **kwargs) -> List[MemoryItem]:
        """Procedural-specific search logic."""
        try:
            nodes = self.graph.search(query, top_k=top_k, **kwargs)
            return [
                MemoryItem(
                    content=getattr(node, "procedure_body", ""),
                    metadata={
                        "title": getattr(node, "name", ""),
                        "description": getattr(node, "description", ""),
                        "steps": getattr(node, "steps", []),
                        "_node": node,
                    },
                    item_id=getattr(node, "item_id", None)
                )
                for node in nodes
            ]
        except Exception as e:
            logger.error(f"Failed to search procedural items: {e}")
            return []

    def _remove_impl(self, key: str) -> bool:
        """Procedural-specific remove logic."""
        try:
            result = self.graph.remove(key)
            if result:
                logger.info(f"Removed procedural item {key}")
            return result
        except Exception as e:
            logger.error(f"Failed to remove procedural item {key}: {e}")
            return False

    def add_macro(self, pattern) -> bool:
        """Add a procedural macro."""
        try:
            macro_item = MemoryItem(
                content=str(pattern),
                metadata={'type': 'macro', 'pattern': pattern}
            )
            result = self.add(macro_item)
            if result:
                logger.info(f"Added procedural macro: {pattern}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to add procedural macro: {e}")
            return False

    def add_relation(self, source_id: str, target_id: str, relation: str,
                     score: float = 1.0, source_name: str = None,
                     target_name: str = None, tags: str = "") -> bool:
        """Add a relation between procedures or steps."""
        try:
            if hasattr(self.graph, 'add_relation'):
                self.graph.add_relation(source_id, target_id, relation, score, source_name, target_name, tags)
                logger.info(f"Added relation {relation} between {source_id} and {target_id}.")
                return True
            else:
                logger.warning("Graph store does not support add_relation")
                return False
        except Exception as e:
            logger.error(f"Failed to add relation {relation} between {source_id} and {target_id}: {e}")
            return False

    def get_procedure_steps(self, key: str) -> List:
        """Get steps for a specific procedure."""
        try:
            if hasattr(self.graph, 'get_procedure_steps'):
                return self.graph.get_procedure_steps(key)
            else:
                # Fallback: get procedure and extract steps from metadata
                item = self.get(key)
                return item.metadata.get('steps', []) if item else []
        except Exception as e:
            logger.error(f"Failed to get steps for procedure {key}: {e}")
            return []

    def get_related_procedures(self, key: str, depth: int = 1) -> List:
        """Get procedures related to the given procedure."""
        try:
            if hasattr(self.graph, 'get_related_procedures'):
                return self.graph.get_related_procedures(key, depth)
            else:
                # Fallback: search for similar procedures
                item = self.get(key)
                if item:
                    return self._search_impl(item.content, top_k=5)
                return []
        except Exception as e:
            logger.error(f"Failed to traverse from procedure {key}: {e}")
            return []
