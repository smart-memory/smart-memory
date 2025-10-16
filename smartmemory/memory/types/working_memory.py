"""
Working memory using unified base classes and mixins.

Migrated from original 107-line implementation to simplified version
while maintaining all functionality through unified patterns.
"""

from collections import deque
from typing import List, Optional

from smartmemory.configuration import MemoryConfig
from smartmemory.memory.base import MemoryBase
from smartmemory.models.memory_item import MemoryItem


class WorkingMemory(MemoryBase):
    """
    Working memory using unified patterns.
    
    Migrated from original 107-line implementation to simplified version
    while maintaining all functionality through unified base classes and mixins.
    
    Uses in-memory buffer instead of persistent storage for short-term context.
    """

    def __init__(self, capacity: int = 10, config: MemoryConfig = None, *args, **kwargs):
        # Initialize with working memory type and store
        super().__init__(
            memory_type="working",
            config=config,
            *args, **kwargs
        )

        # Working memory specific setup - uses buffer instead of graph store
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        # Set store to self for unified base class delegation (buffer-based)
        self.store = self

    def _add_impl(self, item: MemoryItem, **kwargs) -> Optional[MemoryItem]:
        """Working memory specific add logic - uses buffer instead of persistent storage."""
        try:
            self.buffer.append(item)
            return item
        except Exception as e:
            self.logger.error(f"Failed to add to working memory: {e}")
            return None

    def _get_impl(self, key: str) -> Optional[MemoryItem]:
        """Working memory specific get logic - searches buffer."""
        try:
            for item in self.buffer:
                if getattr(item, 'item_id', None) == key:
                    return item
            return None
        except Exception as e:
            self.logger.error(f"Failed to get from working memory: {e}")
            return None

    def search(self, query: str, **kwargs) -> List[MemoryItem]:
        """Search method for unified base class delegation."""
        return self._search_impl(query, **kwargs)

    def _search_impl(self, query: str, **kwargs) -> List[MemoryItem]:
        """Working memory specific search logic - searches buffer."""
        try:
            top_k = kwargs.get('top_k', 5)
            results = []

            for item in self.buffer:
                if query == "*" or query == "" or query.lower() in str(item.content).lower():
                    results.append(item)

            return results[:top_k]
        except Exception as e:
            self.logger.error(f"Failed to search working memory: {e}")
            return []

    def _remove_impl(self, key: str) -> bool:
        """Working memory specific remove logic - removes from buffer."""
        try:
            for i, item in enumerate(self.buffer):
                if getattr(item, 'item_id', None) == key:
                    del self.buffer[i]
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to remove from working memory: {e}")
            return False

    def _clear_impl(self):
        """Working memory specific clear logic - clears buffer."""
        try:
            self.buffer.clear()
        except Exception as e:
            self.logger.error(f"Failed to clear working memory: {e}")

    def get_buffer(self) -> List[MemoryItem]:
        """Return all items in working memory buffer as a list."""
        return list(self.buffer)

    def summarize_buffer(self) -> MemoryItem:
        """Summarize buffer contents into a single MemoryItem."""
        if not self.buffer:
            raise ValueError("Buffer is empty, nothing to summarize.")
        content = "\n".join(str(item.content) for item in self.buffer)
        return MemoryItem(content=content, metadata={"summarized": True})

    def clear_buffer(self, archive_reason: str = "buffer_cleared"):
        """Clear the working memory buffer by archiving items (soft delete)."""
        from datetime import datetime, timezone
        
        # Archive each item instead of hard deleting
        for item in self.buffer:
            item.metadata['archived'] = True
            item.metadata['archive_reason'] = archive_reason
            item.metadata['archive_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            # Update the item in storage if available
            if hasattr(self, 'update') and callable(self.update):
                self.update(item)
        
        # Clear the in-memory buffer
        self.buffer.clear()

    def detect_skill_patterns(self, min_count: int = 5) -> List[str]:
        """Detect repeated skill/tool patterns in buffer."""
        from collections import Counter
        skills = []
        tools = []
        for item in self.buffer:
            skills.extend(item.metadata.get('skills', []))
            tools.extend(item.metadata.get('tools', []))
        skill_counts = Counter(skills)
        tool_counts = Counter(tools)
        patterns = [s for s, c in skill_counts.items() if c >= min_count]
        patterns += [t for t, c in tool_counts.items() if c >= min_count]
        return patterns

    def search_by_embedding(self, query_embedding, top_k: int = 5) -> List[MemoryItem]:
        """Return top_k items by embedding similarity."""
        try:
            scored = []
            for item in self.buffer:
                emb = item.embedding
                if emb is None:
                    emb = MemoryItem.text_to_dummy_embedding(str(item.content))
                score = MemoryItem.cosine_similarity(query_embedding, emb) if emb else 0.0
                scored.append((score, item))
            scored.sort(reverse=True, key=lambda x: x[0])
            return [item for score, item in scored[:top_k]]
        except Exception as e:
            self.logger.error(f"Failed embedding search in working memory: {e}")
            return []

    def prune_by_relevance(self, query_embedding, threshold: float = 0.35) -> List[MemoryItem]:
        """Remove items with embedding similarity below threshold."""
        try:
            keep = []
            removed = []
            for item in self.buffer:
                emb = item.embedding
                if emb is None:
                    emb = MemoryItem.text_to_dummy_embedding(str(item.content))
                score = MemoryItem.cosine_similarity(query_embedding, emb) if emb else 0.0
                if score >= threshold:
                    keep.append(item)
                else:
                    removed.append(item)
            self.buffer = type(self.buffer)(keep, maxlen=self.capacity)
            return removed
        except Exception as e:
            self.logger.error(f"Failed to prune working memory: {e}")
            return []

    def __len__(self) -> int:
        """Return the number of items in working memory."""
        return len(self.buffer)
