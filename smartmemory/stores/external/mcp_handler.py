from typing import List, Union, Optional, Dict, Any

try:
    from smartmemory.models.memory_item import MemoryItem
    from smartmemory.smart_memory import SmartMemory
except ImportError:
    MemoryItem = None
    SmartMemory = None


class MCPHandler:
    """
    Handler for MCP (Model Context Protocol) integration with SmartMemory.
    All methods accept and return canonical MemoryItem objects.
    Provides a bridge between MCP protocol and SmartMemory operations.
    """

    def __init__(self):
        """Initialize MCP handler with SmartMemory instance."""
        if SmartMemory is None:
            raise ImportError("SmartMemory not available")
        self.memory = SmartMemory()

    def get(self, item: Union[str, 'MemoryItem'], **kwargs) -> Optional['MemoryItem']:
        """
        Retrieve a MemoryItem from SmartMemory.
        Args:
            item: item_id (str) or MemoryItem with item_id
        Returns:
            MemoryItem or None if not found
        """
        if MemoryItem is None:
            raise ImportError("MemoryItem not available")

        # Extract item_id
        if isinstance(item, str):
            item_id = item
        elif isinstance(item, MemoryItem):
            item_id = item.item_id
        else:
            raise TypeError('MCPHandler.get accepts item_id (str) or MemoryItem')

        try:
            return self.memory.get(item_id)
        except Exception as e:
            print(f"MCPHandler.get error: {e}")
            return None

    def add(self, item: 'MemoryItem', **kwargs) -> str:
        """
        Add a MemoryItem to SmartMemory.
        Args:
            item: MemoryItem to add
        Returns:
            item_id of added item
        """
        if MemoryItem is None:
            raise ImportError("MemoryItem not available")

        if not isinstance(item, MemoryItem):
            raise TypeError('MCPHandler.add only accepts MemoryItem objects')

        try:
            return self.memory.add(item, **kwargs)
        except Exception as e:
            raise RuntimeError(f"MCPHandler.add failed: {e}")

    def update(self, item: 'MemoryItem', **kwargs) -> None:
        """
        Update a MemoryItem in SmartMemory.
        Args:
            item: MemoryItem with updated content
        """
        if MemoryItem is None:
            raise ImportError("MemoryItem not available")

        if not isinstance(item, MemoryItem):
            raise TypeError('MCPHandler.update only accepts MemoryItem objects')

        try:
            self.memory.update(item)
        except Exception as e:
            raise RuntimeError(f"MCPHandler.update failed: {e}")

    def search(self, query: str, top_k: int = 5, **kwargs) -> List['MemoryItem']:
        """
        Search for MemoryItems in SmartMemory.
        Args:
            query: search query string
            top_k: maximum number of results
        Returns:
            List of matching MemoryItems
        """
        if MemoryItem is None:
            raise ImportError("MemoryItem not available")

        try:
            return self.memory.search(query, top_k=top_k)
        except Exception as e:
            print(f"MCPHandler.search error: {e}")
            return []

    def delete(self, item: Union[str, 'MemoryItem'], **kwargs) -> None:
        """
        Delete a MemoryItem from SmartMemory.
        Args:
            item: item_id (str) or MemoryItem with item_id
        """
        if MemoryItem is None:
            raise ImportError("MemoryItem not available")

        # Extract item_id
        if isinstance(item, str):
            item_id = item
        elif isinstance(item, MemoryItem):
            item_id = item.item_id
        else:
            raise TypeError('MCPHandler.delete accepts item_id (str) or MemoryItem')

        try:
            self.memory.delete(item_id)
        except Exception as e:
            raise RuntimeError(f"MCPHandler.delete failed: {e}")

    def ingest(self, content: str, extractor_name: str = "spacy", **kwargs) -> Dict[str, Any]:
        """
        Ingest content with full extraction pipeline via SmartMemory.
        
        Args:
            content: Raw content string to ingest
            extractor_name: Extraction method to use
            **kwargs: Additional arguments for ingestion
            
        Returns:
            Ingestion result dict with MemoryItem and extracted data
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Content must be a non-empty string")

        try:
            return self.memory.ingest(content, extractor_name=extractor_name, **kwargs)
        except Exception as e:
            raise RuntimeError(f"MCPHandler.ingest failed: {e}")
