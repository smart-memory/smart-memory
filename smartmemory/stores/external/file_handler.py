import os
from typing import Any, Dict, List

from smartmemory.models.memory_item import MemoryItem
from smartmemory.stores.base import BaseHandler


class FileHandler(BaseHandler):
    """
    Handler for file resources (local file paths).
    All methods accept and return canonical MemoryItem objects or dicts with at least 'content' and 'item_id'.
    Supports reading, writing, updating, searching, and deleting files.
    """

    def get(self, item: 'MemoryItem', **kwargs) -> 'MemoryItem':
        """
        Retrieve file content as a MemoryItem/dict. Accepts file path, MemoryItem, or dict.
        """
        if not isinstance(item, MemoryItem):
            raise TypeError('FileHandler only accepts MemoryItem objects')
        path = item.item_id or item.metadata.get('path')
        item_id = item.item_id
        with open(path, 'r', encoding=kwargs.get('encoding', 'utf-8')) as f:
            content = f.read()
        return {'content': content, 'item_id': item_id, 'metadata': {'path': path}}

    def add(self, item: 'MemoryItem', **kwargs) -> str:
        """
        Add a file. Accepts MemoryItem or dict with 'content' and 'item_id' (or 'path').
        Returns the file path (item_id).
        """
        if not isinstance(item, MemoryItem):
            raise TypeError('FileHandler only accepts MemoryItem objects')
        path = item.item_id or item.metadata.get('path')
        content = item.content
        with open(path, 'w', encoding=kwargs.get('encoding', 'utf-8')) as f:
            f.write(content)
        return path

    def update(self, item: 'MemoryItem', **kwargs) -> str:
        """
        Update a file. Accepts MemoryItem or dict with 'content' and 'item_id' (or 'path').
        Returns the file path (item_id).
        """
        if not isinstance(item, MemoryItem):
            raise TypeError('FileHandler only accepts MemoryItem objects')
        return self.add(item, **kwargs)

    def search(self, query: str, item: 'MemoryItem', **kwargs) -> List[Dict[str, Any]]:
        """
        Search for lines containing the query in the file. Returns list of dicts with 'line' and 'content'.
        """
        if isinstance(item, str):
            path = item
        elif MemoryItem is not None and isinstance(item, MemoryItem):
            path = item.item_id or item.metadata.get('path')
        else:
            path = item.get('item_id') or item.get('path')
        results = []
        with open(path, 'r', encoding=kwargs.get('encoding', 'utf-8')) as f:
            for i, line in enumerate(f):
                if query in line:
                    results.append({'line': i + 1, 'content': line.strip()})
        return results

    def delete(self, item: 'MemoryItem', **kwargs) -> bool:
        """
        Delete a file. Accepts file path, MemoryItem, or dict.
        Returns True on success.
        """
        if isinstance(item, str):
            path = item
        elif MemoryItem is not None and isinstance(item, MemoryItem):
            path = item.item_id or item.metadata.get('path')
        else:
            path = item.get('item_id') or item.get('path')
        os.remove(path)
        return True
