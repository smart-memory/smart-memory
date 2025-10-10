import requests
from typing import Any, Dict

from smartmemory.models.memory_item import MemoryItem


class WebHandler:
    """
    Handler for web resources (HTTP/HTTPS URLs).
    All methods accept and return canonical MemoryItem objects or dicts with at least 'content' and 'item_id'.
    Supports GET, POST (add), PUT/PATCH (update), and DELETE.
    """

    def get(self, item: 'MemoryItem', **kwargs) -> Dict[str, Any]:
        """
        Retrieve content from a web resources. Accepts URI, MemoryItem, or dict.
        Returns a dict with 'content', 'item_id', and 'metadata'.
        """
        if not isinstance(item, MemoryItem):
            raise TypeError('WebHandler only accepts MemoryItem objects')
        uri = item.metadata.get('uri') or item.item_id
        item_id = item.item_id
        resp = requests.get(uri, **kwargs)
        resp.raise_for_status()
        return {'content': resp.text, 'item_id': item_id, 'metadata': {'uri': uri}}

    def add(self, item: 'MemoryItem', **kwargs) -> Dict[str, Any]:
        """
        POST to the URI with item as JSON. Accepts MemoryItem or dict.
        Returns the response as a dict.
        """
        if MemoryItem is not None and isinstance(item, MemoryItem):
            uri = item.metadata.get('uri') or item.item_id
            data = {'content': item.content, 'item_id': item.item_id, 'metadata': item.metadata}
        else:
            uri = item.get('metadata') or {}.get('uri') or item.get('item_id') or item.get('uri')
            data = item
        resp = requests.post(uri, json=data)
        resp.raise_for_status()
        return resp.json()

    def update(self, item: 'MemoryItem', **kwargs) -> Dict[str, Any]:
        """
        PUT to the URI with item as JSON. Accepts MemoryItem or dict.
        Returns the response as a dict.
        """
        if MemoryItem is not None and isinstance(item, MemoryItem):
            uri = item.metadata.get('uri') or item.item_id
            data = {'content': item.content, 'item_id': item.item_id, 'metadata': item.metadata}
        else:
            uri = item.get('metadata') or {}.get('uri') or item.get('item_id') or item.get('uri')
            data = item
        resp = requests.put(uri, json=data)
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str, item: 'MemoryItem', **kwargs) -> Dict[str, Any]:
        """
        Not universally supported; implement per API. Returns NotImplementedError.
        """
        if not isinstance(item, MemoryItem):
            raise TypeError('WebHandler only accepts MemoryItem objects')
        raise NotImplementedError('Search not supported for generic web handler')

    def delete(self, item: 'MemoryItem', **kwargs) -> int:
        """
        DELETE the resources at the URI. Accepts URI, MemoryItem, or dict.
        Returns the HTTP status code.
        """
        if not isinstance(item, MemoryItem):
            raise TypeError('WebHandler only accepts MemoryItem objects')
        uri = item.metadata.get('uri') or item.item_id
        resp = requests.delete(uri)
        resp.raise_for_status()
        return resp.status_code
