from typing import Optional, Any

from smartmemory.models.memory_item import MemoryItem


def create_memory_item(
        note: str,
        valid_start_time: Optional[Any],
        valid_end_time: Optional[Any],
        tags: Optional[list[str]],
        grounding: Optional[dict] = None
) -> Any:
    """
    Create a MemoryItem with optional tags and grounding metadata.

    Args:
        note (str): The note content.
        valid_start_time (Optional[Any]): The valid start time (datetime or str).
        valid_end_time (Optional[Any]): The valid end time (datetime or str).
        tags (Optional[list[str]]): Optional tags for the note.
        grounding (Optional[dict]): Optional grounding metadata.
    Returns:
        MemoryItem: The created memory item.
    """
    metadata = {"tags": tags} if tags else {}
    if grounding:
        metadata["grounding"] = grounding
    return MemoryItem(content=note, valid_start_time=valid_start_time, valid_end_time=valid_end_time, metadata=metadata)


def get_content_preview(items: list[Any], max_items: int) -> list[str]:
    """
    Get a preview of the content for up to max_items.

    Args:
        items (list[Any]): List of memory items.
        max_items (int): Maximum number of items to preview.
    Returns:
        list[str]: List of content previews (first 100 chars).
    """
    return [str(getattr(item, "content", getattr(item, "item", None) and getattr(item.item, "content", "")))[:100] for item in items[:max_items]]


def get_last_updated(items: list[Any]) -> Any:
    """
    Get the most recent transaction_time among the items.

    Args:
        items (list[Any]): List of memory items or Zettel objects.
    Returns:
        Any: The latest transaction_time, or None if items is empty or none have transaction_time.
    """
    if not items:
        return None
    times = []
    for item in items:
        t = getattr(item, "transaction_time", None)
        if t is None and hasattr(item, "item"):
            t = getattr(item.item, "transaction_time", None)
        if t is not None:
            times.append(t)
    if not times:
        return None
    return max(times)
