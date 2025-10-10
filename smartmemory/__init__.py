"""
SmartMemory: Multi-layered AI memory system with graph databases, vector stores, and
intelligent processing pipelines for context-aware AI applications. Provides semantic,
episodic, procedural, and working memory types with advanced relationship modeling and
storage and retrieval for AI applications.
"""

from smartmemory.__version__ import __version__, __version_info__

__author__ = "SmartMemory Team"

from smartmemory.smart_memory import SmartMemory
from smartmemory.models.memory_item import MemoryItem

__all__ = [
    "SmartMemory",
    "MemoryItem",
    "__version__",
    "__version_info__",
]
