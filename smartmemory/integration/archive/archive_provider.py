"""
ArchiveProvider interface and registry (design-first) for durable conversation transcript storage.

Pattern mirrors PromptProvider: ABC + set/get registry and a default no-op provider.
Core remains dependency-free; services inject concrete implementations (e.g., S3/FS/DB).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class ArchiveProvider(ABC):
    """Abstract archive provider for storing and retrieving transcripts/artifacts."""

    @abstractmethod
    def put(self, conversation_id: str, payload: Union[bytes, Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Persist an artifact durably and return a resolvable URI and content hash.
        Returns: { "archive_uri": str, "content_hash": str }
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, archive_uri: str) -> Union[bytes, Dict[str, Any]]:
        """Retrieve an artifact by its URI."""
        raise NotImplementedError


class NoOpArchiveProvider(ArchiveProvider):
    """Default provider that fails fast if archiving is attempted without configuration."""

    def put(self, conversation_id: str, payload: Union[bytes, Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, str]:
        msg = "ArchiveProvider not configured; cannot archive transcript. Set a provider via set_archive_provider(...)"
        logger.error(msg)
        raise RuntimeError(msg)

    def get(self, archive_uri: str) -> Union[bytes, Dict[str, Any]]:
        msg = "ArchiveProvider not configured; cannot retrieve archived artifact. Set a provider via set_archive_provider(...)"
        logger.error(msg)
        raise RuntimeError(msg)


_archive_provider: Optional[ArchiveProvider] = None


def set_archive_provider(provider: ArchiveProvider) -> None:
    global _archive_provider
    _archive_provider = provider


def get_archive_provider() -> ArchiveProvider:
    global _archive_provider
    if _archive_provider is None:
        _archive_provider = NoOpArchiveProvider()
    return _archive_provider
