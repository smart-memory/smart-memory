from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Single input document for extraction."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    library_id: str
    name: str
    content_type: str  # text/plain, application/pdf, text/markdown
    content_hash: str = ""  # SHA-256 of content for provenance
    storage_phase: str = "buffer"  # buffer | digest | evicted | pinned
    storage_uri: str = ""  # file:// or s3:// — empty when evicted
    evict_at: Optional[datetime] = None
    size_bytes: int = 0
    char_count: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class Library(BaseModel):
    """
    Collection of documents for batch processing.
    Represents a semantic grouping of documents (e.g. "Legal Contracts", "Q4 Reports").
    Renamed from Corpus for friendlier terminology.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    registry_id: str
    name: str
    description: Optional[str] = None
    retention_policy: str = "standard"  # ephemeral | standard | permanent
    retention_days: int = 90  # ephemeral=7, standard=90, permanent=-1
    document_count: int = 0
    total_size_bytes: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
