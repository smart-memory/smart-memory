from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Single input document for extraction."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    corpus_id: str
    name: str
    content_type: str  # text/plain, application/pdf, text/markdown
    storage_uri: str   # file:// or s3://
    size_bytes: int
    char_count: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class Corpus(BaseModel):
    """
    Collection of documents for batch processing.
    Represents a semantic grouping of documents (e.g. "Legal Contracts", "Q4 Reports").
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    registry_id: str
    name: str
    description: Optional[str] = None
    document_count: int = 0
    total_size_bytes: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
