"""Response models for Claude CLI."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Usage:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Response:
    """Response from Claude CLI."""
    content: str
    model: str
    usage: Optional[Usage] = None
    stop_reason: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    raw: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return self.content


@dataclass
class StructuredResponse:
    """Structured response with parsed data."""
    data: Any
    content: str
    model: str
    success: bool = True
    error: Optional[str] = None
