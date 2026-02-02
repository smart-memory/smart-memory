"""
LinkExpansionEnricher - Expands URLs in memory items into rich graph structures.

Fetches URL content, extracts metadata (title, description, OG tags), and optionally
uses LLM for summarization and entity extraction. Creates WebResource nodes linked
to Entity child nodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smartmemory.models.base import MemoryBaseModel, StageRequest


@dataclass
class LinkExpansionEnricherConfig(MemoryBaseModel):
    """Configuration for LinkExpansionEnricher."""

    # LLM settings
    enable_llm: bool = False
    model_name: str = "gpt-4o-mini"
    openai_api_key: str | None = None

    # Fetch settings
    timeout_seconds: int = 10
    max_urls_per_item: int = 5
    user_agent: str = "SmartMemory/0.2.7"

    # Content extraction
    max_content_length: int = 50000

    # Prompt template for LLM summarization
    prompt_template_key: str = "enrichers.link_expansion.prompt_template"


@dataclass
class LinkExpansionEnricherRequest(StageRequest):
    """Request object for LinkExpansionEnricher service layer."""

    enable_llm: bool = False
    model_name: str = "gpt-4o-mini"
    timeout_seconds: int = 10
    max_urls_per_item: int = 5
    context: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
