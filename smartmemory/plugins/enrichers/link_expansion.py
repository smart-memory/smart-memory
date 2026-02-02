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
from smartmemory.plugins.base import EnricherPlugin, PluginMetadata


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


class LinkExpansionEnricher(EnricherPlugin):
    """
    Enricher that expands URLs in memory items into rich graph structures.

    Fetches URL content, extracts metadata (title, description, OG tags),
    and optionally uses LLM for summarization and entity extraction.
    Creates WebResource nodes linked to Entity child nodes via MENTIONS edges.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="link_expansion_enricher",
            version="1.0.0",
            author="SmartMemory Team",
            description="Expands URLs into rich graph structures with metadata and entities",
            plugin_type="enricher",
            dependencies=["httpx>=0.24.0", "beautifulsoup4>=4.12.0"],
            min_smartmemory_version="0.2.7",
            requires_network=True,
            requires_llm=False,  # Optional - only when enable_llm=True
        )

    def __init__(self, config: LinkExpansionEnricherConfig | None = None):
        """Initialize the enricher with optional configuration.

        Args:
            config: Configuration for the enricher. If None, uses defaults.

        Raises:
            TypeError: If config is provided but not a LinkExpansionEnricherConfig.
        """
        self.config = config or LinkExpansionEnricherConfig()
        if not isinstance(self.config, LinkExpansionEnricherConfig):
            raise TypeError(
                "LinkExpansionEnricher requires typed config (LinkExpansionEnricherConfig)"
            )

    def enrich(self, item, node_ids=None) -> dict:
        """Enrich a memory item by expanding URLs.

        Args:
            item: The memory item to enrich.
            node_ids: Optional dict of node IDs from extraction stage.

        Returns:
            dict: Enrichment results with web_resources, provenance_candidates, tags.
        """
        # Placeholder - will be implemented in subsequent tasks
        return {
            "web_resources": [],
            "provenance_candidates": [],
            "tags": [],
        }
