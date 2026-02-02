"""
LinkExpansionEnricher - Expands URLs in memory items into rich graph structures.

Fetches URL content, extracts metadata (title, description, OG tags), and optionally
uses LLM for summarization and entity extraction. Creates WebResource nodes linked
to Entity child nodes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import httpx

# URL pattern - matches http/https URLs
URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')

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

    def _extract_urls(self, item, node_ids: dict[str, Any] | None) -> list[str]:
        """Extract URLs from content, then merge with extraction stage output.

        Args:
            item: The memory item (string or object with content attribute).
            node_ids: Optional dict that may contain 'urls' key.

        Returns:
            list: Deduplicated list of URLs, limited to max_urls_per_item.
        """
        # Get content from item
        if hasattr(item, "content"):
            content = item.content
        else:
            content = str(item)

        # 1. Regex extraction
        urls = set(URL_PATTERN.findall(content))

        # 2. Merge with extraction stage (if present)
        if isinstance(node_ids, dict):
            urls.update(node_ids.get("urls", []))

        # 3. Dedupe and limit
        return list(urls)[: self.config.max_urls_per_item]

    def _fetch_url(self, url: str) -> dict:
        """Fetch URL and return result dict with status.

        Args:
            url: The URL to fetch.

        Returns:
            dict: Result with status, html/error, final_url, content_type.
        """
        try:
            response = httpx.get(
                url,
                timeout=self.config.timeout_seconds,
                headers={"User-Agent": self.config.user_agent},
                follow_redirects=True,
            )
            response.raise_for_status()

            html = response.text[: self.config.max_content_length]
            return {
                "status": "success",
                "html": html,
                "final_url": str(response.url),
                "content_type": response.headers.get("content-type", ""),
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
            }

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
