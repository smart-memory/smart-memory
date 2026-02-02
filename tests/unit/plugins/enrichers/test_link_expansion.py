"""Unit tests for LinkExpansionEnricher plugin."""


class TestLinkExpansionEnricherConfig:
    """Tests for LinkExpansionEnricherConfig dataclass."""

    def test_create_default_config(self):
        """Test creating config with default values."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricherConfig

        config = LinkExpansionEnricherConfig()

        assert config.enable_llm is False
        assert config.model_name == "gpt-4o-mini"
        assert config.timeout_seconds == 10
        assert config.max_urls_per_item == 5
        assert config.max_content_length == 50000

    def test_create_config_with_custom_values(self):
        """Test creating config with custom values."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricherConfig

        config = LinkExpansionEnricherConfig(
            enable_llm=True,
            model_name="gpt-4o",
            timeout_seconds=30,
            max_urls_per_item=10,
        )

        assert config.enable_llm is True
        assert config.model_name == "gpt-4o"
        assert config.timeout_seconds == 30
        assert config.max_urls_per_item == 10


class TestLinkExpansionEnricherMetadata:
    """Tests for LinkExpansionEnricher plugin metadata."""

    def test_metadata_returns_plugin_metadata(self):
        """Test that metadata() returns PluginMetadata."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher
        from smartmemory.plugins.base import PluginMetadata

        metadata = LinkExpansionEnricher.metadata()

        assert isinstance(metadata, PluginMetadata)
        assert metadata.name == "link_expansion_enricher"
        assert metadata.plugin_type == "enricher"
        assert metadata.requires_network is True

    def test_enricher_inherits_from_enricher_plugin(self):
        """Test that LinkExpansionEnricher inherits from EnricherPlugin."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher
        from smartmemory.plugins.base import EnricherPlugin

        assert issubclass(LinkExpansionEnricher, EnricherPlugin)

    def test_enricher_init_with_default_config(self):
        """Test creating enricher with default config."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()

        assert enricher.config.enable_llm is False
        assert enricher.config.timeout_seconds == 10

    def test_enricher_init_with_custom_config(self):
        """Test creating enricher with custom config."""
        from smartmemory.plugins.enrichers.link_expansion import (
            LinkExpansionEnricher,
            LinkExpansionEnricherConfig,
        )

        config = LinkExpansionEnricherConfig(enable_llm=True, timeout_seconds=30)
        enricher = LinkExpansionEnricher(config=config)

        assert enricher.config.enable_llm is True
        assert enricher.config.timeout_seconds == 30

    def test_enricher_init_rejects_wrong_config_type(self):
        """Test that wrong config type raises TypeError."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher
        import pytest

        with pytest.raises(TypeError, match="requires typed config"):
            LinkExpansionEnricher(config={"enable_llm": True})


class TestURLExtraction:
    """Tests for URL extraction from content."""

    def test_extract_single_url(self):
        """Test extracting a single URL from content."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        content = "Check out https://example.com for more info."

        urls = enricher._extract_urls(content, None)

        assert urls == ["https://example.com"]

    def test_extract_multiple_urls(self):
        """Test extracting multiple URLs from content."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        content = "See https://example.com and http://test.org/page for details."

        urls = enricher._extract_urls(content, None)

        assert len(urls) == 2
        assert "https://example.com" in urls
        assert "http://test.org/page" in urls

    def test_extract_urls_with_query_params(self):
        """Test extracting URLs with query parameters."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        content = "Link: https://example.com/search?q=test&page=1"

        urls = enricher._extract_urls(content, None)

        assert urls == ["https://example.com/search?q=test&page=1"]

    def test_extract_urls_deduplicates(self):
        """Test that duplicate URLs are removed."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        content = "Visit https://example.com and https://example.com again."

        urls = enricher._extract_urls(content, None)

        assert urls == ["https://example.com"]

    def test_extract_urls_respects_max_limit(self):
        """Test that URL extraction respects max_urls_per_item."""
        from smartmemory.plugins.enrichers.link_expansion import (
            LinkExpansionEnricher,
            LinkExpansionEnricherConfig,
        )

        config = LinkExpansionEnricherConfig(max_urls_per_item=2)
        enricher = LinkExpansionEnricher(config=config)
        content = "https://a.com https://b.com https://c.com https://d.com"

        urls = enricher._extract_urls(content, None)

        assert len(urls) == 2

    def test_extract_urls_merges_with_node_ids(self):
        """Test that URLs from node_ids are merged."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        content = "See https://example.com for info."
        node_ids = {"urls": ["https://other.com"]}

        urls = enricher._extract_urls(content, node_ids)

        assert len(urls) == 2
        assert "https://example.com" in urls
        assert "https://other.com" in urls

    def test_extract_urls_from_item_object(self):
        """Test extracting URLs from an item with content attribute."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        class MockItem:
            content = "Visit https://example.com today."

        enricher = LinkExpansionEnricher()
        urls = enricher._extract_urls(MockItem(), None)

        assert urls == ["https://example.com"]

    def test_extract_urls_empty_content(self):
        """Test extracting URLs from empty content."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        urls = enricher._extract_urls("No links here.", None)

        assert urls == []
