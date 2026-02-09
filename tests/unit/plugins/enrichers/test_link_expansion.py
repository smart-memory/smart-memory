"""Unit tests for LinkExpansionEnricher plugin."""
import pytest

pytestmark = pytest.mark.unit


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


class TestURLFetching:
    """Tests for URL fetching functionality."""

    def test_fetch_url_success(self):
        """Test successful URL fetch."""
        from unittest.mock import Mock, patch

        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()

        with patch("smartmemory.plugins.enrichers.link_expansion.httpx") as mock_httpx:
            mock_response = Mock()
            mock_response.text = "<html><head><title>Test</title></head></html>"
            mock_response.url = "https://example.com"
            mock_response.headers = {"content-type": "text/html"}
            mock_response.raise_for_status = Mock()
            mock_httpx.get.return_value = mock_response

            result = enricher._fetch_url("https://example.com")

        assert result["status"] == "success"
        assert result["html"] == "<html><head><title>Test</title></head></html>"
        assert result["final_url"] == "https://example.com"

    def test_fetch_url_timeout(self):
        """Test URL fetch timeout handling."""
        from unittest.mock import patch

        import httpx

        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()

        with patch("smartmemory.plugins.enrichers.link_expansion.httpx") as mock_httpx:
            mock_httpx.get.side_effect = httpx.TimeoutException("Timeout")

            result = enricher._fetch_url("https://example.com")

        assert result["status"] == "failed"
        assert "Timeout" in result["error"]
        assert result["error_type"] == "TimeoutException"

    def test_fetch_url_http_error(self):
        """Test URL fetch HTTP error handling."""
        from unittest.mock import Mock, patch

        import httpx

        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()

        with patch("smartmemory.plugins.enrichers.link_expansion.httpx") as mock_httpx:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found", request=Mock(), response=Mock()
            )
            mock_httpx.get.return_value = mock_response

            result = enricher._fetch_url("https://example.com/missing")

        assert result["status"] == "failed"
        assert "404" in result["error"]

    def test_fetch_url_truncates_large_content(self):
        """Test that large content is truncated."""
        from unittest.mock import Mock, patch

        from smartmemory.plugins.enrichers.link_expansion import (
            LinkExpansionEnricher,
            LinkExpansionEnricherConfig,
        )

        config = LinkExpansionEnricherConfig(max_content_length=100)
        enricher = LinkExpansionEnricher(config=config)

        with patch("smartmemory.plugins.enrichers.link_expansion.httpx") as mock_httpx:
            mock_response = Mock()
            mock_response.text = "x" * 1000  # Large content
            mock_response.url = "https://example.com"
            mock_response.headers = {"content-type": "text/html"}
            mock_response.raise_for_status = Mock()
            mock_httpx.get.return_value = mock_response

            result = enricher._fetch_url("https://example.com")

        assert len(result["html"]) == 100

    def test_fetch_url_follows_redirects(self):
        """Test that redirects are followed and final URL captured."""
        from unittest.mock import Mock, patch

        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()

        with patch("smartmemory.plugins.enrichers.link_expansion.httpx") as mock_httpx:
            mock_response = Mock()
            mock_response.text = "<html></html>"
            mock_response.url = "https://www.example.com/final"  # Redirected URL
            mock_response.headers = {"content-type": "text/html"}
            mock_response.raise_for_status = Mock()
            mock_httpx.get.return_value = mock_response

            result = enricher._fetch_url("https://example.com")

        assert result["final_url"] == "https://www.example.com/final"


class TestMetadataExtraction:
    """Tests for heuristic metadata extraction from HTML."""

    def test_extract_title_from_og_tag(self):
        """Test extracting title from Open Graph tag."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        html = '''
        <html>
        <head>
            <meta property="og:title" content="OG Title">
            <title>HTML Title</title>
        </head>
        </html>
        '''

        metadata = enricher._extract_metadata(html, "https://example.com")

        assert metadata["title"] == "OG Title"

    def test_extract_title_fallback_to_html_title(self):
        """Test fallback to HTML title when no OG tag."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        html = "<html><head><title>HTML Title</title></head></html>"

        metadata = enricher._extract_metadata(html, "https://example.com")

        assert metadata["title"] == "HTML Title"

    def test_extract_description_from_og_tag(self):
        """Test extracting description from Open Graph tag."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        html = '''
        <html>
        <head>
            <meta property="og:description" content="OG Description">
            <meta name="description" content="Meta Description">
        </head>
        </html>
        '''

        metadata = enricher._extract_metadata(html, "https://example.com")

        assert metadata["description"] == "OG Description"

    def test_extract_domain(self):
        """Test extracting domain from URL."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        html = "<html><head><title>Test</title></head></html>"

        metadata = enricher._extract_metadata(html, "https://www.example.com/page")

        assert metadata["domain"] == "www.example.com"

    def test_extract_og_image(self):
        """Test extracting Open Graph image."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        html = '''
        <html>
        <head>
            <meta property="og:image" content="https://example.com/image.jpg">
        </head>
        </html>
        '''

        metadata = enricher._extract_metadata(html, "https://example.com")

        assert metadata["og_image"] == "https://example.com/image.jpg"

    def test_extract_author(self):
        """Test extracting author from meta tag."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        html = '''
        <html>
        <head>
            <meta name="author" content="John Doe">
        </head>
        </html>
        '''

        metadata = enricher._extract_metadata(html, "https://example.com")

        assert metadata["author"] == "John Doe"

    def test_extract_canonical_url(self):
        """Test extracting canonical URL."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        html = '''
        <html>
        <head>
            <link rel="canonical" href="https://example.com/canonical">
        </head>
        </html>
        '''

        metadata = enricher._extract_metadata(html, "https://example.com/page?ref=123")

        assert metadata["canonical_url"] == "https://example.com/canonical"

    def test_extract_metadata_truncates_long_values(self):
        """Test that long metadata values are truncated."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        long_title = "T" * 1000
        html = f"<html><head><title>{long_title}</title></head></html>"

        metadata = enricher._extract_metadata(html, "https://example.com")

        assert len(metadata["title"]) == 500  # Truncated to 500


class TestEntityExtraction:
    """Tests for heuristic entity extraction from HTML."""

    def test_extract_author_as_person_entity(self):
        """Test extracting author as PERSON entity."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        html = '<html><head><meta name="author" content="John Doe"></head></html>'
        metadata = {"author": "John Doe"}

        entities = enricher._extract_entities_heuristic(html, metadata)

        assert len(entities) >= 1
        author_entity = next((e for e in entities if e["name"] == "John Doe"), None)
        assert author_entity is not None
        assert author_entity["type"] == "PERSON"
        assert author_entity["source"] == "meta"

    def test_extract_entities_from_jsonld_person(self):
        """Test extracting entities from JSON-LD Person."""
        import json

        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        jsonld = {
            "@type": "Person",
            "name": "Jane Smith",
        }
        html = f'''
        <html>
        <head>
            <script type="application/ld+json">{json.dumps(jsonld)}</script>
        </head>
        </html>
        '''

        entities = enricher._extract_entities_heuristic(html, {})

        assert len(entities) >= 1
        person = next((e for e in entities if e["name"] == "Jane Smith"), None)
        assert person is not None
        assert person["type"] == "PERSON"
        assert person["source"] == "jsonld"

    def test_extract_entities_from_jsonld_organization(self):
        """Test extracting entities from JSON-LD Organization."""
        import json

        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        jsonld = {
            "@type": "Organization",
            "name": "Acme Corp",
        }
        html = f'''
        <html>
        <head>
            <script type="application/ld+json">{json.dumps(jsonld)}</script>
        </head>
        </html>
        '''

        entities = enricher._extract_entities_heuristic(html, {})

        org = next((e for e in entities if e["name"] == "Acme Corp"), None)
        assert org is not None
        assert org["type"] == "ORG"
        assert org["source"] == "jsonld"

    def test_extract_entities_from_jsonld_article(self):
        """Test extracting entities from JSON-LD Article with author."""
        import json

        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        jsonld = {
            "@type": "Article",
            "author": {"@type": "Person", "name": "Bob Writer"},
        }
        html = f'''
        <html>
        <head>
            <script type="application/ld+json">{json.dumps(jsonld)}</script>
        </head>
        </html>
        '''

        entities = enricher._extract_entities_heuristic(html, {})

        author = next((e for e in entities if e["name"] == "Bob Writer"), None)
        assert author is not None
        assert author["type"] == "PERSON"

    def test_extract_entities_deduplicates(self):
        """Test that duplicate entities are removed."""
        import json

        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        jsonld = {"@type": "Person", "name": "John Doe"}
        html = f'''
        <html>
        <head>
            <meta name="author" content="John Doe">
            <script type="application/ld+json">{json.dumps(jsonld)}</script>
        </head>
        </html>
        '''
        metadata = {"author": "John Doe"}

        entities = enricher._extract_entities_heuristic(html, metadata)

        john_entities = [e for e in entities if e["name"] == "John Doe"]
        assert len(john_entities) == 1  # Deduplicated

    def test_extract_entities_empty_html(self):
        """Test entity extraction from HTML with no structured data."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        html = "<html><body>Just text</body></html>"

        entities = enricher._extract_entities_heuristic(html, {})

        assert entities == []


class TestEnrichMethod:
    """Tests for the main enrich() method."""

    def test_enrich_with_no_urls(self):
        """Test enriching content with no URLs."""
        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        result = enricher.enrich("No links here.")

        assert result["web_resources"] == []
        assert result["provenance_candidates"] == []
        assert result["tags"] == []

    def test_enrich_with_single_url(self):
        """Test enriching content with a single URL."""
        from unittest.mock import Mock, patch

        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()

        with patch("smartmemory.plugins.enrichers.link_expansion.httpx") as mock_httpx:
            mock_response = Mock()
            mock_response.text = '''
            <html>
            <head>
                <title>Test Page</title>
                <meta name="author" content="Test Author">
            </head>
            </html>
            '''
            mock_response.url = "https://example.com"
            mock_response.headers = {"content-type": "text/html"}
            mock_response.raise_for_status = Mock()
            mock_httpx.get.return_value = mock_response

            result = enricher.enrich("Check out https://example.com for info.")

        assert len(result["web_resources"]) == 1
        resource = result["web_resources"][0]
        assert resource["url"] == "https://example.com"
        assert resource["status"] == "success"
        assert resource["metadata"]["title"] == "Test Page"
        assert len(result["provenance_candidates"]) == 1
        assert "Test Author" in result["tags"]

    def test_enrich_with_failed_fetch(self):
        """Test enriching when URL fetch fails."""
        from unittest.mock import patch

        import httpx

        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()

        with patch("smartmemory.plugins.enrichers.link_expansion.httpx") as mock_httpx:
            mock_httpx.get.side_effect = httpx.TimeoutException("Timeout")

            result = enricher.enrich("Check https://example.com")

        assert len(result["web_resources"]) == 1
        resource = result["web_resources"][0]
        assert resource["status"] == "failed"
        assert "error" in resource
        # Failed resources still get node_id for retry capability
        assert "node_id" in resource

    def test_enrich_creates_correct_node_id(self):
        """Test that node IDs are created correctly."""
        import hashlib
        from unittest.mock import Mock, patch

        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()

        with patch("smartmemory.plugins.enrichers.link_expansion.httpx") as mock_httpx:
            mock_response = Mock()
            mock_response.text = "<html><title>Test</title></html>"
            mock_response.url = "https://example.com"
            mock_response.headers = {}
            mock_response.raise_for_status = Mock()
            mock_httpx.get.return_value = mock_response

            result = enricher.enrich("Visit https://example.com")

        resource = result["web_resources"][0]
        expected_hash = hashlib.md5("https://example.com".encode()).hexdigest()[:12]
        assert resource["node_id"] == f"webresource:{expected_hash}"

    def test_enrich_extracts_entities(self):
        """Test that entities are extracted and returned as tags."""
        import json
        from unittest.mock import Mock, patch

        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()
        jsonld = {"@type": "Person", "name": "John Doe"}

        with patch("smartmemory.plugins.enrichers.link_expansion.httpx") as mock_httpx:
            mock_response = Mock()
            mock_response.text = f'''
            <html>
            <head>
                <script type="application/ld+json">{json.dumps(jsonld)}</script>
            </head>
            </html>
            '''
            mock_response.url = "https://example.com"
            mock_response.headers = {}
            mock_response.raise_for_status = Mock()
            mock_httpx.get.return_value = mock_response

            result = enricher.enrich("See https://example.com")

        assert "John Doe" in result["tags"]
        resource = result["web_resources"][0]
        assert len(resource["extracted_entities"]) >= 1


class TestLLMAnalysis:
    """Tests for optional LLM analysis."""

    def test_enrich_with_llm_enabled(self):
        """Test enriching with LLM analysis enabled."""
        import json
        from unittest.mock import Mock, patch

        from smartmemory.plugins.enrichers.link_expansion import (
            LinkExpansionEnricher,
            LinkExpansionEnricherConfig,
        )

        config = LinkExpansionEnricherConfig(enable_llm=True)
        enricher = LinkExpansionEnricher(config=config)

        with patch("smartmemory.plugins.enrichers.link_expansion.httpx") as mock_httpx:
            mock_response = Mock()
            mock_response.text = "<html><body>Long article content here...</body></html>"
            mock_response.url = "https://example.com"
            mock_response.headers = {}
            mock_response.raise_for_status = Mock()
            mock_httpx.get.return_value = mock_response

            with patch("smartmemory.plugins.enrichers.link_expansion.openai") as mock_openai:
                mock_completion = Mock()
                mock_completion.choices = [
                    Mock(
                        message=Mock(
                            content=json.dumps({
                                "summary": "This is a summary.",
                                "entities": [{"name": "LLM Entity", "type": "TOPIC"}],
                                "topics": ["AI", "Technology"],
                            })
                        )
                    )
                ]
                mock_openai.chat.completions.create.return_value = mock_completion

                result = enricher.enrich("Check https://example.com")

        resource = result["web_resources"][0]
        assert resource["summary"] == "This is a summary."
        assert "LLM Entity" in result["tags"]

    def test_enrich_llm_failure_falls_back(self):
        """Test that LLM failure doesn't break enrichment."""
        from unittest.mock import Mock, patch

        from smartmemory.plugins.enrichers.link_expansion import (
            LinkExpansionEnricher,
            LinkExpansionEnricherConfig,
        )

        config = LinkExpansionEnricherConfig(enable_llm=True)
        enricher = LinkExpansionEnricher(config=config)

        with patch("smartmemory.plugins.enrichers.link_expansion.httpx") as mock_httpx:
            mock_response = Mock()
            mock_response.text = "<html><title>Test</title></html>"
            mock_response.url = "https://example.com"
            mock_response.headers = {}
            mock_response.raise_for_status = Mock()
            mock_httpx.get.return_value = mock_response

            with patch("smartmemory.plugins.enrichers.link_expansion.openai") as mock_openai:
                mock_openai.chat.completions.create.side_effect = Exception("API Error")

                result = enricher.enrich("Check https://example.com")

        # Should still succeed with heuristic results
        assert len(result["web_resources"]) == 1
        assert result["web_resources"][0]["status"] == "success"

    def test_llm_disabled_by_default(self):
        """Test that LLM is not called when disabled."""
        from unittest.mock import Mock, patch

        from smartmemory.plugins.enrichers.link_expansion import LinkExpansionEnricher

        enricher = LinkExpansionEnricher()

        with patch("smartmemory.plugins.enrichers.link_expansion.httpx") as mock_httpx:
            mock_response = Mock()
            mock_response.text = "<html><title>Test</title></html>"
            mock_response.url = "https://example.com"
            mock_response.headers = {}
            mock_response.raise_for_status = Mock()
            mock_httpx.get.return_value = mock_response

            with patch("smartmemory.plugins.enrichers.link_expansion.openai") as mock_openai:
                result = enricher.enrich("Check https://example.com")

                # OpenAI should not be called
                mock_openai.chat.completions.create.assert_not_called()


class TestPluginRegistration:
    """Tests for plugin registration."""

    def test_enricher_exported_from_module(self):
        """Test that LinkExpansionEnricher is exported from enrichers module."""
        from smartmemory.plugins.enrichers import LinkExpansionEnricher

        assert LinkExpansionEnricher is not None

    def test_enricher_in_all(self):
        """Test that LinkExpansionEnricher is in __all__."""
        from smartmemory.plugins import enrichers

        assert "LinkExpansionEnricher" in enrichers.__all__
