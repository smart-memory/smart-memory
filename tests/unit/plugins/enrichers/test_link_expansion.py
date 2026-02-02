"""Unit tests for LinkExpansionEnricher plugin."""

import pytest
from dataclasses import FrozenInstanceError


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
