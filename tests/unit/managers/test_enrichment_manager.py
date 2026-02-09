"""Unit tests for EnrichmentManager."""

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.managers.enrichment import EnrichmentManager
from smartmemory.models.memory_item import MemoryItem


@pytest.fixture
def mock_enrichment():
    e = MagicMock()
    e.enrich.return_value = None
    return e


@pytest.fixture
def mock_grounding():
    g = MagicMock()
    g.ground.return_value = {"grounded": True}
    return g


@pytest.fixture
def mock_external_resolver():
    r = MagicMock()
    r.resolve_external.return_value = [{"source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Test"}]
    return r


@pytest.fixture
def manager(mock_enrichment, mock_grounding, mock_external_resolver):
    return EnrichmentManager(mock_enrichment, mock_grounding, mock_external_resolver)


class TestEnrich:
    def test_enrich_delegates(self, manager, mock_enrichment):
        manager.enrich("item_123")
        mock_enrichment.enrich.assert_called_once_with("item_123", None)

    def test_enrich_with_routines(self, manager, mock_enrichment):
        manager.enrich("item_123", routines=["sentiment", "temporal"])
        mock_enrichment.enrich.assert_called_once_with("item_123", ["sentiment", "temporal"])

    def test_enrich_propagates_error(self, manager, mock_enrichment):
        mock_enrichment.enrich.side_effect = RuntimeError("Enrichment failed")
        with pytest.raises(RuntimeError, match="Enrichment failed"):
            manager.enrich("item_123")


class TestGround:
    def test_ground_builds_context_and_delegates(self, manager, mock_grounding):
        manager.ground("item_123", "https://example.com/source")
        call_args = mock_grounding.ground.call_args[0][0]
        assert call_args["item_id"] == "item_123"
        assert call_args["source_url"] == "https://example.com/source"
        assert call_args["validation"] is None

    def test_ground_with_validation(self, manager, mock_grounding):
        validation = {"checksum": "abc123"}
        manager.ground("item_123", "https://example.com", validation=validation)
        call_args = mock_grounding.ground.call_args[0][0]
        assert call_args["validation"] == {"checksum": "abc123"}

    def test_ground_context_passes_dict_directly(self, manager, mock_grounding):
        ctx = {"item_id": "item_456", "source_url": "https://example.com", "extra": "data"}
        manager.ground_context(ctx)
        mock_grounding.ground.assert_called_once_with(ctx)


class TestResolveExternal:
    def test_resolve_external_delegates(self, manager, mock_external_resolver):
        item = MemoryItem(content="Albert Einstein was a physicist", item_id="item_789")
        result = manager.resolve_external(item)
        mock_external_resolver.resolve_external.assert_called_once_with(item)
        assert len(result) == 1
        assert result[0]["source"] == "wikipedia"

    def test_resolve_external_returns_none_when_no_match(self, manager, mock_external_resolver):
        mock_external_resolver.resolve_external.return_value = None
        item = MemoryItem(content="random gibberish", item_id="item_000")
        result = manager.resolve_external(item)
        assert result is None
