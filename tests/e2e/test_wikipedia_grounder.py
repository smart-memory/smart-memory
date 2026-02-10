"""E2E: WikipediaEnricher direct invocation.

Exercises: plugins/enrichers/WikipediaEnricher with real Wikipedia API.
"""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.golden]


def test_wikipedia_enricher_direct():
    """WikipediaEnricher fetches data for known entities."""
    from smartmemory.plugins.enrichers import WikipediaEnricher
    from smartmemory.models.memory_item import MemoryItem

    enricher = WikipediaEnricher()
    item = MemoryItem(content="The rain in Spain falls in the plain")
    entities = ["rain", "Spain", "plain"]

    result = enricher.enrich(item, {"semantic_entities": entities})

    assert isinstance(result, dict)
    wiki_data = result.get("wikipedia_data", {})
    assert len(wiki_data) > 0, "Expected at least one Wikipedia entry"

    # Spain should always resolve
    spain = wiki_data.get("Spain")
    assert spain is not None, "Spain should be found in Wikipedia"
    assert spain.get("exists") is True
    assert spain.get("url")
    assert spain.get("summary")

    provenance = result.get("provenance_candidates", [])
    assert isinstance(provenance, list)
