#!/usr/bin/env python3
"""Test Wikipedia grounding to debug what's happening."""

import sys
sys.path.insert(0, '/Users/ruze/reg/my/SmartMemory/smart-memory')

from smartmemory.plugins.enrichers import WikipediaEnricher
from smartmemory.models.memory_item import MemoryItem

# Test the enricher directly
print("=" * 60)
print("Testing WikipediaEnricher")
print("=" * 60)

enricher = WikipediaEnricher()
item = MemoryItem(content="The rain in Spain falls in the plain")
entities = ['rain', 'Spain', 'plain']

print(f"\nTesting with entities: {entities}")
result = enricher.enrich(item, {'semantic_entities': entities})

print(f"\nResult keys: {result.keys()}")
print(f"\nWikipedia data entries: {len(result.get('wikipedia_data', {}))}")

for entity, data in result.get('wikipedia_data', {}).items():
    print(f"\n--- Entity: {entity} ---")
    print(f"  exists: {data.get('exists')}")
    print(f"  url: {data.get('url')}")
    print(f"  summary: {data.get('summary', '')[:100]}...")
    print(f"  categories: {len(data.get('categories', []))} categories")

print(f"\nProvenance candidates: {result.get('provenance_candidates', [])}")
