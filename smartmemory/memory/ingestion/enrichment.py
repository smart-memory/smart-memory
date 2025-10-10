"""
Enrichment pipeline module for ingestion flow.

This module handles all enrichment operations including:
- Running enrichment plugins
- Processing enrichment results
- Creating derivative items
- Wikipedia grounding operations
"""
from datetime import datetime
from typing import Dict, Any

from smartmemory.memory.ingestion import utils as ingestion_utils
from smartmemory.models.memory_item import MemoryItem


class EnrichmentPipeline:
    """
    Handles all enrichment operations for the ingestion pipeline.
    
    Manages enrichment plugin execution, result processing, and derivative item creation.
    """

    def __init__(self, memory, enrichment, observer):
        """
        Initialize enrichment pipeline.
        
        Args:
            memory: SmartMemory instance for storage operations
            enrichment: Enrichment module instance
            observer: IngestionObserver instance for event emission
        """
        self.memory = memory
        self.enrichment = enrichment
        self.observer = observer

    def run_enrichment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run enrichment plugins via the enrichment module.
        By default, runs all enrichers if context['enricher_names'] is not set.
        """
        if context.get('enricher_names'):
            enrichment_result = {}
            for enricher_name in context['enricher_names']:
                enrichment_result.update(self.enrichment.enrich(context, enricher_names=[enricher_name]))
        else:
            # Run all enrichers by default
            enrichment_result = self.enrichment.enrich(context)

        # Add new items from enrichment (derivatives) to the memory graph and link them
        if enrichment_result and enrichment_result.get('new_items'):
            original_id = context['item'].item_id if hasattr(context['item'], 'item_id') else None

            for new_item in enrichment_result['new_items']:
                if not hasattr(new_item, 'item_id'):
                    if isinstance(new_item, dict):
                        mem_item = MemoryItem(**new_item)
                    else:
                        mem_item = MemoryItem(content=str(new_item))
                else:
                    mem_item = new_item

                derived_id = self.memory.add(mem_item)
                edge_props = {'provenance': 'enrichment'}
                self.memory._graph.add_edge(
                    source_id=derived_id,
                    target_id=original_id,
                    edge_type=ingestion_utils.sanitize_relation_type('CANONICAL'),
                    properties=edge_props,
                )

                # If the new item has relations/triples, add them to the graph
                if hasattr(mem_item, 'relations') and mem_item.relations:
                    new_context = {'item': mem_item, 'semantic_relations': mem_item.relations}
                    if 'enrichment_result' in context:
                        new_context['enrichment_result'] = enrichment_result
                    # Note: This would need to be handled by the storage pipeline
                    # self._add_triples(new_context)

        return enrichment_result

    def ground_with_wikipedia(self, context: Dict[str, Any]):
        """Ground entities with Wikipedia data if present in enrichment."""
        enrichment_result = context.get('enrichment_result')
        wiki_data = enrichment_result.get('wikipedia_data') or {} if enrichment_result else {}

        def infer_times(entity=None, enrichment_result=None):
            valid_start = None
            valid_end = None
            tx_time = None
            if enrichment_result:
                if entity and enrichment_result.get('temporal') or {}.get(entity):
                    temporal = enrichment_result['temporal'][entity]
                    valid_start = temporal.get('valid_start')
                    valid_end = temporal.get('valid_end')
                    tx_time = temporal.get('transaction_time')
            valid_start = valid_start or datetime.now()
            tx_time = tx_time or datetime.now()
            return valid_start, valid_end, tx_time

        for entity, wiki in wiki_data.items():
            url = wiki.get('url')
            if url:
                wiki_item = MemoryItem(
                    content=url,
                    metadata={
                        'type': 'wikipedia',
                        'title': wiki.get('title', entity),
                        'summary': wiki.get('summary', ''),
                        'categories': wiki.get('categories', [])
                    }
                )
                valid_start, valid_end, tx_time = infer_times(entity, enrichment_result)
                # Additional grounding logic would go here
