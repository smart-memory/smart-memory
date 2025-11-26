from typing import List, Dict, Any, Optional, Tuple

from smartmemory.memory.pipeline.stages.clustering import cluster_extraction_result


class GraphAggregator:
    """
    Aggregates multiple extraction results into a unified graph.

    aggregate() method, this combines entities and relations
    from multiple sources while handling duplicates and conflicts.

    Usage:
        aggregator = GraphAggregator()
        combined = aggregator.aggregate([graph1, graph2, graph3])
        # Optionally cluster the combined result
        clustered = cluster_extraction_result(combined, context="...")
    """

    def __init__(self, merge_strategy: str = "union"):
        """
        Initialize aggregator.

        Args:
            merge_strategy: How to handle conflicts
                - "union": Keep all unique entities/relations
                - "latest": Prefer later graphs for conflicts
                - "highest_confidence": Prefer higher confidence entries
        """
        self.merge_strategy = merge_strategy

    def aggregate(
        self,
        graphs: List[Dict[str, Any]],
        cluster: bool = False,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Aggregate multiple extraction results into one.

        Args:
            graphs: List of extraction results, each with 'entities' and 'relations'
            cluster: Whether to apply LLM clustering after aggregation
            context: Optional context for clustering

        Returns:
            Combined extraction result with:
            - entities: Merged entity list
            - relations: Merged relation list
            - source_count: Number of source graphs
            - entity_clusters: (if cluster=True) Entity cluster mapping
            - edge_clusters: (if cluster=True) Relation cluster mapping
        """
        if not graphs:
            return {"entities": [], "relations": [], "source_count": 0}

        if len(graphs) == 1:
            result = dict(graphs[0])
            result["source_count"] = 1
            if cluster:
                return cluster_extraction_result(result, context)
            return result

        # Merge entities
        merged_entities = self._merge_entities(graphs)

        # Merge relations
        merged_relations = self._merge_relations(graphs, merged_entities)

        result = {
            "entities": merged_entities,
            "relations": merged_relations,
            "source_count": len(graphs)
        }

        # Optionally cluster
        if cluster:
            result = cluster_extraction_result(result, context)

        return result

    def _merge_entities(self, graphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge entities from multiple graphs.

        Uses canonical key matching for deduplication.
        """
        from smartmemory.utils.deduplication import get_canonical_key

        # Track entities by canonical key
        entity_map: Dict[str, Dict[str, Any]] = {}

        for graph_idx, graph in enumerate(graphs):
            entities = graph.get('entities', [])

            for entity in entities:
                # Handle both dict and MemoryItem
                if hasattr(entity, 'metadata'):
                    name = entity.metadata.get('name') or entity.content
                    entity_type = entity.metadata.get('entity_type', 'concept')
                    entity_dict = {
                        'name': name,
                        'entity_type': entity_type,
                        'confidence': entity.metadata.get('confidence', 0.8),
                        'item_id': entity.item_id,
                        **{k: v for k, v in entity.metadata.items()
                           if k not in ['name', 'entity_type', 'confidence']}
                    }
                else:
                    name = entity.get('name') or entity.get('content', '')
                    entity_type = entity.get('entity_type', 'concept')
                    entity_dict = dict(entity)

                if not name:
                    continue

                # Generate canonical key
                key = get_canonical_key(name, entity_type)

                if key not in entity_map:
                    entity_dict['_source_graphs'] = [graph_idx]
                    entity_map[key] = entity_dict
                else:
                    # Handle merge based on strategy
                    existing = entity_map[key]
                    existing['_source_graphs'] = existing.get('_source_graphs', []) + [graph_idx]

                    if self.merge_strategy == "latest":
                        # Update with newer values
                        for k, v in entity_dict.items():
                            if v and k != '_source_graphs':
                                existing[k] = v
                    elif self.merge_strategy == "highest_confidence":
                        # Keep higher confidence
                        if entity_dict.get('confidence', 0) > existing.get('confidence', 0):
                            for k, v in entity_dict.items():
                                if v and k != '_source_graphs':
                                    existing[k] = v
                    # "union" strategy: keep first occurrence, just track sources

        return list(entity_map.values())

    def _merge_relations(
        self,
        graphs: List[Dict[str, Any]],
        merged_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge relations from multiple graphs.

        Deduplicates based on (subject, predicate, object) tuple.
        """
        # Build entity name lookup for normalization
        entity_names = {e.get('name', '').lower() for e in merged_entities}

        # Track relations by (subject, predicate, object) key
        relation_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for graph_idx, graph in enumerate(graphs):
            relations = graph.get('relations', [])

            for rel in relations:
                # Extract subject, predicate, object
                subject = (rel.get('subject') or rel.get('source') or
                          rel.get('source_name', '')).strip()
                predicate = (rel.get('predicate') or rel.get('relation_type', '')).strip()
                obj = (rel.get('object') or rel.get('target') or
                      rel.get('target_name', '')).strip()

                if not (subject and predicate and obj):
                    continue

                # Create normalized key
                key = (subject.lower(), predicate.lower(), obj.lower())

                if key not in relation_map:
                    rel_dict = dict(rel)
                    rel_dict['_source_graphs'] = [graph_idx]
                    relation_map[key] = rel_dict
                else:
                    # Track source
                    relation_map[key]['_source_graphs'] = (
                        relation_map[key].get('_source_graphs', []) + [graph_idx]
                    )

        return list(relation_map.values())
