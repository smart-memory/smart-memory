"""
Global Clustering Stage.

This stage performs global deduplication by clustering entities based on their vector embeddings
and optionally using LLM for semantic clustering.

Features:
- Embedding-based clustering for fast similarity detection
- LLM-based clustering for semantic equivalence (Joe ↔ Joseph, ML ↔ machine learning)
- Edge/relation clustering for predicate normalization
- Context-aware clustering with domain hints
"""

import logging
from typing import List, Dict, Any, Optional


logger = logging.getLogger(__name__)


def cluster_extraction_result(
    extraction_result: Dict[str, Any],
    context: Optional[str] = None,
    model_name: str = "gpt-5-mini"
) -> Dict[str, Any]:
    """
    Apply LLM clustering to an extraction result.
    
    This is a convenience function that clusters both entities and relations
    in a single extraction result.
    
    Args:
        extraction_result: Dict with 'entities' and 'relations' keys
        context: Optional domain context for clustering
        model_name: LLM model to use
        
    Returns:
        Extraction result with clustered entities/relations and cluster mappings
    """
    from smartmemory.clustering.llm import LLMClustering
    clusterer = LLMClustering(model_name=model_name, context=context)
    
    entities = extraction_result.get('entities', [])
    relations = extraction_result.get('relations', [])
    
    result = dict(extraction_result)
    
    # Cluster entities
    if entities and len(entities) >= 2:
        entity_clusters = clusterer.cluster_entities(entities, context)
        if entity_clusters:
            deduplicated, name_mapping = clusterer.apply_entity_clusters(entities, entity_clusters)
            result['entities'] = deduplicated
            result['entity_clusters'] = {k: list(v) for k, v in entity_clusters.items()}
            
            # Update relation references to use canonical names
            for rel in relations:
                for key in ['subject', 'source', 'source_name']:
                    if key in rel:
                        rel[key] = name_mapping.get(rel[key].lower(), rel[key])
                for key in ['object', 'target', 'target_name']:
                    if key in rel:
                        rel[key] = name_mapping.get(rel[key].lower(), rel[key])
    
    # Cluster relations
    if relations and len(relations) >= 2:
        predicates = []
        for rel in relations:
            pred = rel.get('predicate') or rel.get('relation_type', '')
            if pred:
                predicates.append(pred)
        
        if len(set(predicates)) >= 2:
            relation_clusters = clusterer.cluster_relations(predicates, context)
            if relation_clusters:
                result['relations'] = clusterer.apply_relation_clusters(relations, relation_clusters)
                result['edge_clusters'] = {k: list(v) for k, v in relation_clusters.items()}
    
    return result


def aggregate_graphs(
    graphs: List[Dict[str, Any]],
    cluster: bool = False,
    context: Optional[str] = None,
    merge_strategy: str = "union"
) -> Dict[str, Any]:
    """
    Convenience function to aggregate multiple extraction results.
    
    Args:
        graphs: List of extraction results to combine
        cluster: Whether to apply LLM clustering after aggregation
        context: Optional domain context for clustering
        merge_strategy: How to handle conflicts ("union", "latest", "highest_confidence")
        
    Returns:
        Combined extraction result
        
    Example:
        graph1 = extractor.extract("Linda is Joe's mother.")
        graph2 = extractor.extract("Joseph is Andrew's son.")  # Joseph = Joe
        combined = aggregate_graphs([graph1, graph2], cluster=True, context="Family")
        # combined['entity_clusters'] = {'Joseph': ['Joe', 'Joseph']}
    """
    from smartmemory.clustering import GraphAggregator
    aggregator = GraphAggregator(merge_strategy=merge_strategy)
    return aggregator.aggregate(graphs, cluster=cluster, context=context)


# ============================================================================
# KMeans Pre-clustering for Efficient LLM Deduplication
# ============================================================================


def deduplicate_extraction(
    extraction_result: Dict[str, Any],
    method: str = "full",
    semhash_threshold: float = 0.95,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for hybrid deduplication.
    
    Args:
        extraction_result: Dict with 'entities' and 'relations'
        method: "semhash", "kmeans_llm", or "full"
        semhash_threshold: SemHash similarity threshold
        context: Optional domain context
        
    Returns:
        Deduplicated extraction result
    """
    from smartmemory.clustering.deduplicator import HybridDeduplicator
    deduplicator = HybridDeduplicator(
        semhash_threshold=semhash_threshold,
        context=context
    )
    return deduplicator.deduplicate(extraction_result, method=method)
