"""
SmartMemory Clustering Module.

Provides entity and relation clustering capabilities:
- GlobalClustering: Pipeline stage for graph-wide deduplication
- LLMClustering: LLM-based semantic clustering
- EmbeddingClusterer: KMeans clustering on embeddings
- HybridDeduplicator: Combined SemHash + KMeans + LLM deduplication
- GraphAggregator: Merge multiple extraction results

Usage:
    from smartmemory.clustering import (
        GlobalClustering,
        LLMClustering,
        EmbeddingClusterer,
        HybridDeduplicator,
        GraphAggregator,
        cluster_extraction_result,
        aggregate_graphs,
        deduplicate_extraction,
    )
"""

from smartmemory.clustering.global_cluster import GlobalClustering
from smartmemory.clustering.llm import LLMClustering
from smartmemory.clustering.graph_aggregator import GraphAggregator
from smartmemory.clustering.embedding import EmbeddingClusterer
from smartmemory.clustering.deduplicator import HybridDeduplicator

__all__ = [
    # Classes
    "GlobalClustering",
    "LLMClustering",
    "EmbeddingClusterer",
    "HybridDeduplicator",
    "GraphAggregator",
]
