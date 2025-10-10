"""
Enhanced Similarity Framework - Main orchestrator for all similarity metrics.

Provides unified interface for calculating comprehensive similarity between memory items
using multiple metrics (semantic, graph-based, temporal, content, metadata).
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple

from .enhanced_metrics import (
    SimilarityResult,
    SimilarityMetric,
    ContentSimilarityMetric,
    SemanticSimilarityMetric,
    TemporalSimilarityMetric,
    GraphSimilarityMetric,
    MetadataSimilarityMetric,
    AgentWorkflowSimilarityMetric
)
from ..models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


@dataclass
class SimilarityConfig:
    """Configuration for similarity calculations."""
    # Metric weights (should sum to 1.0)
    content_weight: float = 0.20
    semantic_weight: float = 0.25
    temporal_weight: float = 0.15
    graph_weight: float = 0.15
    metadata_weight: float = 0.10
    agent_workflow_weight: float = 0.15

    # Individual metric configurations
    use_fuzzy_matching: bool = True
    fuzzy_threshold: float = 0.8
    time_decay_factor: float = 0.1
    embedding_model: Optional[str] = None

    # Thresholds
    similarity_threshold: float = 0.5
    high_similarity_threshold: float = 0.8

    # Performance settings
    enable_caching: bool = True
    max_cache_size: int = 1000

    def __post_init__(self):
        """Validate configuration."""
        total_weight = (
                self.content_weight + self.semantic_weight + self.temporal_weight +
                self.graph_weight + self.metadata_weight + self.agent_workflow_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Similarity weights sum to {total_weight}, not 1.0. Normalizing...")
            # Normalize weights
            self.content_weight /= total_weight
            self.semantic_weight /= total_weight
            self.temporal_weight /= total_weight
            self.graph_weight /= total_weight
            self.metadata_weight /= total_weight
            self.agent_workflow_weight /= total_weight


class EnhancedSimilarityFramework:
    """
    Main framework for enhanced similarity calculations.
    
    Orchestrates multiple similarity metrics to provide comprehensive
    similarity scoring between memory items.
    """

    def __init__(self, config: Optional[SimilarityConfig] = None, graph_store=None):
        self.config = config or SimilarityConfig()
        self.graph_store = graph_store

        # Initialize metrics
        self.metrics: Dict[str, SimilarityMetric] = {}
        self._initialize_metrics()

        # Caching
        self._similarity_cache = {} if self.config.enable_caching else None

        logger.info("Enhanced Similarity Framework initialized")

    def _initialize_metrics(self):
        """Initialize all similarity metrics."""
        self.metrics['content'] = ContentSimilarityMetric(
            use_fuzzy_matching=self.config.use_fuzzy_matching,
            fuzzy_threshold=self.config.fuzzy_threshold
        )

        self.metrics['semantic'] = SemanticSimilarityMetric(
            embedding_model=self.config.embedding_model
        )

        self.metrics['temporal'] = TemporalSimilarityMetric(
            time_decay_factor=self.config.time_decay_factor
        )

        self.metrics['graph'] = GraphSimilarityMetric(
            graph_store=self.graph_store
        )

        self.metrics['metadata'] = MetadataSimilarityMetric()

        self.metrics['agent_workflow'] = AgentWorkflowSimilarityMetric()

    def calculate_similarity(
            self,
            item1: MemoryItem,
            item2: MemoryItem,
            metrics: Optional[List[str]] = None,
            return_detailed: bool = False
    ) -> Union[float, SimilarityResult]:
        """
        Calculate comprehensive similarity between two memory items.
        
        Args:
            item1: First memory item
            item2: Second memory item
            metrics: List of specific metrics to use (default: all)
            return_detailed: Whether to return detailed SimilarityResult
            
        Returns:
            Float similarity score or detailed SimilarityResult
        """
        if not item1 or not item2:
            return SimilarityResult(0.0) if return_detailed else 0.0

        # Check cache
        cache_key = self._get_cache_key(item1, item2, metrics)
        if self._similarity_cache and cache_key in self._similarity_cache:
            cached_result = self._similarity_cache[cache_key]
            return cached_result if return_detailed else cached_result.overall_score

        # Use all metrics if none specified
        if metrics is None:
            metrics = list(self.metrics.keys())

        # Calculate individual metric scores
        scores = {}
        explanations = []

        for metric_name in metrics:
            if metric_name not in self.metrics:
                logger.warning(f"Unknown metric: {metric_name}")
                continue

            try:
                score = self.metrics[metric_name].calculate(item1, item2)
                scores[metric_name] = score
                explanations.append(f"{metric_name}: {score:.3f}")

            except Exception as e:
                logger.error(f"Error calculating {metric_name} similarity: {e}")
                scores[metric_name] = 0.0

        # Calculate weighted overall score
        overall_score = self._calculate_weighted_score(scores)

        # Round scores to avoid floating-point precision issues in caching
        overall_score = round(overall_score, 6)

        # Create detailed result with rounded scores
        result = SimilarityResult(
            overall_score=overall_score,
            semantic_score=round(scores.get('semantic', 0.0), 6),
            graph_score=round(scores.get('graph', 0.0), 6),
            temporal_score=round(scores.get('temporal', 0.0), 6),
            metadata_score=round(scores.get('metadata', 0.0), 6),
            content_score=round(scores.get('content', 0.0), 6),
            agent_workflow_score=round(scores.get('agent_workflow', 0.0), 6),
            explanation="; ".join(explanations),
            confidence=round(self._calculate_confidence(scores), 6)
        )

        # Cache result
        if self._similarity_cache:
            self._manage_cache_size()
            self._similarity_cache[cache_key] = result

        return result if return_detailed else overall_score

    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted similarity score with adaptive weighting for strong patterns."""
        # Base weights from config
        weights = {
            'semantic': self.config.semantic_weight,
            'graph': self.config.graph_weight,
            'temporal': self.config.temporal_weight,
            'metadata': self.config.metadata_weight,
            'content': self.config.content_weight,
            'agent_workflow': self.config.agent_workflow_weight,
        }

        # Adaptive weighting: boost agent workflow when strong patterns detected
        agent_workflow_score = scores.get('agent_workflow', 0.0)
        if agent_workflow_score > 0.7:  # Strong agent workflow pattern detected
            # Increase agent workflow weight and reduce others proportionally
            boost_factor = 1.5
            weights['agent_workflow'] *= boost_factor

            # Reduce other weights to maintain balance
            other_reduction = 0.8
            for key in weights:
                if key != 'agent_workflow':
                    weights[key] *= other_reduction

        # Normalize weights to sum to 1.0
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {k: v / weight_sum for k, v in weights.items()}

        # Calculate weighted sum
        total_weight = 0.0
        weighted_sum = 0.0

        for metric_name, score in scores.items():
            if metric_name in weights:
                weight = weights[metric_name]
                weighted_sum += score * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate confidence in the similarity result."""
        if not scores:
            return 0.0

        # Confidence based on metric agreement
        score_values = list(scores.values())
        mean_score = sum(score_values) / len(score_values)

        # Calculate variance
        variance = sum((score - mean_score) ** 2 for score in score_values) / len(score_values)

        # Lower variance = higher confidence
        confidence = max(0.0, 1.0 - variance)

        return confidence

    def find_similar_items(
            self,
            target_item: MemoryItem,
            candidate_items: List[MemoryItem],
            threshold: Optional[float] = None,
            max_results: int = 10,
            metrics: Optional[List[str]] = None
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Find items similar to target item from candidates.
        
        Args:
            target_item: Item to find similarities for
            candidate_items: List of candidate items to compare
            threshold: Minimum similarity threshold (default: config threshold)
            max_results: Maximum number of results to return
            metrics: Specific metrics to use
            
        Returns:
            List of (item, similarity_score) tuples, sorted by similarity
        """
        if threshold is None:
            threshold = self.config.similarity_threshold

        similar_items = []

        for candidate in candidate_items:
            if candidate.item_id == target_item.item_id:
                continue  # Skip self-comparison

            similarity = self.calculate_similarity(target_item, candidate, metrics)

            if similarity >= threshold:
                similar_items.append((candidate, similarity))

        # Sort by similarity (descending) and limit results
        similar_items.sort(key=lambda x: x[1], reverse=True)
        return similar_items[:max_results]

    def cluster_items(
            self,
            items: List[MemoryItem],
            similarity_threshold: Optional[float] = None,
            metrics: Optional[List[str]] = None
    ) -> List[List[MemoryItem]]:
        """
        Cluster items based on similarity.
        
        Args:
            items: Items to cluster
            similarity_threshold: Threshold for clustering (default: high threshold)
            metrics: Specific metrics to use
            
        Returns:
            List of clusters (each cluster is a list of similar items)
        """
        if similarity_threshold is None:
            similarity_threshold = self.config.high_similarity_threshold

        clusters = []
        unassigned = items.copy()

        while unassigned:
            # Start new cluster with first unassigned item
            seed_item = unassigned.pop(0)
            cluster = [seed_item]

            # Find items similar to seed
            to_remove = []
            for i, candidate in enumerate(unassigned):
                similarity = self.calculate_similarity(seed_item, candidate, metrics)
                if similarity >= similarity_threshold:
                    cluster.append(candidate)
                    to_remove.append(i)

            # Remove assigned items from unassigned list
            for i in reversed(to_remove):
                unassigned.pop(i)

            clusters.append(cluster)

        return clusters

    def get_similarity_matrix(
            self,
            items: List[MemoryItem],
            metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate similarity matrix for a list of items.
        
        Args:
            items: Items to generate matrix for
            metrics: Specific metrics to use
            
        Returns:
            Dictionary mapping item_id -> item_id -> similarity_score
        """
        matrix = {}

        for i, item1 in enumerate(items):
            matrix[item1.item_id] = {}

            for j, item2 in enumerate(items):
                if i == j:
                    matrix[item1.item_id][item2.item_id] = 1.0
                elif item2.item_id in matrix and item1.item_id in matrix[item2.item_id]:
                    # Use symmetry to avoid duplicate calculations
                    matrix[item1.item_id][item2.item_id] = matrix[item2.item_id][item1.item_id]
                else:
                    similarity = self.calculate_similarity(item1, item2, metrics)
                    matrix[item1.item_id][item2.item_id] = similarity

        return matrix

    def _get_cache_key(
            self,
            item1: MemoryItem,
            item2: MemoryItem,
            metrics: Optional[List[str]]
    ) -> str:
        """Generate cache key for similarity calculation."""
        # Sort item IDs for consistent caching regardless of order
        id1, id2 = sorted([item1.item_id, item2.item_id])
        metrics_str = ",".join(sorted(metrics)) if metrics else "all"
        return f"{id1}:{id2}:{metrics_str}"

    def _manage_cache_size(self):
        """Manage cache size to prevent memory issues."""
        if not self._similarity_cache:
            return

        if len(self._similarity_cache) >= self.config.max_cache_size:
            # Remove oldest 20% of entries (simple FIFO)
            items_to_remove = len(self._similarity_cache) // 5
            keys_to_remove = list(self._similarity_cache.keys())[:items_to_remove]

            for key in keys_to_remove:
                del self._similarity_cache[key]

    def clear_cache(self):
        """Clear similarity cache."""
        if self._similarity_cache:
            self._similarity_cache.clear()
            logger.info("Similarity cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._similarity_cache:
            return {"caching_enabled": False}

        return {
            "caching_enabled": True,
            "cache_size": len(self._similarity_cache),
            "max_cache_size": self.config.max_cache_size,
            "cache_utilization": len(self._similarity_cache) / self.config.max_cache_size
        }

    def update_config(self, new_config: SimilarityConfig):
        """Update framework configuration."""
        self.config = new_config
        self._initialize_metrics()  # Reinitialize with new config
        self.clear_cache()  # Clear cache as config changed
        logger.info("Similarity framework configuration updated")


# Convenience functions for common use cases

def calculate_similarity(
        item1: MemoryItem,
        item2: MemoryItem,
        config: Optional[SimilarityConfig] = None,
        graph_store=None
) -> float:
    """Convenience function to calculate similarity between two items."""
    framework = EnhancedSimilarityFramework(config, graph_store)
    return framework.calculate_similarity(item1, item2)


def find_similar_items(
        target_item: MemoryItem,
        candidate_items: List[MemoryItem],
        threshold: float = 0.5,
        max_results: int = 10,
        config: Optional[SimilarityConfig] = None,
        graph_store=None
) -> List[Tuple[MemoryItem, float]]:
    """Convenience function to find similar items."""
    framework = EnhancedSimilarityFramework(config, graph_store)
    return framework.find_similar_items(target_item, candidate_items, threshold, max_results)


def cluster_similar_items(
        items: List[MemoryItem],
        similarity_threshold: float = 0.8,
        config: Optional[SimilarityConfig] = None,
        graph_store=None
) -> List[List[MemoryItem]]:
    """Convenience function to cluster similar items."""
    framework = EnhancedSimilarityFramework(config, graph_store)
    return framework.cluster_items(items, similarity_threshold)
