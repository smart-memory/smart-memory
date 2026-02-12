"""
Candidate Scorer for CFS-3b Recommendation Engine.

Provides multi-metric scoring for procedure promotion candidates.
"""

import logging
import math
from datetime import datetime, timezone
from typing import List, Optional

from smartmemory.models.memory_item import MemoryItem
from smartmemory.similarity.framework import EnhancedSimilarityFramework, SimilarityConfig
from smartmemory.procedures.models import CandidateScores, PatternDetectorConfig

logger = logging.getLogger(__name__)


class CandidateScorer:
    """
    Multi-metric scorer for procedure promotion candidates.

    Calculates composite scores based on:
    - Frequency: How often the pattern appears
    - Consistency: How similar items within the cluster are
    - Recency: Preference for recent patterns
    - Agent Workflow: Boost for tool/skill patterns
    """

    def __init__(
        self,
        config: Optional[PatternDetectorConfig] = None,
        similarity_framework: Optional[EnhancedSimilarityFramework] = None,
    ):
        self.config = config or PatternDetectorConfig()

        # Use provided framework or create one
        if similarity_framework:
            self.similarity = similarity_framework
        else:
            sim_config = SimilarityConfig(similarity_threshold=self.config.cluster_threshold)
            self.similarity = EnhancedSimilarityFramework(sim_config)

    def score_cluster(self, cluster: List[MemoryItem], total_items: int) -> CandidateScores:
        """
        Calculate composite scores for a cluster of items.

        Args:
            cluster: List of similar memory items forming a candidate
            total_items: Total number of working memory items analyzed

        Returns:
            CandidateScores with all metric breakdowns
        """
        if not cluster or total_items == 0:
            return CandidateScores()

        # Calculate individual metrics
        frequency_score = self._calculate_frequency_score(len(cluster), total_items)
        consistency_score = self._calculate_consistency_score(cluster)
        recency_score = self._calculate_recency_score(cluster)
        agent_workflow_score = self._calculate_agent_workflow_score(cluster)

        # Calculate weighted recommendation score
        recommendation_score = (
            frequency_score * self.config.frequency_weight
            + consistency_score * self.config.consistency_weight
            + recency_score * self.config.recency_weight
            + agent_workflow_score * self.config.agent_workflow_weight
        )

        return CandidateScores(
            recommendation_score=recommendation_score,
            frequency_score=frequency_score,
            consistency_score=consistency_score,
            recency_score=recency_score,
            agent_workflow_score=agent_workflow_score,
        )

    def _calculate_frequency_score(self, cluster_size: int, total_items: int) -> float:
        """
        Calculate frequency score based on cluster size relative to total items.

        Uses log scaling to prevent very large clusters from dominating.
        """
        if total_items == 0:
            return 0.0

        # Raw frequency ratio
        raw_frequency = cluster_size / total_items

        # Log scaling to normalize (prevents domination by large clusters)
        # Formula: log(1 + cluster_size) / log(1 + total_items)
        log_frequency = math.log(1 + cluster_size) / math.log(1 + total_items)

        # Combine raw and log-scaled frequencies
        # Bias toward log scaling but keep some raw signal
        return min(1.0, 0.4 * raw_frequency + 0.6 * log_frequency)

    def _calculate_consistency_score(self, cluster: List[MemoryItem]) -> float:
        """
        Calculate consistency score based on intra-cluster similarity.

        Measures how similar items within the cluster are to each other.
        Higher consistency = more coherent pattern.
        """
        if len(cluster) < 2:
            return 1.0  # Single item is perfectly consistent

        # Calculate pairwise similarities
        similarities = []
        for i, item1 in enumerate(cluster):
            for item2 in cluster[i + 1 :]:
                try:
                    similarity = self.similarity.calculate_similarity(item1, item2)
                    similarities.append(similarity)
                except Exception as e:
                    logger.warning(f"Failed to calculate similarity: {e}")
                    continue

        if not similarities:
            return 0.5  # Default to medium consistency if calculation fails

        # Mean similarity as consistency score
        return sum(similarities) / len(similarities)

    def _calculate_recency_score(self, cluster: List[MemoryItem]) -> float:
        """
        Calculate recency score with preference for recent patterns.

        Uses exponential decay based on time since most recent item.
        """
        now = datetime.now(timezone.utc)
        max_age_days = self.config.days_back

        # Find most recent item
        latest_time = None
        for item in cluster:
            tx_time = getattr(item, "transaction_time", None)
            if tx_time:
                if not isinstance(tx_time, datetime):
                    try:
                        tx_time = datetime.fromisoformat(str(tx_time))
                    except (ValueError, TypeError):
                        continue
                if tx_time.tzinfo is None:
                    tx_time = tx_time.replace(tzinfo=timezone.utc)
                if latest_time is None or tx_time > latest_time:
                    latest_time = tx_time

        if latest_time is None:
            return 0.5  # Default to medium recency if no timestamps

        # Calculate age in days
        age_days = (now - latest_time).total_seconds() / 86400

        # Exponential decay with half-life of max_age_days / 3
        half_life = max_age_days / 3
        decay_factor = 0.5 ** (age_days / half_life)

        return min(1.0, decay_factor)

    def _calculate_agent_workflow_score(self, cluster: List[MemoryItem]) -> float:
        """
        Calculate agent workflow score based on presence of skills/tools.

        Patterns with explicit tool/skill usage are more valuable as procedures.
        """
        if not cluster:
            return 0.0

        total_skills = 0
        total_tools = 0
        items_with_workflow = 0

        for item in cluster:
            metadata = getattr(item, "metadata", {}) or {}
            skills = metadata.get("skills", [])
            tools = metadata.get("tools", [])

            if skills or tools:
                items_with_workflow += 1
                total_skills += len(skills)
                total_tools += len(tools)

        if not items_with_workflow:
            # Check content for workflow indicators
            return self._content_workflow_score(cluster)

        # Ratio of items with explicit workflow metadata
        workflow_ratio = items_with_workflow / len(cluster)

        # Average density of skills/tools per item
        avg_density = (total_skills + total_tools) / len(cluster)
        density_score = min(1.0, avg_density / 3)  # Normalize to max 3 skills/tools

        # Combine ratio and density
        return 0.6 * workflow_ratio + 0.4 * density_score

    def _content_workflow_score(self, cluster: List[MemoryItem]) -> float:
        """
        Fallback scoring based on content analysis for workflow patterns.

        Looks for common workflow indicators in content.
        """
        workflow_keywords = {
            "run",
            "execute",
            "call",
            "invoke",
            "trigger",
            "api",
            "endpoint",
            "function",
            "method",
            "step",
            "process",
            "workflow",
            "task",
            "command",
            "action",
            "operation",
            "request",
            "then",
            "after",
            "before",
            "next",
            "finally",
        }

        matches = 0
        for item in cluster:
            content = getattr(item, "content", "") or ""
            content_lower = content.lower()
            words = set(content_lower.split())
            if words & workflow_keywords:
                matches += 1

        return matches / len(cluster) if cluster else 0.0
