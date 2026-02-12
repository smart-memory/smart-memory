"""
Pattern Detector for CFS-3b Recommendation Engine.

Detects repeated patterns in working memory that are candidates
for procedure promotion using the EnhancedSimilarityFramework.
"""

import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, TYPE_CHECKING

from smartmemory.models.memory_item import MemoryItem
from smartmemory.similarity.framework import EnhancedSimilarityFramework, SimilarityConfig
from smartmemory.procedures.models import (
    ProcedureCandidate,
    PatternDetectorConfig,
)
from smartmemory.procedures.candidate_scorer import CandidateScorer
from smartmemory.procedures.candidate_namer import CandidateNamer

if TYPE_CHECKING:
    from smartmemory import SmartMemory

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    Detects repeated patterns in working memory for procedure promotion.

    Uses EnhancedSimilarityFramework for clustering similar items and
    scores clusters based on frequency, consistency, recency, and
    agent workflow patterns.

    Usage:
        ```python
        from smartmemory import SmartMemory
        from smartmemory.procedures import PatternDetector

        memory = SmartMemory()
        detector = PatternDetector(memory)

        candidates = detector.detect_candidates(
            days_back=30,
            min_cluster_size=3,
            min_score=0.6
        )

        for candidate in candidates:
            print(f"{candidate.suggested_name}: {candidate.scores.recommendation_score}")
        ```
    """

    def __init__(self, smart_memory: Optional["SmartMemory"] = None, config: Optional[PatternDetectorConfig] = None):
        """
        Initialize the pattern detector.

        Args:
            smart_memory: SmartMemory instance for fetching working memory items.
                         If None, items must be provided directly to detect_candidates.
            config: Configuration for pattern detection thresholds and weights.
        """
        self.memory = smart_memory
        self.config = config or PatternDetectorConfig()

        # Initialize similarity framework
        sim_config = SimilarityConfig(
            similarity_threshold=self.config.cluster_threshold,
            high_similarity_threshold=self.config.cluster_threshold,
        )
        self.similarity = EnhancedSimilarityFramework(sim_config)

        # Initialize scorer and namer
        self.scorer = CandidateScorer(self.config, self.similarity)
        self.namer = CandidateNamer()

    def detect_candidates(
        self,
        days_back: Optional[int] = None,
        min_cluster_size: Optional[int] = None,
        min_score: Optional[float] = None,
        limit: Optional[int] = None,
        items: Optional[List[MemoryItem]] = None,
    ) -> List[ProcedureCandidate]:
        """
        Detect procedure promotion candidates from working memory patterns.

        Args:
            days_back: Look back period in days (default: config value)
            min_cluster_size: Minimum items in cluster (default: config value)
            min_score: Minimum recommendation score (default: config value)
            limit: Maximum candidates to return (default: config value)
            items: Optional list of items to analyze (bypasses memory fetch)

        Returns:
            List of ProcedureCandidate sorted by recommendation score (descending)
        """
        # Use config defaults if not specified
        days_back = days_back if days_back is not None else self.config.days_back
        min_cluster_size = min_cluster_size if min_cluster_size is not None else self.config.min_cluster_size
        min_score = min_score if min_score is not None else self.config.min_score
        limit = limit if limit is not None else self.config.max_candidates

        # Fetch or use provided items
        if items is not None:
            working_items = items
        else:
            working_items = self._fetch_working_items(days_back)

        if not working_items:
            logger.info("No working memory items found for pattern detection")
            return []

        logger.info(f"Analyzing {len(working_items)} working memory items for patterns")

        # Cluster items by similarity
        clusters = self.similarity.cluster_items(working_items, similarity_threshold=self.config.cluster_threshold)

        logger.info(f"Found {len(clusters)} clusters")

        # Score and filter clusters
        candidates = []
        for cluster in clusters:
            # Skip small clusters
            if len(cluster) < min_cluster_size:
                continue

            # Build candidate
            candidate = self._build_candidate(cluster, len(working_items))

            # Filter by score
            if candidate.scores.recommendation_score >= min_score:
                candidates.append(candidate)

        # Sort by recommendation score (descending)
        candidates.sort(key=lambda c: c.scores.recommendation_score, reverse=True)

        # Limit results
        candidates = candidates[:limit]

        logger.info(f"Detected {len(candidates)} procedure candidates")
        return candidates

    def _fetch_working_items(self, days_back: int) -> List[MemoryItem]:
        """
        Fetch working memory items from the past N days.

        Args:
            days_back: Number of days to look back

        Returns:
            List of MemoryItem from working memory
        """
        if self.memory is None:
            logger.warning("No SmartMemory instance provided; cannot fetch items")
            return []

        try:
            # Calculate cutoff time
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

            # Search for working memory items
            # Using empty query with memory_type filter
            results = self.memory.search(
                query="",
                top_k=500,  # Reasonable limit for clustering
                memory_type="working",
            )

            # Filter by date if items have timestamps
            filtered = []
            for item in results:
                # Convert dict to MemoryItem if needed
                if isinstance(item, dict):
                    item = MemoryItem(**item)

                # Check transaction time
                tx_time = getattr(item, "transaction_time", None)
                if tx_time:
                    if not isinstance(tx_time, datetime):
                        try:
                            tx_time = datetime.fromisoformat(str(tx_time))
                        except (ValueError, TypeError):
                            tx_time = None

                    if tx_time and tx_time.tzinfo is None:
                        tx_time = tx_time.replace(tzinfo=timezone.utc)

                    # Include if within time window or no timestamp
                    if tx_time is None or tx_time >= cutoff:
                        filtered.append(item)
                else:
                    # Include items without timestamps
                    filtered.append(item)

            return filtered

        except Exception as e:
            logger.error(f"Error fetching working memory items: {e}")
            return []

    def _build_candidate(self, cluster: List[MemoryItem], total_items: int) -> ProcedureCandidate:
        """
        Build a ProcedureCandidate from a cluster of items.

        Args:
            cluster: List of similar memory items
            total_items: Total number of items analyzed

        Returns:
            ProcedureCandidate with scores and metadata
        """
        # Generate stable cluster ID from sorted item IDs
        # This ensures the same cluster produces the same ID across API calls
        item_ids = sorted(getattr(item, "item_id", "") for item in cluster if getattr(item, "item_id", ""))
        hash_input = "|".join(item_ids)
        cluster_id = hashlib.sha256(hash_input.encode()).hexdigest()[:32]

        # Calculate scores
        scores = self.scorer.score_cluster(cluster, total_items)

        # Generate name and description
        name, description = self.namer.generate_name_and_description(cluster)

        # Get common skills and tools
        common_skills = self.namer.get_common_skills(cluster)
        common_tools = self.namer.get_common_tools(cluster)

        # Get representative content (first item)
        representative = cluster[0]
        representative_content = getattr(representative, "content", "") or ""

        # Get sample item IDs
        sample_ids = [getattr(item, "item_id", "") for item in cluster[: self.config.max_sample_items]]

        # Get date range
        earliest, latest = self._get_date_range(cluster)

        return ProcedureCandidate(
            cluster_id=cluster_id,
            suggested_name=name,
            suggested_description=description,
            representative_content=representative_content,
            item_count=len(cluster),
            scores=scores,
            common_skills=common_skills,
            common_tools=common_tools,
            sample_item_ids=sample_ids,
            earliest_date=earliest,
            latest_date=latest,
        )

    def _get_date_range(self, cluster: List[MemoryItem]) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get the earliest and latest timestamps from cluster items."""
        dates = []

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
                dates.append(tx_time)

        if not dates:
            return None, None

        return min(dates), max(dates)
