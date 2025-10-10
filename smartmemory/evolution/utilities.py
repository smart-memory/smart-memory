"""
Shared utilities for all evolution algorithms to eliminate code duplication.
"""

import math
from datetime import datetime, timezone
from typing import List, Dict

from smartmemory.models.memory_item import MemoryItem


class EvolutionUtilities:
    """Shared utilities for all evolution algorithms."""

    @staticmethod
    def calculate_comprehensive_similarity(item1: MemoryItem, item2: MemoryItem, graph_store=None) -> float:
        """
        Calculate comprehensive similarity between two memory items using enhanced framework.
        
        Uses the new Enhanced Similarity Framework for accurate, multi-dimensional similarity.
        """
        if not item1 or not item2:
            return 0.0

        try:
            from smartmemory.similarity import EnhancedSimilarityFramework, SimilarityConfig

            # Use evolution-optimized config
            config = SimilarityConfig(
                content_weight=0.30,
                semantic_weight=0.35,
                temporal_weight=0.15,
                graph_weight=0.15,
                metadata_weight=0.05,
                similarity_threshold=0.3,  # Lower threshold for evolution decisions
                high_similarity_threshold=0.7
            )

            framework = EnhancedSimilarityFramework(config, graph_store)
            return framework.calculate_similarity(item1, item2)

        except ImportError:
            # Fallback to legacy similarity if enhanced framework not available
            return EvolutionUtilities._legacy_similarity(item1, item2)

    @staticmethod
    def _legacy_similarity(item1: MemoryItem, item2: MemoryItem) -> float:
        """
        Calculate comprehensive similarity between two memory items.
        
        Combines content, metadata, and temporal similarity with weighted scoring.
        """
        if not item1 or not item2:
            return 0.0

        content_sim = EvolutionUtilities._content_similarity(item1.content, item2.content)
        metadata_sim = EvolutionUtilities._metadata_similarity(item1, item2)
        temporal_sim = EvolutionUtilities._temporal_similarity(item1, item2)

        # Weighted combination optimized for evolution decisions
        weighted_similarity = (
                0.5 * content_sim +
                0.3 * metadata_sim +
                0.2 * temporal_sim
        )

        return min(1.0, weighted_similarity)

    @staticmethod
    def _content_similarity(content1: str, content2: str) -> float:
        """Calculate content similarity with enhanced fuzzy matching."""
        if not content1 or not content2:
            return 0.0

        # Normalize content
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Basic Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0.0

        # Enhanced fuzzy matching with partial word matching
        partial_matches = 0
        for word1 in words1:
            for word2 in words2:
                if len(word1) >= 4 and len(word2) >= 4:
                    # Check for partial matches in longer words
                    if word1 in word2 or word2 in word1:
                        partial_matches += 0.5

        # Length bonus for similar content lengths
        len_diff = abs(len(content1) - len(content2))
        max_len = max(len(content1), len(content2))
        length_bonus = 1.0 - (len_diff / max_len) if max_len > 0 else 0.0

        # Combine scores with beneficial fuzziness
        return min(1.0, jaccard + (partial_matches / max(len(words1), len(words2))) * 0.3 + length_bonus * 0.1)

    @staticmethod
    def _metadata_similarity(meta1: Dict, meta2: Dict) -> float:
        """Calculate metadata similarity."""
        if not meta1 or not meta2:
            return 0.0

        # Compare categories
        cats1 = set(meta1.get('categories', []))
        cats2 = set(meta2.get('categories', []))

        if not cats1 or not cats2:
            return 0.0

        intersection = len(cats1 & cats2)
        union = len(cats1 | cats2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _temporal_similarity(item1: MemoryItem, item2: MemoryItem) -> float:
        """Calculate temporal proximity between two items."""
        try:
            time1 = getattr(item1, 'created_at', None) or getattr(item1, 'transaction_time', None)
            time2 = getattr(item2, 'created_at', None) or getattr(item2, 'transaction_time', None)

            if not time1 or not time2:
                return 0.0

            # Convert to datetime if needed
            if isinstance(time1, str):
                time1 = datetime.fromisoformat(time1.replace('Z', '+00:00'))
            if isinstance(time2, str):
                time2 = datetime.fromisoformat(time2.replace('Z', '+00:00'))

            time_diff = abs((time1 - time2).total_seconds())
            # Proximity decreases exponentially with time difference (1-hour half-life)
            return math.exp(-time_diff / 3600)
        except Exception:
            return 0.0

    @staticmethod
    def safe_archive_item(memory, item: MemoryItem, reason: str):
        """
        Safely archive a memory item with consistent metadata.
        
        Archives rather than deletes for safety and auditability.
        """
        if not item:
            return False

        try:
            # Add archival metadata
            item.metadata['archived'] = True
            item.metadata['archive_reason'] = reason
            item.metadata['archive_timestamp'] = datetime.now(timezone.utc).isoformat()

            # Use memory's archive method if available, otherwise update
            if hasattr(memory, 'archive'):
                return memory.archive(item)
            elif hasattr(memory, 'update'):
                return memory.update(item)
            else:
                # Fallback: mark as archived in metadata
                return True
        except Exception:
            return False

    @staticmethod
    def get_items_by_criteria(memory, criteria: Dict) -> List[MemoryItem]:
        """
        Get memory items matching specified criteria.
        
        Supports filtering by age, type, confidence, etc.
        """
        try:
            # Get all items if method exists
            if hasattr(memory, 'get_all_items'):
                all_items = memory.get_all_items()
            else:
                return []

            filtered_items = []
            for item in all_items:
                if EvolutionUtilities._matches_criteria(item, criteria):
                    filtered_items.append(item)

            return filtered_items
        except Exception:
            return []

    @staticmethod
    def _matches_criteria(item: MemoryItem, criteria: Dict) -> bool:
        """Check if an item matches the specified criteria."""
        try:
            # Age criteria
            if 'max_age_days' in criteria:
                created_at = getattr(item, 'created_at', None) or getattr(item, 'transaction_time', None)
                if created_at:
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    age_days = (datetime.now(timezone.utc) - created_at).days
                    if age_days > criteria['max_age_days']:
                        return False

            # Type criteria
            if 'memory_type' in criteria:
                item_type = item.metadata.get('memory_type', 'unknown')
                if item_type != criteria['memory_type']:
                    return False

            # Confidence criteria
            if 'min_confidence' in criteria:
                confidence = item.metadata.get('confidence', 1.0)
                if confidence < criteria['min_confidence']:
                    return False

            # Archived criteria
            if 'exclude_archived' in criteria and criteria['exclude_archived']:
                if item.metadata.get('archived', False):
                    return False

            return True
        except Exception:
            return False

    @staticmethod
    def find_similarity_clusters(items: List[MemoryItem], similarity_threshold: float = 0.7) -> List[List[MemoryItem]]:
        """
        Find clusters of similar items for consolidation.
        
        Uses comprehensive similarity calculation to group related items.
        """
        if not items:
            return []

        clusters = []
        processed = set()

        for i, item1 in enumerate(items):
            if i in processed:
                continue

            cluster = [item1]
            processed.add(i)

            for j, item2 in enumerate(items[i + 1:], i + 1):
                if j in processed:
                    continue

                similarity = EvolutionUtilities.calculate_comprehensive_similarity(item1, item2)
                if similarity >= similarity_threshold:
                    cluster.append(item2)
                    processed.add(j)

            if len(cluster) > 1:  # Only return clusters with multiple items
                clusters.append(cluster)

        return clusters

    @staticmethod
    def consolidate_cluster(cluster: List[MemoryItem], consolidation_strategy: str = 'merge') -> MemoryItem:
        """
        Consolidate a cluster of similar items into a single item.
        
        Strategies:
        - 'merge': Combine content and metadata
        - 'best': Keep the highest quality item
        - 'newest': Keep the most recent item
        """
        if not cluster:
            return None

        if len(cluster) == 1:
            return cluster[0]

        if consolidation_strategy == 'best':
            # Return item with highest confidence
            return max(cluster, key=lambda x: x.metadata.get('confidence', 0.5))

        elif consolidation_strategy == 'newest':
            # Return most recent item
            return max(cluster, key=lambda x: getattr(x, 'created_at', datetime.min))

        else:  # merge strategy
            # Combine content and metadata from all items
            combined_content = []
            combined_metadata = {}

            for item in cluster:
                if item.content and item.content not in combined_content:
                    combined_content.append(item.content)

                # Merge metadata
                for key, value in item.metadata.items():
                    if key not in combined_metadata:
                        combined_metadata[key] = value
                    elif isinstance(value, list) and isinstance(combined_metadata[key], list):
                        combined_metadata[key].extend(value)

            # Create consolidated item
            consolidated = MemoryItem(
                item_id=cluster[0].item_id,  # Use first item's ID
                content='\n'.join(combined_content),
                metadata=combined_metadata
            )

            # Add consolidation metadata
            consolidated.metadata['consolidated'] = True
            consolidated.metadata['source_items'] = [item.item_id for item in cluster]
            consolidated.metadata['consolidation_timestamp'] = datetime.now(timezone.utc).isoformat()

            return consolidated
