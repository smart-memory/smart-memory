"""
Temporal relationship queries for SmartMemory.

Enables querying relationships as they existed at specific times,
tracking relationship changes, and performing temporal joins.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemporalRelationship:
    """Represents a relationship at a specific time."""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    valid_time_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_time_end: Optional[datetime] = None
    transaction_time_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transaction_time_end: Optional[datetime] = None
    
    def is_valid_at(self, time: datetime) -> bool:
        """Check if relationship is valid at given time."""
        if self.valid_time_start > time:
            return False
        if self.valid_time_end and self.valid_time_end <= time:
            return False
        return True
    
    def overlaps_with(self, other: 'TemporalRelationship') -> bool:
        """Check if this relationship's valid time overlaps with another."""
        # Check if time ranges overlap
        if self.valid_time_end and other.valid_time_start >= self.valid_time_end:
            return False
        if other.valid_time_end and self.valid_time_start >= other.valid_time_end:
            return False
        return True


class TemporalRelationshipQueries:
    """
    Query relationships across time.
    
    Features:
    - Query relationships at specific times
    - Track relationship changes
    - Find temporal patterns
    - Perform bi-temporal joins
    """
    
    def __init__(self, graph_backend):
        """
        Initialize temporal relationship queries.
        
        Args:
            graph_backend: Graph database backend
        """
        self.graph = graph_backend
        self._relationship_cache: Dict[str, List[TemporalRelationship]] = {}
    
    def get_relationships_at_time(
        self,
        item_id: str,
        time: datetime,
        relationship_type: Optional[str] = None,
        direction: str = 'both'
    ) -> List[TemporalRelationship]:
        """
        Get all relationships for an item as they existed at a specific time.
        
        Args:
            item_id: Memory item ID
            time: Time point to query
            relationship_type: Optional filter by relationship type
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of relationships valid at that time
            
        Example:
            # What relationships did this memory have on Sept 1st?
            rels = temporal_rels.get_relationships_at_time(
                "memory123",
                datetime(2024, 9, 1)
            )
        """
        try:
            # Query all relationships for this item
            all_relationships = self._query_relationships(
                item_id,
                relationship_type,
                direction
            )
            
            # Filter to those valid at the specified time
            valid_relationships = [
                rel for rel in all_relationships
                if rel.is_valid_at(time)
            ]
            
            logger.info(
                f"Found {len(valid_relationships)} relationships for {item_id} "
                f"at {time.isoformat()}"
            )
            
            return valid_relationships
            
        except Exception as e:
            logger.error(f"Error querying relationships at time: {e}")
            return []
    
    def get_relationship_history(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[str] = None
    ) -> List[TemporalRelationship]:
        """
        Get complete history of relationships between two items.
        
        Args:
            source_id: Source item ID
            target_id: Target item ID
            relationship_type: Optional filter by type
            
        Returns:
            List of all relationship versions, ordered by time
            
        Example:
            # How has the relationship between these items changed?
            history = temporal_rels.get_relationship_history(
                "memory123",
                "memory456"
            )
        """
        try:
            query = """
            MATCH (s {item_id: $source_id})-[r]->(t {item_id: $target_id})
            WHERE $rel_type IS NULL OR type(r) = $rel_type
            RETURN r, type(r) as rel_type
            ORDER BY r.transaction_time_start DESC
            """
            
            params = {
                'source_id': source_id,
                'target_id': target_id,
                'rel_type': relationship_type
            }
            
            result = self.graph.execute_query(query, params)
            
            relationships = []
            for row in result:
                rel_data = row['r']
                rel = TemporalRelationship(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=row['rel_type'],
                    properties=rel_data.get('properties', {}),
                    valid_time_start=self._parse_time(rel_data.get('valid_time_start')),
                    valid_time_end=self._parse_time(rel_data.get('valid_time_end')),
                    transaction_time_start=self._parse_time(rel_data.get('transaction_time_start')),
                    transaction_time_end=self._parse_time(rel_data.get('transaction_time_end'))
                )
                relationships.append(rel)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error getting relationship history: {e}")
            return []
    
    def find_temporal_patterns(
        self,
        item_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Find patterns in how relationships changed over time.
        
        Args:
            item_id: Memory item ID
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Dictionary with pattern analysis
            
        Example:
            patterns = temporal_rels.find_temporal_patterns(
                "memory123",
                datetime(2024, 9, 1),
                datetime(2024, 9, 30)
            )
        """
        try:
            # Get all relationships in time range
            all_rels = self._query_relationships_in_range(
                item_id,
                start_time,
                end_time
            )
            
            # Analyze patterns
            patterns = {
                'total_relationships': len(all_rels),
                'relationship_types': {},
                'created': [],
                'deleted': [],
                'modified': []
            }
            
            for rel in all_rels:
                # Count by type
                rel_type = rel.relationship_type
                patterns['relationship_types'][rel_type] = \
                    patterns['relationship_types'].get(rel_type, 0) + 1
                
                # Categorize changes
                if rel.transaction_time_start >= start_time and \
                   rel.transaction_time_start <= end_time:
                    if rel.valid_time_end is None:
                        patterns['created'].append({
                            'type': rel_type,
                            'target': rel.target_id,
                            'time': rel.transaction_time_start.isoformat()
                        })
                
                if rel.valid_time_end and \
                   rel.valid_time_end >= start_time and \
                   rel.valid_time_end <= end_time:
                    patterns['deleted'].append({
                        'type': rel_type,
                        'target': rel.target_id,
                        'time': rel.valid_time_end.isoformat()
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error finding temporal patterns: {e}")
            return {}
    
    def temporal_join(
        self,
        item_ids: List[str],
        start_time: datetime,
        end_time: datetime,
        join_type: str = 'overlap'
    ) -> List[Dict[str, Any]]:
        """
        Perform bi-temporal join to find items with overlapping valid times.
        
        Args:
            item_ids: List of item IDs to join
            start_time: Start of time range
            end_time: End of time range
            join_type: 'overlap' (any overlap) or 'concurrent' (fully concurrent)
            
        Returns:
            List of join results with temporal information
            
        Example:
            # Find memories that were valid at the same time
            results = temporal_rels.temporal_join(
                ["memory1", "memory2", "memory3"],
                datetime(2024, 9, 1),
                datetime(2024, 9, 30),
                join_type='overlap'
            )
        """
        try:
            if len(item_ids) < 2:
                logger.warning("Need at least 2 items for temporal join")
                return []
            
            # Get all relationships for each item in time range
            item_relationships = {}
            for item_id in item_ids:
                rels = self._query_relationships_in_range(
                    item_id,
                    start_time,
                    end_time
                )
                item_relationships[item_id] = rels
            
            # Find overlapping relationships
            join_results = []
            
            if join_type == 'overlap':
                # Find any temporal overlap
                for i, item1 in enumerate(item_ids):
                    for item2 in item_ids[i+1:]:
                        overlaps = self._find_overlaps(
                            item_relationships[item1],
                            item_relationships[item2]
                        )
                        
                        if overlaps:
                            join_results.append({
                                'item1': item1,
                                'item2': item2,
                                'overlaps': overlaps,
                                'overlap_count': len(overlaps)
                            })
            
            elif join_type == 'concurrent':
                # Find fully concurrent relationships
                concurrent = self._find_concurrent(
                    item_relationships,
                    start_time,
                    end_time
                )
                join_results = concurrent
            
            logger.info(f"Temporal join found {len(join_results)} results")
            return join_results
            
        except Exception as e:
            logger.error(f"Error in temporal join: {e}")
            return []
    
    def find_co_occurring_relationships(
        self,
        item_id: str,
        time_window: int = 3600  # seconds
    ) -> List[Dict[str, Any]]:
        """
        Find relationships that were created or modified around the same time.
        
        Args:
            item_id: Memory item ID
            time_window: Time window in seconds (default 1 hour)
            
        Returns:
            List of co-occurring relationship groups
            
        Example:
            # Find relationships created within 1 hour of each other
            co_occurring = temporal_rels.find_co_occurring_relationships(
                "memory123",
                time_window=3600
            )
        """
        try:
            all_rels = self._query_relationships(item_id)
            
            # Group by time windows
            groups = []
            sorted_rels = sorted(
                all_rels,
                key=lambda r: r.transaction_time_start
            )
            
            current_group = []
            for rel in sorted_rels:
                if not current_group:
                    current_group.append(rel)
                else:
                    time_diff = (
                        rel.transaction_time_start - 
                        current_group[0].transaction_time_start
                    ).total_seconds()
                    
                    if time_diff <= time_window:
                        current_group.append(rel)
                    else:
                        if len(current_group) > 1:
                            groups.append({
                                'time': current_group[0].transaction_time_start.isoformat(),
                                'count': len(current_group),
                                'relations': [
                                    {
                                        'type': r.relationship_type,
                                        'target': r.target_id
                                    }
                                    for r in current_group
                                ]
                            })
                        current_group = [rel]
            
            # Add last group
            if len(current_group) > 1:
                groups.append({
                    'time': current_group[0].transaction_time_start.isoformat(),
                    'count': len(current_group),
                    'relations': [
                        {
                            'type': r.relationship_type,
                            'target': r.target_id
                        }
                        for r in current_group
                    ]
                })
            
            return groups
            
        except Exception as e:
            logger.error(f"Error finding co-occurring relationships: {e}")
            return []
    
    # Helper methods
    
    def _query_relationships(
        self,
        item_id: str,
        relationship_type: Optional[str] = None,
        direction: str = 'both'
    ) -> List[TemporalRelationship]:
        """Query all relationships for an item."""
        try:
            if direction == 'outgoing':
                query = """
                MATCH (s {item_id: $item_id})-[r]->(t)
                WHERE $rel_type IS NULL OR type(r) = $rel_type
                RETURN s.item_id as source, t.item_id as target, r, type(r) as rel_type
                """
            elif direction == 'incoming':
                query = """
                MATCH (s)-[r]->(t {item_id: $item_id})
                WHERE $rel_type IS NULL OR type(r) = $rel_type
                RETURN s.item_id as source, t.item_id as target, r, type(r) as rel_type
                """
            else:  # both
                query = """
                MATCH (s {item_id: $item_id})-[r]-(t)
                WHERE $rel_type IS NULL OR type(r) = $rel_type
                RETURN s.item_id as source, t.item_id as target, r, type(r) as rel_type
                """
            
            params = {
                'item_id': item_id,
                'rel_type': relationship_type
            }
            
            result = self.graph.execute_query(query, params)
            
            relationships = []
            for row in result:
                rel_data = row['r']
                rel = TemporalRelationship(
                    source_id=row['source'],
                    target_id=row['target'],
                    relationship_type=row['rel_type'],
                    properties=rel_data.get('properties', {}),
                    valid_time_start=self._parse_time(
                        rel_data.get('valid_time_start')
                    ),
                    valid_time_end=self._parse_time(
                        rel_data.get('valid_time_end')
                    ),
                    transaction_time_start=self._parse_time(
                        rel_data.get('transaction_time_start')
                    ),
                    transaction_time_end=self._parse_time(
                        rel_data.get('transaction_time_end')
                    )
                )
                relationships.append(rel)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error querying relationships: {e}")
            return []
    
    def _query_relationships_in_range(
        self,
        item_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[TemporalRelationship]:
        """Query relationships within a time range."""
        all_rels = self._query_relationships(item_id)
        
        # Filter to time range
        return [
            rel for rel in all_rels
            if (rel.valid_time_start <= end_time and
                (rel.valid_time_end is None or rel.valid_time_end >= start_time))
        ]
    
    def _find_overlaps(
        self,
        rels1: List[TemporalRelationship],
        rels2: List[TemporalRelationship]
    ) -> List[Dict[str, Any]]:
        """Find overlapping relationships between two lists."""
        overlaps = []
        
        for r1 in rels1:
            for r2 in rels2:
                if r1.overlaps_with(r2):
                    overlaps.append({
                        'rel1_type': r1.relationship_type,
                        'rel1_target': r1.target_id,
                        'rel2_type': r2.relationship_type,
                        'rel2_target': r2.target_id,
                        'overlap_start': max(
                            r1.valid_time_start,
                            r2.valid_time_start
                        ).isoformat(),
                        'overlap_end': min(
                            r1.valid_time_end or datetime.max.replace(tzinfo=timezone.utc),
                            r2.valid_time_end or datetime.max.replace(tzinfo=timezone.utc)
                        ).isoformat() if (r1.valid_time_end and r2.valid_time_end) else None
                    })
        
        return overlaps
    
    def _find_concurrent(
        self,
        item_relationships: Dict[str, List[TemporalRelationship]],
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Find relationships that are concurrent across all items."""
        # Implementation for finding fully concurrent relationships
        # This is more complex and would need specific business logic
        return []
    
    def _parse_time(self, time_str: Optional[str]) -> Optional[datetime]:
        """Parse time string to datetime."""
        if not time_str:
            return None
        
        try:
            if isinstance(time_str, datetime):
                return time_str
            return datetime.fromisoformat(time_str)
        except Exception as e:
            logger.warning(f"Could not parse time: {time_str}")
            return None
