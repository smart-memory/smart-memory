"""
Temporal query utilities for SmartMemory.

Provides user-friendly methods for time-travel queries, audit trails,
and temporal analysis.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

from smartmemory.temporal.version_tracker import VersionTracker, Version
from smartmemory.temporal.relationships import TemporalRelationshipQueries

logger = logging.getLogger(__name__)


@dataclass
class TemporalVersion:
    """Represents a version of a memory at a specific time."""
    item_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    valid_time_start: Optional[datetime] = None
    valid_time_end: Optional[datetime] = None
    transaction_time_start: Optional[datetime] = None
    transaction_time_end: Optional[datetime] = None
    version: int = 1


@dataclass
class TemporalChange:
    """Represents a change between two versions."""
    item_id: str
    timestamp: datetime
    change_type: str  # 'created', 'updated', 'deleted'
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    changed_fields: List[str] = field(default_factory=list)


class TemporalQueries:
    """
    User-friendly temporal query interface.
    
    Provides convenient methods for:
    - Time-travel queries
    - History tracking
    - Audit trails
    - Version comparison
    - Rollback operations
    
    Example:
        temporal = TemporalQueries(memory)
        
        # Get history
        history = temporal.get_history("item123")
        
        # Time travel
        past_state = temporal.at_time("2024-09-01")
        
        # Find changes
        changes = temporal.get_changes("item123", since="2024-09-01")
    """
    
    def __init__(self, memory_system):
        """
        Initialize temporal queries.
        
        Args:
            memory_system: SmartMemory instance
        """
        self.memory = memory_system
        self.version_tracker = VersionTracker(memory_system.graph) if hasattr(memory_system, 'graph') else None
        self.relationships = TemporalRelationshipQueries(memory_system.graph) if hasattr(memory_system, 'graph') else None
    
    def get_history(
        self, 
        item_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[TemporalVersion]:
        """
        Get complete history of a memory item.
        
        Args:
            item_id: Memory item ID
            start_time: Start of time range (ISO format)
            end_time: End of time range (ISO format)
            
        Returns:
            List of versions ordered by time (newest first)
            
        Example:
            history = temporal.get_history("item123")
            for version in history:
                print(f"Version {version.version} at {version.transaction_time_start}")
        """
        try:
            # Query backend for all versions
            versions = self._query_versions(item_id, start_time, end_time)
            
            # Sort by transaction time (newest first)
            return sorted(
                versions, 
                key=lambda v: v.transaction_time_start or datetime.min, 
                reverse=True
            )
        except Exception as e:
            logger.error(f"Error getting history for {item_id}: {e}")
            return []
    
    def at_time(
        self, 
        time: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Get all memories as they existed at a specific time.
        
        Args:
            time: Time point (ISO format: "2024-09-01T00:00:00" or "2024-09-01")
            filters: Optional filters (user_id, memory_type, etc.)
            
        Returns:
            List of memory items as they existed at that time
            
        Example:
            # What did the AI know on Sept 1st?
            past_memories = temporal.at_time("2024-09-01T00:00:00")
        """
        try:
            # Parse time string
            dt = self._parse_time(time)
            
            logger.info(f"Querying memories at time: {dt.isoformat()}")
            
            if not self.version_tracker:
                logger.warning("Version tracker not available")
                return []
            
            # Query all memory items and get their version at the specified time
            # This requires querying the graph for all items
            try:
                query = """
                MATCH (m:Memory)
                RETURN m.item_id as item_id
                """
                result = self.memory.graph.execute_query(query, {})
                
                memories_at_time = []
                for row in result:
                    item_id = row['item_id']
                    version = self.version_tracker.get_version_at_time(item_id, dt)
                    if version:
                        memories_at_time.append(version)
                
                return memories_at_time
                
            except Exception as e:
                logger.error(f"Error querying graph: {e}")
                return []
            
        except Exception as e:
            logger.error(f"Error querying at time {time}: {e}")
            return []
    
    def search_temporal(
        self,
        query: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search memories across time with temporal filtering.
        
        Args:
            query: Search query string
            start_time: Start of time range (ISO format)
            end_time: End of time range (ISO format)
            filters: Optional filters (user_id, memory_type, etc.)
            
        Returns:
            List of search results with temporal metadata
            
        Example:
            # Find all mentions of "Python" in September
            results = temporal.search_temporal(
                "Python",
                start_time="2024-09-01",
                end_time="2024-09-30"
            )
        """
        try:
            # Parse time range
            start_dt = self._parse_time(start_time) if start_time else None
            end_dt = self._parse_time(end_time) if end_time else None
            
            logger.info(f"Temporal search: '{query}' from {start_dt} to {end_dt}")
            
            # Perform regular search
            search_results = self.memory.search(query, **(filters or {}))
            
            # Filter and enrich with temporal information
            temporal_results = []
            for result in search_results:
                item_id = result.item_id if hasattr(result, 'item_id') else result.get('item_id')
                
                if not item_id:
                    continue
                
                # Get versions in time range
                if self.version_tracker:
                    versions = self.version_tracker.get_versions(
                        item_id,
                        start_time=start_dt,
                        end_time=end_dt
                    )
                    
                    for version in versions:
                        temporal_results.append({
                            'item_id': item_id,
                            'content': version.content,
                            'metadata': version.metadata,
                            'version': version.version_number,
                            'valid_time_start': version.valid_time_start.isoformat(),
                            'transaction_time': version.transaction_time_start.isoformat(),
                            'relevance_score': getattr(result, 'score', 1.0)
                        })
                else:
                    # Fallback to current version
                    temporal_results.append({
                        'item_id': item_id,
                        'content': result.content if hasattr(result, 'content') else result.get('content'),
                        'metadata': result.metadata if hasattr(result, 'metadata') else result.get('metadata', {}),
                        'version': 1,
                        'relevance_score': getattr(result, 'score', 1.0)
                    })
            
            return temporal_results
            
        except Exception as e:
            logger.error(f"Error in temporal search: {e}")
            return []
    
    def get_changes(
        self,
        item_id: str,
        since: Optional[str] = None,
        until: Optional[str] = None
    ) -> List[TemporalChange]:
        """
        Get all changes to a memory item.
        
        Args:
            item_id: Memory item ID
            since: Start time (ISO format)
            until: End time (ISO format)
            
        Returns:
            List of changes ordered by time
            
        Example:
            # What changed in the last week?
            changes = temporal.get_changes("item123", since="2024-09-27")
            for change in changes:
                print(f"{change.change_type}: {change.changed_fields}")
        """
        try:
            # Get all versions in time range
            versions = self.get_history(item_id, since, until)
            
            if len(versions) < 2:
                return []
            
            # Compare consecutive versions
            changes = []
            for i in range(len(versions) - 1):
                new_ver = versions[i]
                old_ver = versions[i + 1]
                
                change = self._compare_versions(old_ver, new_ver)
                if change:
                    changes.append(change)
            
            return changes
            
        except Exception as e:
            logger.error(f"Error getting changes for {item_id}: {e}")
            return []
    
    def compare_versions(
        self,
        item_id: str,
        time1: str,
        time2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a memory item.
        
        Args:
            item_id: Memory item ID
            time1: First time point
            time2: Second time point
            
        Returns:
            Dict with differences
            
        Example:
            diff = temporal.compare_versions(
                "item123",
                "2024-09-01",
                "2024-10-01"
            )
            print(f"Changed fields: {diff['changed_fields']}")
        """
        try:
            # Get versions at both times
            ver1 = self.memory.graph.get_node(item_id, as_of_time=time1)
            ver2 = self.memory.graph.get_node(item_id, as_of_time=time2)
            
            if not ver1 or not ver2:
                return {'error': 'Version not found at one or both times'}
            
            # Compare
            return self._diff_items(ver1, ver2)
            
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            return {'error': str(e)}
    
    def rollback(
        self,
        item_id: str,
        to_time: str,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Rollback a memory item to a previous state.
        
        Args:
            item_id: Memory item ID
            to_time: Time to rollback to
            dry_run: If True, don't actually rollback (preview only)
            
        Returns:
            Dict with rollback info
            
        Example:
            # Preview rollback
            preview = temporal.rollback("item123", "2024-09-01", dry_run=True)
            
            # Actually rollback
            result = temporal.rollback("item123", "2024-09-01", dry_run=False)
        """
        try:
            # Get version at target time
            target_version = self.memory.graph.get_node(item_id, as_of_time=to_time)
            
            if not target_version:
                return {'error': 'Version not found at that time'}
            
            # Get current version
            current_version = self.memory.graph.get_node(item_id)
            
            if not current_version:
                return {'error': 'Current version not found'}
            
            # Calculate diff
            diff = self._diff_items(current_version, target_version)
            
            if dry_run:
                return {
                    'dry_run': True,
                    'would_change': diff.get('changed_fields', []),
                    'preview': diff
                }
            
            # Actually rollback by updating to target version
            # This creates a new version with the old content
            self.memory.update(target_version)
            
            return {
                'success': True,
                'rolled_back_to': to_time,
                'changes': diff
            }
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return {'error': str(e)}
    
    def get_audit_trail(
        self,
        item_id: str,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get complete audit trail for compliance.
        
        Args:
            item_id: Memory item ID
            include_metadata: Include full metadata
            
        Returns:
            List of audit events
            
        Example:
            trail = temporal.get_audit_trail("item123")
            for event in trail:
                print(f"{event['timestamp']}: {event['action']}")
        """
        try:
            history = self.get_history(item_id)
            
            audit_trail = []
            for i, version in enumerate(history):
                event = {
                    'version': len(history) - i,
                    'timestamp': version.transaction_time_start.isoformat() if version.transaction_time_start else 'unknown',
                    'valid_from': version.valid_time_start.isoformat() if version.valid_time_start else 'unknown',
                    'valid_to': version.valid_time_end.isoformat() if version.valid_time_end else 'current',
                    'action': self._infer_action(version, history[i+1] if i+1 < len(history) else None)
                }
                
                if include_metadata:
                    event['metadata'] = version.metadata
                    event['content_preview'] = version.content[:100] + '...' if len(version.content) > 100 else version.content
                
                audit_trail.append(event)
            
            return audit_trail
            
        except Exception as e:
            logger.error(f"Error getting audit trail: {e}")
            return []
    
    def find_memories_changed_since(
        self,
        since: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Find all memories that changed since a time.
        
        Args:
            since: Time threshold
            filters: Optional filters
            
        Returns:
            List of item IDs that changed
            
        Example:
            # What changed today?
            changed = temporal.find_memories_changed_since("2024-10-04T00:00:00")
        """
        try:
            dt = self._parse_time(since)
            
            # This would query the graph for items with transaction_time > since
            # For now, return empty list as we need graph query implementation
            logger.info(f"Finding memories changed since: {dt.isoformat()}")
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding changed memories: {e}")
            return []
    
    def get_timeline(
        self,
        item_id: str,
        granularity: str = 'day'
    ) -> Dict[str, List[Dict]]:
        """
        Get timeline of changes for visualization.
        
        Args:
            item_id: Memory item ID
            granularity: 'hour', 'day', 'week', 'month'
            
        Returns:
            Timeline data structure
            
        Example:
            timeline = temporal.get_timeline("item123", granularity='day')
            # Returns: {'2024-09-01': [changes], '2024-09-02': [changes], ...}
        """
        try:
            changes = self.get_changes(item_id)
            
            # Group by time period
            timeline = {}
            for change in changes:
                period = self._round_to_granularity(change.timestamp, granularity)
                if period not in timeline:
                    timeline[period] = []
                timeline[period].append({
                    'type': change.change_type,
                    'fields': change.changed_fields,
                    'timestamp': change.timestamp.isoformat()
                })
            
            return timeline
            
        except Exception as e:
            logger.error(f"Error getting timeline: {e}")
            return {}
    
    # Helper methods
    
    def _query_versions(
        self, 
        item_id: str, 
        start: Optional[str], 
        end: Optional[str]
    ) -> List[TemporalVersion]:
        """Query all versions from backend."""
        # For now, we only have access to current version via as_of_time
        # A full implementation would query the graph database for all versions
        
        try:
            current = self.memory.graph.get_node(item_id)
            if current:
                version = TemporalVersion(
                    item_id=item_id,
                    content=getattr(current, 'content', ''),
                    metadata=getattr(current, 'metadata', {}),
                    transaction_time_start=datetime.now(),
                    version=1
                )
                return [version]
        except Exception as e:
            logger.error(f"Error querying versions: {e}")
        
        return []
    
    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime."""
        try:
            # Try ISO format with timezone
            return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except:
            try:
                # Try date only
                return datetime.strptime(time_str, '%Y-%m-%d')
            except:
                try:
                    # Try datetime without timezone
                    return datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
                except:
                    logger.warning(f"Could not parse time: {time_str}, using now")
                    return datetime.now()
    
    def _compare_versions(
        self, 
        old: TemporalVersion, 
        new: TemporalVersion
    ) -> Optional[TemporalChange]:
        """Compare two versions and return change."""
        try:
            changed_fields = []
            
            # Compare content
            if old.content != new.content:
                changed_fields.append('content')
            
            # Compare metadata
            old_meta = old.metadata or {}
            new_meta = new.metadata or {}
            
            for key in set(old_meta.keys()) | set(new_meta.keys()):
                if old_meta.get(key) != new_meta.get(key):
                    changed_fields.append(f'metadata.{key}')
            
            if not changed_fields:
                return None
            
            return TemporalChange(
                item_id=new.item_id,
                timestamp=new.transaction_time_start or datetime.now(),
                change_type='updated',
                old_value={'content': old.content, 'metadata': old_meta},
                new_value={'content': new.content, 'metadata': new_meta},
                changed_fields=changed_fields
            )
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            return None
    
    def _diff_items(self, item1: Any, item2: Any) -> Dict[str, Any]:
        """Calculate diff between two items."""
        try:
            changed_fields = []
            additions = {}
            deletions = {}
            modifications = {}
            
            # Get attributes
            content1 = getattr(item1, 'content', '')
            content2 = getattr(item2, 'content', '')
            meta1 = getattr(item1, 'metadata', {}) or {}
            meta2 = getattr(item2, 'metadata', {}) or {}
            
            # Compare content
            if content1 != content2:
                changed_fields.append('content')
                modifications['content'] = {
                    'old': content1,
                    'new': content2
                }
            
            # Compare metadata
            all_keys = set(meta1.keys()) | set(meta2.keys())
            for key in all_keys:
                if key in meta1 and key not in meta2:
                    deletions[f'metadata.{key}'] = meta1[key]
                    changed_fields.append(f'metadata.{key}')
                elif key in meta2 and key not in meta1:
                    additions[f'metadata.{key}'] = meta2[key]
                    changed_fields.append(f'metadata.{key}')
                elif meta1.get(key) != meta2.get(key):
                    modifications[f'metadata.{key}'] = {
                        'old': meta1[key],
                        'new': meta2[key]
                    }
                    changed_fields.append(f'metadata.{key}')
            
            return {
                'changed_fields': changed_fields,
                'additions': additions,
                'deletions': deletions,
                'modifications': modifications
            }
        except Exception as e:
            logger.error(f"Error diffing items: {e}")
            return {'error': str(e)}
    
    def _infer_action(
        self, 
        version: TemporalVersion, 
        prev_version: Optional[TemporalVersion]
    ) -> str:
        """Infer what action created this version."""
        if prev_version is None:
            return 'created'
        elif version.valid_time_end is not None:
            return 'deleted'
        else:
            return 'updated'
    
    def _round_to_granularity(self, dt: datetime, granularity: str) -> str:
        """Round datetime to granularity."""
        if granularity == 'hour':
            return dt.strftime('%Y-%m-%d %H:00')
        elif granularity == 'day':
            return dt.strftime('%Y-%m-%d')
        elif granularity == 'week':
            return dt.strftime('%Y-W%W')
        elif granularity == 'month':
            return dt.strftime('%Y-%m')
        return dt.isoformat()
