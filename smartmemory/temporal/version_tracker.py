"""
Version tracking system for bi-temporal memory management.

Tracks all versions of memory items with both valid time and transaction time.
Enables full history tracking, version comparison, and temporal queries.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class Version:
    """Represents a single version of a memory item."""
    item_id: str
    version_number: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Bi-temporal timestamps
    valid_time_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_time_end: Optional[datetime] = None  # None = current/valid
    transaction_time_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transaction_time_end: Optional[datetime] = None  # None = current version
    
    # Change tracking
    changed_by: Optional[str] = None
    change_reason: Optional[str] = None
    previous_version: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key in ['valid_time_start', 'valid_time_end', 'transaction_time_start', 'transaction_time_end']:
            if data[key] and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Version':
        """Create from dictionary."""
        # Parse datetime strings
        for key in ['valid_time_start', 'valid_time_end', 'transaction_time_start', 'transaction_time_end']:
            if data.get(key) and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)
    
    def is_current(self) -> bool:
        """Check if this is the current version."""
        return self.transaction_time_end is None
    
    def is_valid_at(self, time: datetime) -> bool:
        """Check if version is valid at given time."""
        if self.valid_time_start > time:
            return False
        if self.valid_time_end and self.valid_time_end <= time:
            return False
        return True


class VersionTracker:
    """
    Tracks versions of memory items with bi-temporal support.
    
    Features:
    - Full version history
    - Bi-temporal tracking (valid time + transaction time)
    - Automatic version numbering
    - Change detection
    - Version comparison
    """
    
    def __init__(self, graph_backend):
        """
        Initialize version tracker.
        
        Args:
            graph_backend: Graph database backend for storage
        """
        self.graph = graph_backend
        self._version_cache: Dict[str, List[Version]] = {}
    
    def create_version(
        self,
        item_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        valid_time_start: Optional[datetime] = None,
        changed_by: Optional[str] = None,
        change_reason: Optional[str] = None
    ) -> Version:
        """
        Create a new version of a memory item.
        
        Args:
            item_id: Memory item ID
            content: Item content
            metadata: Item metadata
            valid_time_start: When this version becomes valid (default: now)
            changed_by: User/system that made the change
            change_reason: Reason for the change
            
        Returns:
            New Version object
        """
        now = datetime.now(timezone.utc)
        
        # Get previous versions to determine version number
        previous_versions = self.get_versions(item_id)
        
        if previous_versions:
            # Close the previous current version
            current_version = previous_versions[0]  # Most recent
            if current_version.is_current():
                self._close_version(current_version, now)
            
            version_number = max(v.version_number for v in previous_versions) + 1
            previous_version_num = current_version.version_number
        else:
            version_number = 1
            previous_version_num = None
        
        # Create new version
        version = Version(
            item_id=item_id,
            version_number=version_number,
            content=content,
            metadata=metadata or {},
            valid_time_start=valid_time_start or now,
            valid_time_end=None,  # Open-ended
            transaction_time_start=now,
            transaction_time_end=None,  # Current version
            changed_by=changed_by,
            change_reason=change_reason,
            previous_version=previous_version_num
        )
        
        # Store in graph
        self._store_version(version)
        
        # Invalidate cache
        if item_id in self._version_cache:
            del self._version_cache[item_id]
        
        logger.info(f"Created version {version_number} for item {item_id}")
        return version
    
    def get_versions(
        self,
        item_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        include_closed: bool = True
    ) -> List[Version]:
        """
        Get all versions of an item.
        
        Args:
            item_id: Memory item ID
            start_time: Filter by transaction time start
            end_time: Filter by transaction time end
            include_closed: Include closed (non-current) versions
            
        Returns:
            List of versions, ordered by version number (newest first)
        """
        # Check cache
        if item_id in self._version_cache and not (start_time or end_time):
            return self._version_cache[item_id]
        
        # Query from graph
        versions = self._query_versions(item_id, start_time, end_time, include_closed)
        
        # Sort by version number (newest first)
        versions.sort(key=lambda v: v.version_number, reverse=True)
        
        # Cache if no time filters
        if not (start_time or end_time):
            self._version_cache[item_id] = versions
        
        return versions
    
    def get_version_at_time(
        self,
        item_id: str,
        time: datetime,
        time_type: str = 'transaction'
    ) -> Optional[Version]:
        """
        Get the version that was current at a specific time.
        
        Args:
            item_id: Memory item ID
            time: Time point to query
            time_type: 'transaction' or 'valid' time
            
        Returns:
            Version that was current at that time, or None
        """
        versions = self.get_versions(item_id, include_closed=True)
        
        for version in versions:
            if time_type == 'transaction':
                # Check transaction time
                if version.transaction_time_start <= time:
                    if version.transaction_time_end is None or version.transaction_time_end > time:
                        return version
            else:  # valid time
                if version.is_valid_at(time):
                    return version
        
        return None
    
    def get_current_version(self, item_id: str) -> Optional[Version]:
        """
        Get the current (latest) version of an item.
        
        Args:
            item_id: Memory item ID
            
        Returns:
            Current version or None
        """
        versions = self.get_versions(item_id, include_closed=False)
        return versions[0] if versions else None
    
    def compare_versions(
        self,
        item_id: str,
        version1: int,
        version2: int
    ) -> Dict[str, Any]:
        """
        Compare two versions of an item.
        
        Args:
            item_id: Memory item ID
            version1: First version number
            version2: Second version number
            
        Returns:
            Dictionary with comparison results
        """
        versions = self.get_versions(item_id)
        v1 = next((v for v in versions if v.version_number == version1), None)
        v2 = next((v for v in versions if v.version_number == version2), None)
        
        if not v1 or not v2:
            return {"error": "One or both versions not found"}
        
        # Compare content
        content_changed = v1.content != v2.content
        
        # Compare metadata
        metadata_changes = self._compare_metadata(v1.metadata, v2.metadata)
        
        return {
            "item_id": item_id,
            "version1": version1,
            "version2": version2,
            "content_changed": content_changed,
            "metadata_changes": metadata_changes,
            "time_difference": (v2.transaction_time_start - v1.transaction_time_start).total_seconds(),
            "changed_by": v2.changed_by,
            "change_reason": v2.change_reason
        }
    
    def _close_version(self, version: Version, close_time: datetime):
        """Close a version by setting its transaction_time_end."""
        version.transaction_time_end = close_time
        self._update_version(version)
    
    def _store_version(self, version: Version):
        """Store version in graph database."""
        try:
            # Store as a Version node in the graph
            properties = version.to_dict()
            properties['label'] = 'Version'
            
            # Use graph backend to store
            self.graph.add_node(
                item_id=f"version_{version.item_id}_{version.version_number}",
                label='Version',
                properties=properties
            )
            
            # Link to the main memory item
            self.graph.add_edge(
                source_id=version.item_id,
                target_id=f"version_{version.item_id}_{version.version_number}",
                edge_type='HAS_VERSION',
                properties={'version_number': version.version_number}
            )
            
        except Exception as e:
            logger.error(f"Error storing version: {e}")
            raise
    
    def _update_version(self, version: Version):
        """Update an existing version in the graph."""
        try:
            version_id = f"version_{version.item_id}_{version.version_number}"
            properties = version.to_dict()
            
            # Update node properties
            self.graph.update_node(version_id, properties)
            
        except Exception as e:
            logger.error(f"Error updating version: {e}")
    
    def _query_versions(
        self,
        item_id: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        include_closed: bool
    ) -> List[Version]:
        """Query versions from graph database."""
        try:
            # Query all version nodes for this item
            query = f"""
            MATCH (m {{item_id: $item_id}})-[:HAS_VERSION]->(v:Version)
            RETURN v
            ORDER BY v.version_number DESC
            """
            
            result = self.graph.execute_query(query, {'item_id': item_id})
            
            versions = []
            for row in result:
                version_data = row['v']
                version = Version.from_dict(version_data)
                
                # Apply filters
                if not include_closed and not version.is_current():
                    continue
                
                if start_time and version.transaction_time_start < start_time:
                    continue
                
                if end_time and version.transaction_time_start > end_time:
                    continue
                
                versions.append(version)
            
            return versions
            
        except Exception as e:
            logger.error(f"Error querying versions: {e}")
            return []
    
    def _compare_metadata(self, meta1: Dict, meta2: Dict) -> Dict[str, Any]:
        """Compare two metadata dictionaries."""
        added = {k: v for k, v in meta2.items() if k not in meta1}
        removed = {k: v for k, v in meta1.items() if k not in meta2}
        changed = {
            k: {'old': meta1[k], 'new': meta2[k]}
            for k in meta1.keys() & meta2.keys()
            if meta1[k] != meta2[k]
        }
        
        return {
            'added': added,
            'removed': removed,
            'changed': changed
        }
