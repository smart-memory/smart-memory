"""
Episodic memory converter for MemoryItem transformations.
Handles episodic-specific conversion logic including temporal boundaries and context.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from smartmemory.graph.types.interfaces import MemoryItemConverter, GraphData, GraphRelation
from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


class EpisodicConverter(MemoryItemConverter):
    """
    Converter for episodic memory graphs.
    
    Handles temporal boundaries, episode context, and time-based
    metadata during conversion between MemoryItem and graph format.
    """

    def to_graph_data(self, item: MemoryItem) -> GraphData:
        """
        Convert MemoryItem to episodic graph format.
        
        Processes temporal information and episode boundaries.
        
        Args:
            item: MemoryItem to convert
            
        Returns:
            Dict in episodic graph format
        """
        if not self.validate_item(item):
            raise ValueError(f"Invalid MemoryItem for episodic conversion: {item}")

        # Base node structure
        node_data = {
            'item_id': item.item_id,
            'content': item.content,
            'type': getattr(item, 'type', 'episodic'),
            'user_id': getattr(item, 'user_id', None),
            'group_id': getattr(item, 'group_id', None),
            'transaction_time': getattr(item, 'transaction_time', None),
            'valid_start_time': getattr(item, 'valid_start_time', None),
            'valid_end_time': getattr(item, 'valid_end_time', None),
        }

        # Process temporal information
        temporal_data = self._process_temporal_data(item)
        node_data.update(temporal_data)

        # Process episode boundaries
        episode_data = self._process_episode_boundaries(item)
        node_data.update(episode_data)

        # Process metadata for episodic context
        metadata = getattr(item, 'metadata', {}) or {}
        episodic_metadata = self._process_episodic_metadata(metadata)
        node_data.update(episodic_metadata)

        # Add episodic-specific fields
        node_data['episodic_processed'] = True
        node_data['temporal_indexed'] = True

        # Extract temporal and participant relations
        graph_relations = []

        # Add temporal relations if timestamp exists
        if 'timestamp' in node_data:
            graph_relations.append(GraphRelation(
                source_id=item.item_id,
                target_id=f"time_{node_data['timestamp']}",
                relation_type="OCCURRED_AT",
                properties={'temporal_type': 'timestamp'}
            ))

        # Add participant relations
        participants = metadata.get('participants', [])
        for participant in participants[:5]:  # Limit relations
            graph_relations.append(GraphRelation(
                source_id=item.item_id,
                target_id=f"person_{participant}",
                relation_type="INVOLVES",
                properties={'participant_type': 'person'}
            ))

        # Add location relation if present
        if 'location' in metadata:
            graph_relations.append(GraphRelation(
                source_id=item.item_id,
                target_id=f"location_{metadata['location']}",
                relation_type="OCCURRED_AT",
                properties={'spatial_type': 'location'}
            ))

        return GraphData(
            node_id=item.item_id,
            node_properties=node_data,
            relations=graph_relations
        )

    def from_graph_format(self, data: Dict[str, Any]) -> MemoryItem:
        """
        Convert episodic graph format to MemoryItem.
        
        Reconstructs MemoryItem from episodic graph data.
        
        Args:
            data: Episodic graph data dict
            
        Returns:
            MemoryItem instance
        """
        if not isinstance(data, dict) or 'item_id' not in data:
            raise ValueError(f"Invalid episodic graph data: {data}")

        # Extract core fields
        item_data = {
            'item_id': data['item_id'],
            'content': data.get('content', ''),
            'type': data.get('type', 'episodic'),
            'user_id': data.get('user_id'),
            'group_id': data.get('group_id'),
            'transaction_time': self._parse_datetime(data.get('transaction_time')),
            'valid_start_time': self._parse_datetime(data.get('valid_start_time')),
            'valid_end_time': self._parse_datetime(data.get('valid_end_time')),
        }

        # Reconstruct metadata
        metadata = self._reconstruct_metadata(data)
        item_data['metadata'] = metadata

        return MemoryItem(**item_data)

    def validate_item(self, item: MemoryItem) -> bool:
        """
        Validate MemoryItem for episodic processing.
        
        Args:
            item: MemoryItem to validate
            
        Returns:
            bool: True if valid for episodic processing
        """
        if not super().validate_item(item):
            return False

        # Episodic-specific validation
        # Should have some temporal context
        has_temporal = (
                hasattr(item, 'transaction_time') and item.transaction_time or
                hasattr(item, 'valid_start_time') and item.valid_start_time or
                (hasattr(item, 'metadata') and item.metadata and
                 any(key in item.metadata for key in ['timestamp', 'created_at', 'date', 'time']))
        )

        if not has_temporal:
            logger.warning(f"Episodic item {item.item_id} lacks temporal information")
            # Don't fail validation, just warn - we can add current time

        return True

    def _process_temporal_data(self, item: MemoryItem) -> Dict[str, Any]:
        """
        Process temporal information from MemoryItem.
        
        Args:
            item: MemoryItem to process
            
        Returns:
            Dict with temporal data
        """
        temporal_data = {}

        # Extract timestamps
        transaction_time = getattr(item, 'transaction_time', None)
        if transaction_time:
            if isinstance(transaction_time, datetime):
                temporal_data['timestamp'] = transaction_time.isoformat()
                temporal_data['timestamp_unix'] = transaction_time.timestamp()
            else:
                temporal_data['timestamp'] = str(transaction_time)
        else:
            # Use current time if no timestamp
            now = datetime.now(timezone.utc)
            temporal_data['timestamp'] = now.isoformat()
            temporal_data['timestamp_unix'] = now.timestamp()

        # Extract valid time range
        if hasattr(item, 'valid_start_time') and item.valid_start_time:
            temporal_data['valid_start'] = item.valid_start_time.isoformat()
        if hasattr(item, 'valid_end_time') and item.valid_end_time:
            temporal_data['valid_end'] = item.valid_end_time.isoformat()

        # Extract temporal patterns from metadata
        metadata = getattr(item, 'metadata', {}) or {}
        for key in ['created_at', 'updated_at', 'date', 'time', 'timestamp']:
            if key in metadata:
                temporal_data[f"meta_{key}"] = metadata[key]

        # Add temporal indexing information
        if 'timestamp' in temporal_data:
            try:
                dt = datetime.fromisoformat(temporal_data['timestamp'].replace('Z', '+00:00'))
                temporal_data['year'] = dt.year
                temporal_data['month'] = dt.month
                temporal_data['day'] = dt.day
                temporal_data['hour'] = dt.hour
                temporal_data['weekday'] = dt.weekday()
            except Exception as e:
                logger.warning(f"Failed to parse timestamp for temporal indexing: {e}")

        return temporal_data

    def _process_episode_boundaries(self, item: MemoryItem) -> Dict[str, Any]:
        """
        Process episode boundary information.
        
        Args:
            item: MemoryItem to process
            
        Returns:
            Dict with episode boundary data
        """
        episode_data = {}

        metadata = getattr(item, 'metadata', {}) or {}

        # Check for explicit episode markers
        if 'episode_id' in metadata:
            episode_data['episode_id'] = metadata['episode_id']

        if 'episode_boundary' in metadata:
            episode_data['episode_boundary'] = metadata['episode_boundary']

        # Detect episode boundaries from content
        content = item.content.lower()
        boundary_indicators = [
            'session started', 'session ended', 'new conversation',
            'meeting began', 'meeting ended', 'task completed',
            'day started', 'day ended', 'project started'
        ]

        for indicator in boundary_indicators:
            if indicator in content:
                episode_data['boundary_detected'] = True
                episode_data['boundary_type'] = indicator
                break

        # Episode context
        if 'context' in metadata:
            episode_data['episode_context'] = metadata['context']

        if 'session_id' in metadata:
            episode_data['session_id'] = metadata['session_id']

        return episode_data

    def _process_episodic_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata for episodic context.
        
        Args:
            metadata: Original metadata
            
        Returns:
            Processed metadata for episodic graph
        """
        processed = {}

        # Copy episodic-relevant metadata
        episodic_keys = [
            'session_id', 'episode_id', 'context', 'location', 'participants',
            'activity', 'event_type', 'duration', 'sequence_number'
        ]

        for key in episodic_keys:
            if key in metadata:
                processed[f"episodic_{key}"] = metadata[key]

        # Add other metadata with prefix to avoid conflicts
        for key, value in metadata.items():
            if key not in episodic_keys and not key.startswith('episodic_'):
                processed[f"meta_{key}"] = value

        return processed

    def _reconstruct_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct metadata from episodic graph data.
        
        Args:
            data: Episodic graph data
            
        Returns:
            Reconstructed metadata dict
        """
        metadata = {}

        # Extract episodic metadata
        for key, value in data.items():
            if key.startswith('episodic_'):
                original_key = key[9:]  # Remove 'episodic_' prefix
                metadata[original_key] = value
            elif key.startswith('meta_'):
                original_key = key[5:]  # Remove 'meta_' prefix
                metadata[original_key] = value

        # Add temporal information to metadata
        temporal_keys = ['timestamp', 'year', 'month', 'day', 'hour', 'weekday']
        for key in temporal_keys:
            if key in data:
                metadata[key] = data[key]

        # Add episode information
        episode_keys = ['episode_id', 'episode_boundary', 'boundary_detected', 'session_id']
        for key in episode_keys:
            if key in data:
                metadata[key] = data[key]

        # Add processing info
        if data.get('episodic_processed'):
            metadata['episodic_processed'] = True
            metadata['temporal_indexed'] = data.get('temporal_indexed', False)

        return metadata

    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """
        Parse datetime string to datetime object.
        
        Args:
            dt_str: Datetime string
            
        Returns:
            Parsed datetime or None
        """
        if not dt_str:
            return None

        try:
            if isinstance(dt_str, datetime):
                return dt_str
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        except Exception as e:
            logger.warning(f"Failed to parse datetime '{dt_str}': {e}")
            return None
