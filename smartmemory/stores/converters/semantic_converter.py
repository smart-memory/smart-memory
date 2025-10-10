"""
Semantic memory converter for MemoryItem transformations.
Handles semantic-specific conversion logic while maintaining standard interface.
"""

import logging
from typing import Dict, Any, List

from smartmemory.graph.types.interfaces import MemoryItemConverter, GraphData, GraphRelation
from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


class SemanticConverter(MemoryItemConverter):
    """
    Converter for semantic memory graphs.
    
    Handles entity extraction, relationship mapping, and semantic-specific
    metadata during conversion between MemoryItem and graph format.
    """

    def to_graph_data(self, item: MemoryItem) -> GraphData:
        """
        Convert MemoryItem to semantic graph format.
        
        Extracts entities and relationships for semantic processing.
        
        Args:
            item: MemoryItem to convert
            
        Returns:
            Dict in semantic graph format
        """
        if not self.validate_item(item):
            raise ValueError(f"Invalid MemoryItem for semantic conversion: {item}")

        # Base node structure
        node_data = {
            'item_id': item.item_id,
            'content': item.content,
            'type': getattr(item, 'type', 'semantic'),
            'user_id': getattr(item, 'user_id', None),
            'group_id': getattr(item, 'group_id', None),
            'transaction_time': getattr(item, 'transaction_time', None),
            'valid_start_time': getattr(item, 'valid_start_time', None),
            'valid_end_time': getattr(item, 'valid_end_time', None),
        }

        # Add embedding if available
        if hasattr(item, 'embedding') and item.embedding:
            node_data['embedding'] = item.embedding

        # Extract and process entities
        entities = self._extract_entities(item)
        if entities:
            node_data['entities'] = entities
            node_data['entity_count'] = len(entities)

        # Extract and process relations
        relations = self._extract_relations(item)
        if relations:
            node_data['relations'] = relations
            node_data['relation_count'] = len(relations)

        # Process metadata for semantic context
        metadata = getattr(item, 'metadata', {}) or {}
        semantic_metadata = self._process_semantic_metadata(metadata)
        node_data.update(semantic_metadata)

        # Add semantic-specific fields
        node_data['semantic_processed'] = True
        node_data['content_hash'] = MemoryItem.compute_content_hash(item.content)

        # Extract relations and convert to GraphRelation objects
        relations_data = self._extract_relations(item)
        graph_relations = []

        for rel in relations_data:
            graph_relations.append(GraphRelation(
                source_id=item.item_id,
                target_id=rel.get('object', 'unknown'),
                relation_type=rel.get('predicate', 'RELATED'),
                properties={
                    'confidence': rel.get('confidence', 0.7),
                    'subject': rel.get('subject', ''),
                    'semantic_type': 'extracted'
                }
            ))

        return GraphData(
            node_id=item.item_id,
            node_properties=node_data,
            relations=graph_relations
        )

    def from_graph_format(self, data: Dict[str, Any]) -> MemoryItem:
        """
        Convert semantic graph format to MemoryItem.
        
        Reconstructs MemoryItem from semantic graph data.
        
        Args:
            data: Semantic graph data dict
            
        Returns:
            MemoryItem instance
        """
        if not isinstance(data, dict) or 'item_id' not in data:
            raise ValueError(f"Invalid semantic graph data: {data}")

        # Extract core fields
        item_data = {
            'item_id': data['item_id'],
            'content': data.get('content', ''),
            'type': data.get('type', 'semantic'),
            'user_id': data.get('user_id'),
            'group_id': data.get('group_id'),
            'transaction_time': data.get('transaction_time'),
            'valid_start_time': data.get('valid_start_time'),
            'valid_end_time': data.get('valid_end_time'),
        }

        # Add embedding if present
        if 'embedding' in data:
            item_data['embedding'] = data['embedding']

        # Add entities and relations
        if 'entities' in data:
            item_data['entities'] = data['entities']
        if 'relations' in data:
            item_data['relations'] = data['relations']

        # Reconstruct metadata
        metadata = self._reconstruct_metadata(data)
        item_data['metadata'] = metadata

        return MemoryItem(**item_data)

    def validate_item(self, item: MemoryItem) -> bool:
        """
        Validate MemoryItem for semantic processing.
        
        Args:
            item: MemoryItem to validate
            
        Returns:
            bool: True if valid for semantic processing
        """
        if not super().validate_item(item):
            return False

        # Semantic-specific validation
        if hasattr(item, 'content') and len(item.content.strip()) < 3:
            logger.warning(f"Semantic item {item.item_id} has very short content")
            return False

        return True

    def _extract_entities(self, item: MemoryItem) -> List[str]:
        """
        Extract entities from MemoryItem content.
        
        Args:
            item: MemoryItem to process
            
        Returns:
            List of extracted entities
        """
        # Use existing entities if available
        if hasattr(item, 'entities') and item.entities:
            return item.entities

        # Extract from metadata if available
        metadata = getattr(item, 'metadata', {}) or {}
        if 'entities' in metadata:
            return metadata['entities']

        # Simple entity extraction (can be enhanced with NLP)
        entities = []
        content = item.content.lower()

        # Look for capitalized words (simple named entity detection)
        import re
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', item.content)
        entities.extend(capitalized_words[:10])  # Limit to 10 entities

        return list(set(entities))  # Remove duplicates

    def _extract_relations(self, item: MemoryItem) -> List[Dict[str, Any]]:
        """
        Extract relations from MemoryItem content.
        
        Args:
            item: MemoryItem to process
            
        Returns:
            List of extracted relations
        """
        # Use existing relations if available
        if hasattr(item, 'relations') and item.relations:
            return item.relations

        # Extract from metadata if available
        metadata = getattr(item, 'metadata', {}) or {}
        if 'relations' in metadata:
            return metadata['relations']

        # Simple relation extraction
        relations = []
        content = item.content.lower()

        # Look for common relation patterns
        relation_patterns = [
            (r'(\w+) is (\w+)', 'is_a'),
            (r'(\w+) has (\w+)', 'has'),
            (r'(\w+) uses (\w+)', 'uses'),
            (r'(\w+) contains (\w+)', 'contains'),
        ]

        import re
        for pattern, relation_type in relation_patterns:
            matches = re.findall(pattern, content)
            for match in matches[:5]:  # Limit relations
                relations.append({
                    'subject': match[0],
                    'predicate': relation_type,
                    'object': match[1],
                    'confidence': 0.7  # Simple confidence score
                })

        return relations

    def _process_semantic_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata for semantic context.
        
        Args:
            metadata: Original metadata
            
        Returns:
            Processed metadata for semantic graph
        """
        processed = {}

        # Copy semantic-relevant metadata
        semantic_keys = [
            'topic', 'category', 'tags', 'keywords', 'domain',
            'semantic_similarity', 'concept_type', 'abstraction_level'
        ]

        for key in semantic_keys:
            if key in metadata:
                processed[f"semantic_{key}"] = metadata[key]

        # Add other metadata with prefix to avoid conflicts
        for key, value in metadata.items():
            if key not in semantic_keys and not key.startswith('semantic_'):
                processed[f"meta_{key}"] = value

        return processed

    def _reconstruct_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct metadata from semantic graph data.
        
        Args:
            data: Semantic graph data
            
        Returns:
            Reconstructed metadata dict
        """
        metadata = {}

        # Extract semantic metadata
        for key, value in data.items():
            if key.startswith('semantic_'):
                original_key = key[9:]  # Remove 'semantic_' prefix
                metadata[original_key] = value
            elif key.startswith('meta_'):
                original_key = key[5:]  # Remove 'meta_' prefix
                metadata[original_key] = value

        # Add semantic processing info
        if data.get('semantic_processed'):
            metadata['semantic_processed'] = True
            metadata['entity_count'] = data.get('entity_count', 0)
            metadata['relation_count'] = data.get('relation_count', 0)

        return metadata
