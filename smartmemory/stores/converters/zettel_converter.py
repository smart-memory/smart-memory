"""
Zettel-specific converter for Zettelkasten memory graph operations.

Handles conversion between MemoryItem and Zettelkasten graph format,
including tag extraction, note linking, and concept decomposition.
"""

import logging
import re
from typing import Dict, Any, List

from smartmemory.graph.types.interfaces import MemoryItemConverter, GraphData, GraphRelation
from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


class ZettelConverter(MemoryItemConverter):
    """
    Converter for Zettelkasten memory graph operations.
    
    Handles note-specific processing including tag extraction,
    wikilink parsing, concept identification, and metadata processing.
    """

    def to_graph_data(self, item: MemoryItem) -> GraphData:
        """
        Convert MemoryItem to Zettelkasten graph format.
        
        Extracts tags, wikilinks, concepts, and creates appropriate
        graph relations for Zettelkasten-style knowledge management.
        
        Args:
            item: MemoryItem to convert
            
        Returns:
            GraphData with node properties and relations
        """
        # Build base node properties
        node_data = {
            'item_id': item.item_id,
            'content': item.content,
            'type': item.memory_type,
            'user_id': getattr(item, 'user_id', None),
            'group_id': getattr(item, 'group_id', None),
            'transaction_time': getattr(item, 'transaction_time', None),
            'valid_start_time': getattr(item, 'valid_start_time', None),
            'valid_end_time': getattr(item, 'valid_end_time', None)
        }

        # Extract and process Zettelkasten-specific data
        tags = self._extract_tags(item)
        wikilinks = self._extract_wikilinks(item)
        concepts = self._extract_concepts(item)

        # Add extracted data to node properties
        if tags:
            node_data['zettel_tags'] = tags
            node_data['tag_count'] = len(tags)

        if wikilinks:
            node_data['zettel_links'] = wikilinks
            node_data['link_count'] = len(wikilinks)

        if concepts:
            node_data['zettel_concepts'] = concepts
            node_data['concept_count'] = len(concepts)

        # Process metadata for Zettelkasten context
        metadata = getattr(item, 'metadata', {}) or {}
        zettel_metadata = self._process_zettel_metadata(metadata)
        node_data.update(zettel_metadata)

        # Add Zettelkasten-specific fields
        node_data['zettel_processed'] = True
        node_data['note_indexed'] = True
        node_data['title'] = metadata.get('title', item.item_id)
        node_data['name'] = node_data['title']  # Alias for compatibility
        node_data['zettel_body'] = item.content  # Alias for compatibility
        node_data['label'] = 'Note'  # Standard Zettelkasten label

        # Extract relations and convert to GraphRelation objects
        graph_relations = []

        # Add tag relations (standardized as TAGGED_WITH)
        for tag in tags[:10]:  # Limit tag relations
            graph_relations.append(GraphRelation(
                source_id=item.item_id,
                target_id=f"tag_{tag}",
                relation_type="TAGGED_WITH",
                properties={'tag_name': tag, 'zettel_type': 'tag'}
            ))

        # Add wikilink relations (bidirectional)
        for link in wikilinks[:15]:  # Limit link relations
            # Forward link
            graph_relations.append(GraphRelation(
                source_id=item.item_id,
                target_id=link,
                relation_type="LINKS_TO",
                properties={'link_type': 'wikilink', 'zettel_type': 'link', 'direction': 'forward'}
            ))

            # Automatic backlink for true Zettelkasten bidirectional linking
            graph_relations.append(GraphRelation(
                source_id=link,
                target_id=item.item_id,
                relation_type="LINKS_TO_BACK",
                properties={'link_type': 'backlink', 'zettel_type': 'link', 'direction': 'backward', 'auto_created': True}
            ))

        # Add concept relations (MENTIONS)
        for concept in concepts[:10]:  # Limit concept relations
            graph_relations.append(GraphRelation(
                source_id=item.item_id,
                target_id=f"concept_{concept}",
                relation_type="MENTIONS",
                properties={'concept_name': concept, 'zettel_type': 'concept'}
            ))

        return GraphData(
            node_id=item.item_id,
            node_properties=node_data,
            relations=graph_relations
        )

    def from_graph_format(self, data: Dict[str, Any]) -> MemoryItem:
        """
        Convert Zettelkasten graph format to MemoryItem.
        
        Reconstructs MemoryItem from Zettelkasten graph data.
        
        Args:
            data: Zettelkasten graph data dict
            
        Returns:
            MemoryItem instance
        """
        if not isinstance(data, dict) or 'item_id' not in data:
            raise ValueError(f"Invalid Zettelkasten graph data: {data}")

        # Extract core fields
        item_data = {
            'item_id': data['item_id'],
            'content': data.get('content', data.get('zettel_body', '')),
            'type': data.get('type', 'zettel'),
            'user_id': data.get('user_id'),
            'group_id': data.get('group_id'),
            'transaction_time': data.get('transaction_time'),
            'valid_start_time': data.get('valid_start_time'),
            'valid_end_time': data.get('valid_end_time')
        }

        # Reconstruct metadata
        metadata = self._reconstruct_metadata(data)
        item_data['metadata'] = metadata

        return MemoryItem(**item_data)

    def validate_item(self, item: MemoryItem) -> bool:
        """
        Validate MemoryItem for Zettelkasten processing.
        
        Args:
            item: MemoryItem to validate
            
        Returns:
            bool: True if valid for Zettelkasten processing
        """
        if not item or not hasattr(item, 'content'):
            logger.warning("ZettelConverter: Item missing content")
            return False

        if not hasattr(item, 'item_id') or not item.item_id:
            logger.warning("ZettelConverter: Item missing item_id")
            return False

        return True

    def _extract_tags(self, item: MemoryItem) -> List[str]:
        """
        Extract tags from MemoryItem content and metadata.
        
        Args:
            item: MemoryItem to extract tags from
            
        Returns:
            List of extracted tags
        """
        tags = []

        # Check if tags already exist in metadata
        metadata = getattr(item, 'metadata', {}) or {}
        if 'tags' in metadata and isinstance(metadata['tags'], list):
            tags.extend(metadata['tags'])

        # Extract hashtags from content
        content = item.content or ""
        hashtag_pattern = r'#(\w+)'
        hashtags = re.findall(hashtag_pattern, content)
        tags.extend(hashtags)

        # Extract YAML-style tags if present
        yaml_tags_pattern = r'tags:\s*\[(.*?)\]'
        yaml_match = re.search(yaml_tags_pattern, content)
        if yaml_match:
            yaml_tags = [tag.strip().strip('"\'') for tag in yaml_match.group(1).split(',')]
            tags.extend(yaml_tags)

        return list(set(tags))  # Remove duplicates

    def _extract_wikilinks(self, item: MemoryItem) -> List[str]:
        """
        Extract wikilinks from MemoryItem content.
        
        Args:
            item: MemoryItem to extract wikilinks from
            
        Returns:
            List of extracted wikilinks
        """
        content = item.content or ""

        # Extract [[wikilinks]]
        wikilink_pattern = r'\[\[([^\]]+)\]\]'
        wikilinks = re.findall(wikilink_pattern, content)

        # Clean up wikilinks (remove aliases, normalize)
        cleaned_links = []
        for link in wikilinks:
            # Handle aliases: [[Link|Alias]] -> Link
            if '|' in link:
                link = link.split('|')[0]
            cleaned_links.append(link.strip())

        return list(set(cleaned_links))  # Remove duplicates

    def _extract_concepts(self, item: MemoryItem) -> List[str]:
        """
        Extract key concepts from MemoryItem content.
        
        Args:
            item: MemoryItem to extract concepts from
            
        Returns:
            List of extracted concepts
        """
        content = item.content or ""
        concepts = []

        # Extract concepts in double parentheses ((concept))
        concept_pattern = r'\(\(([^)]+)\)\)'
        double_paren_concepts = re.findall(concept_pattern, content)
        concepts.extend(double_paren_concepts)

        # Extract capitalized terms (potential concepts)
        # Simple heuristic: words starting with capital letters that aren't at sentence start
        capitalized_pattern = r'(?<!^)(?<!\. )\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        capitalized_terms = re.findall(capitalized_pattern, content)

        # Filter out common words and keep only meaningful concepts
        meaningful_concepts = []
        common_words = {'The', 'This', 'That', 'These', 'Those', 'When', 'Where', 'Why', 'How', 'What', 'Who'}
        for term in capitalized_terms:
            if term not in common_words and len(term) > 2:
                meaningful_concepts.append(term)

        concepts.extend(meaningful_concepts[:5])  # Limit to top 5 concepts

        return list(set(concepts))  # Remove duplicates

    def _process_zettel_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata for Zettelkasten context.
        
        Args:
            metadata: Original metadata
            
        Returns:
            Processed metadata for Zettelkasten graph
        """
        processed = {}

        # Copy Zettelkasten-relevant metadata
        zettel_keys = [
            'title', 'created_at', 'modified_at', 'author', 'source',
            'note_type', 'importance', 'difficulty', 'status',
            'parent_note', 'child_notes', 'related_notes'
        ]

        for key in zettel_keys:
            if key in metadata:
                processed[f'zettel_{key}'] = metadata[key]

        # Add Zettelkasten-specific processing
        if 'created_at' in metadata:
            processed['zettel_creation_date'] = metadata['created_at']

        if 'importance' in metadata:
            try:
                importance = int(metadata['importance'])
                processed['zettel_importance_level'] = importance
                processed['zettel_high_importance'] = importance >= 8
            except (ValueError, TypeError):
                processed['zettel_importance_level'] = 5  # Default

        return processed

    def _reconstruct_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct metadata from Zettelkasten graph data.
        
        Args:
            data: Graph data dict
            
        Returns:
            Reconstructed metadata
        """
        metadata = {}

        # Extract Zettelkasten-specific fields back to metadata
        for key, value in data.items():
            if key.startswith('zettel_') and key not in ['zettel_processed', 'zettel_body']:
                original_key = key.replace('zettel_', '')
                metadata[original_key] = value

        # Reconstruct arrays
        if 'zettel_tags' in data:
            metadata['tags'] = data['zettel_tags']

        if 'zettel_links' in data:
            metadata['wikilinks'] = data['zettel_links']

        if 'zettel_concepts' in data:
            metadata['concepts'] = data['zettel_concepts']

        # Add title if present
        if 'title' in data:
            metadata['title'] = data['title']

        return metadata
