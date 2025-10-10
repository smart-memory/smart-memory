"""
Procedural memory converter for MemoryItem transformations.
Handles procedural-specific conversion logic including skills, tools, and execution patterns.
"""

import logging
import re
from typing import Dict, Any, List

from smartmemory.graph.types.interfaces import MemoryItemConverter, GraphData, GraphRelation
from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


class ProceduralConverter(MemoryItemConverter):
    """
    Converter for procedural memory graphs.
    
    Handles skill extraction, tool identification, execution patterns,
    and procedural-specific metadata during conversion.
    """

    def to_graph_data(self, item: MemoryItem) -> GraphData:
        """
        Convert MemoryItem to procedural graph format.
        
        Extracts skills, tools, and execution patterns.
        
        Args:
            item: MemoryItem to convert
            
        Returns:
            Dict in procedural graph format
        """
        if not self.validate_item(item):
            raise ValueError(f"Invalid MemoryItem for procedural conversion: {item}")

        # Base node structure
        node_data = {
            'item_id': item.item_id,
            'content': item.content,
            'type': getattr(item, 'type', 'procedural'),
            'user_id': getattr(item, 'user_id', None),
            'group_id': getattr(item, 'group_id', None),
            'transaction_time': getattr(item, 'transaction_time', None),
            'valid_start_time': getattr(item, 'valid_start_time', None),
            'valid_end_time': getattr(item, 'valid_end_time', None),
        }

        # Extract and process skills
        skills = self._extract_skills(item)
        if skills:
            node_data['skills'] = skills
            node_data['skill_count'] = len(skills)

        # Extract and process tools
        tools = self._extract_tools(item)
        if tools:
            node_data['tools'] = tools
            node_data['tool_count'] = len(tools)

        # Extract execution patterns
        patterns = self._extract_execution_patterns(item)
        if patterns:
            node_data['execution_patterns'] = patterns
            node_data['pattern_count'] = len(patterns)

        # Process metadata for procedural context
        metadata = getattr(item, 'metadata', {}) or {}
        procedural_metadata = self._process_procedural_metadata(metadata)
        node_data.update(procedural_metadata)

        # Add procedural-specific fields
        node_data['procedural_processed'] = True
        node_data['skill_indexed'] = True

        # Extract skill and tool relations
        graph_relations = []

        # Add skill relations
        skills = self._extract_skills(item)
        for skill in skills[:5]:  # Limit relations
            graph_relations.append(GraphRelation(
                source_id=item.item_id,
                target_id=f"skill_{skill}",
                relation_type="REQUIRES_SKILL",
                properties={'skill_type': 'extracted'}
            ))

        # Add tool relations
        tools = self._extract_tools(item)
        for tool in tools[:5]:  # Limit relations
            graph_relations.append(GraphRelation(
                source_id=item.item_id,
                target_id=f"tool_{tool}",
                relation_type="USES_TOOL",
                properties={'tool_type': 'extracted'}
            ))

        # Add step sequence relations if steps exist
        steps = metadata.get('steps', [])
        for i, step in enumerate(steps[:10]):  # Limit step relations
            graph_relations.append(GraphRelation(
                source_id=item.item_id,
                target_id=f"step_{step}",
                relation_type="HAS_STEP",
                properties={'step_order': i, 'step_type': 'sequence'}
            ))

        return GraphData(
            node_id=item.item_id,
            node_properties=node_data,
            relations=graph_relations
        )

    def from_graph_format(self, data: Dict[str, Any]) -> MemoryItem:
        """
        Convert procedural graph format to MemoryItem.
        
        Reconstructs MemoryItem from procedural graph data.
        
        Args:
            data: Procedural graph data dict
            
        Returns:
            MemoryItem instance
        """
        if not isinstance(data, dict) or 'item_id' not in data:
            raise ValueError(f"Invalid procedural graph data: {data}")

        # Extract core fields
        item_data = {
            'item_id': data['item_id'],
            'content': data.get('content', ''),
            'type': data.get('type', 'procedural'),
            'user_id': data.get('user_id'),
            'group_id': data.get('group_id'),
            'transaction_time': data.get('transaction_time'),
            'valid_start_time': data.get('valid_start_time'),
            'valid_end_time': data.get('valid_end_time'),
        }

        # Reconstruct metadata
        metadata = self._reconstruct_metadata(data)
        item_data['metadata'] = metadata

        return MemoryItem(**item_data)

    def validate_item(self, item: MemoryItem) -> bool:
        """
        Validate MemoryItem for procedural processing.
        
        Args:
            item: MemoryItem to validate
            
        Returns:
            bool: True if valid for procedural processing
        """
        if not super().validate_item(item):
            return False

        # Procedural-specific validation
        content = item.content.lower()

        # Should contain some procedural indicators
        procedural_indicators = [
            'how to', 'step', 'process', 'method', 'technique', 'skill',
            'tool', 'use', 'implement', 'execute', 'perform', 'do',
            'create', 'build', 'make', 'configure', 'setup'
        ]

        has_procedural_content = any(indicator in content for indicator in procedural_indicators)
        if not has_procedural_content:
            logger.info(f"Procedural item {item.item_id} may not contain procedural content")
            # Don't fail validation, just note it

        return True

    def _extract_skills(self, item: MemoryItem) -> List[Dict[str, Any]]:
        """
        Extract skills from MemoryItem content.
        
        Args:
            item: MemoryItem to process
            
        Returns:
            List of extracted skills
        """
        skills = []

        # Use existing skills if available
        metadata = getattr(item, 'metadata', {}) or {}
        if 'skills' in metadata:
            existing_skills = metadata['skills']
            if isinstance(existing_skills, list):
                for skill in existing_skills:
                    if isinstance(skill, str):
                        skills.append({'name': skill, 'type': 'explicit'})
                    elif isinstance(skill, dict):
                        skills.append(skill)

        # Extract skills from content
        content = item.content.lower()

        # Skill patterns
        skill_patterns = [
            (r'learn(?:ed|ing)?\s+(?:how\s+to\s+)?(\w+(?:\s+\w+){0,2})', 'learned'),
            (r'master(?:ed|ing)?\s+(\w+(?:\s+\w+){0,2})', 'mastered'),
            (r'skill\s+in\s+(\w+(?:\s+\w+){0,2})', 'skill'),
            (r'expert\s+(?:in|at)\s+(\w+(?:\s+\w+){0,2})', 'expert'),
            (r'proficient\s+(?:in|at)\s+(\w+(?:\s+\w+){0,2})', 'proficient'),
            (r'experience\s+with\s+(\w+(?:\s+\w+){0,2})', 'experience'),
        ]

        for pattern, skill_type in skill_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                skill_name = match.strip()
                if len(skill_name) > 2:  # Filter out very short matches
                    skills.append({
                        'name': skill_name,
                        'type': skill_type,
                        'confidence': 0.7,
                        'extracted_from': 'content'
                    })

        # Remove duplicates based on name
        seen_names = set()
        unique_skills = []
        for skill in skills:
            name = skill.get('name', '').lower()
            if name not in seen_names:
                seen_names.add(name)
                unique_skills.append(skill)

        return unique_skills[:10]  # Limit to 10 skills

    def _extract_tools(self, item: MemoryItem) -> List[Dict[str, Any]]:
        """
        Extract tools from MemoryItem content.
        
        Args:
            item: MemoryItem to process
            
        Returns:
            List of extracted tools
        """
        tools = []

        # Use existing tools if available
        metadata = getattr(item, 'metadata', {}) or {}
        if 'tools' in metadata:
            existing_tools = metadata['tools']
            if isinstance(existing_tools, list):
                for tool in existing_tools:
                    if isinstance(tool, str):
                        tools.append({'name': tool, 'type': 'explicit'})
                    elif isinstance(tool, dict):
                        tools.append(tool)

        # Extract tools from content
        content = item.content.lower()

        # Tool patterns
        tool_patterns = [
            (r'us(?:e|ed|ing)\s+(\w+(?:\s+\w+){0,2})', 'used'),
            (r'with\s+(\w+(?:\s+\w+){0,2})', 'with'),
            (r'tool\s+(\w+(?:\s+\w+){0,2})', 'tool'),
            (r'software\s+(\w+(?:\s+\w+){0,2})', 'software'),
            (r'application\s+(\w+(?:\s+\w+){0,2})', 'application'),
            (r'framework\s+(\w+(?:\s+\w+){0,2})', 'framework'),
            (r'library\s+(\w+(?:\s+\w+){0,2})', 'library'),
        ]

        # Common tools to look for specifically
        common_tools = [
            'python', 'javascript', 'java', 'git', 'docker', 'kubernetes',
            'aws', 'azure', 'gcp', 'linux', 'windows', 'macos',
            'vscode', 'intellij', 'eclipse', 'vim', 'emacs',
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch'
        ]

        for tool in common_tools:
            if tool in content:
                tools.append({
                    'name': tool,
                    'type': 'technology',
                    'confidence': 0.8,
                    'extracted_from': 'content'
                })

        for pattern, tool_type in tool_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                tool_name = match.strip()
                if len(tool_name) > 2 and tool_name not in common_tools:
                    tools.append({
                        'name': tool_name,
                        'type': tool_type,
                        'confidence': 0.6,
                        'extracted_from': 'content'
                    })

        # Remove duplicates based on name
        seen_names = set()
        unique_tools = []
        for tool in tools:
            name = tool.get('name', '').lower()
            if name not in seen_names:
                seen_names.add(name)
                unique_tools.append(tool)

        return unique_tools[:15]  # Limit to 15 tools

    def _extract_execution_patterns(self, item: MemoryItem) -> List[Dict[str, Any]]:
        """
        Extract execution patterns from MemoryItem content.
        
        Args:
            item: MemoryItem to process
            
        Returns:
            List of extracted execution patterns
        """
        patterns = []
        content = item.content.lower()

        # Sequential patterns
        if any(word in content for word in ['step', 'first', 'then', 'next', 'finally']):
            patterns.append({
                'type': 'sequential',
                'confidence': 0.8,
                'description': 'Contains sequential execution steps'
            })

        # Conditional patterns
        if any(word in content for word in ['if', 'when', 'unless', 'depending']):
            patterns.append({
                'type': 'conditional',
                'confidence': 0.7,
                'description': 'Contains conditional logic'
            })

        # Loop patterns
        if any(word in content for word in ['repeat', 'loop', 'iterate', 'until', 'while']):
            patterns.append({
                'type': 'iterative',
                'confidence': 0.7,
                'description': 'Contains iterative execution'
            })

        # Error handling patterns
        if any(word in content for word in ['error', 'exception', 'fail', 'catch', 'handle']):
            patterns.append({
                'type': 'error_handling',
                'confidence': 0.6,
                'description': 'Contains error handling logic'
            })

        # Parallel patterns
        if any(word in content for word in ['parallel', 'concurrent', 'simultaneously', 'async']):
            patterns.append({
                'type': 'parallel',
                'confidence': 0.7,
                'description': 'Contains parallel execution'
            })

        return patterns

    def _process_procedural_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata for procedural context.
        
        Args:
            metadata: Original metadata
            
        Returns:
            Processed metadata for procedural graph
        """
        processed = {}

        # Copy procedural-relevant metadata
        procedural_keys = [
            'skill_type', 'tool_type', 'difficulty', 'complexity',
            'execution_time', 'prerequisites', 'outcomes', 'effectiveness',
            'success_rate', 'failure_modes', 'alternatives'
        ]

        for key in procedural_keys:
            if key in metadata:
                processed[f"procedural_{key}"] = metadata[key]

        # Add other metadata with prefix to avoid conflicts
        for key, value in metadata.items():
            if key not in procedural_keys and not key.startswith('procedural_'):
                processed[f"meta_{key}"] = value

        return processed

    def _reconstruct_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct metadata from procedural graph data.
        
        Args:
            data: Procedural graph data
            
        Returns:
            Reconstructed metadata dict
        """
        metadata = {}

        # Extract procedural metadata
        for key, value in data.items():
            if key.startswith('procedural_'):
                original_key = key[11:]  # Remove 'procedural_' prefix
                metadata[original_key] = value
            elif key.startswith('meta_'):
                original_key = key[5:]  # Remove 'meta_' prefix
                metadata[original_key] = value

        # Add extracted information to metadata
        if 'skills' in data:
            metadata['skills'] = data['skills']
            metadata['skill_count'] = data.get('skill_count', 0)

        if 'tools' in data:
            metadata['tools'] = data['tools']
            metadata['tool_count'] = data.get('tool_count', 0)

        if 'execution_patterns' in data:
            metadata['execution_patterns'] = data['execution_patterns']
            metadata['pattern_count'] = data.get('pattern_count', 0)

        # Add processing info
        if data.get('procedural_processed'):
            metadata['procedural_processed'] = True
            metadata['skill_indexed'] = data.get('skill_indexed', False)

        return metadata
