"""
Entity validation pipeline stage.

Provides optional validation of extracted entities against:
- Predefined entity type lists
- User-defined ontologies
- Custom validation rules

This is a post-extraction stage that can be inserted into any pipeline.
"""

import logging
from typing import Dict, List, Optional, Any
from smartmemory.models.entity_types import ENTITY_TYPES

logger = logging.getLogger(__name__)


class EntityValidationStage:
    """
    Optional pipeline stage for entity type validation.
    
    Validates entity types after extraction, works with any extractor.
    
    Modes:
        - 'allow': Accept all types (default, no validation)
        - 'warn': Log warnings for unknown types but keep them
        - 'reject': Replace unknown types with default (e.g., 'concept')
        - 'strict': Raise error for unknown types
    
    Example:
        # Warn about unknown types
        validator = EntityValidationStage(mode='warn')
        result = validator.process(extraction_result)
        
        # Strict ontology validation
        validator = EntityValidationStage(
            mode='strict',
            ontology=my_ontology
        )
    """
    
    def __init__(
        self,
        mode: str = "allow",
        allowed_types: Optional[List[str]] = None,
        ontology: Optional[Any] = None,
        default_type: str = "concept"
    ):
        """
        Initialize validation stage.
        
        Args:
            mode: Validation mode ('allow', 'warn', 'reject', 'strict')
            allowed_types: List of allowed entity types (defaults to ENTITY_TYPES)
            ontology: Optional ontology object with entity_types attribute
            default_type: Type to use when rejecting unknown types
        """
        self.mode = mode
        self.allowed_types = allowed_types or ENTITY_TYPES
        self.ontology = ontology
        self.default_type = default_type
        
        # Stats
        self.validated_count = 0
        self.rejected_count = 0
        self.warned_count = 0
    
    def process(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate entity types in extraction result.
        
        Args:
            extraction_result: Dict with 'entities' and 'relations' keys
        
        Returns:
            Modified extraction result (if mode='reject') or original
        
        Raises:
            ValueError: If mode='strict' and unknown types found
        """
        if self.mode == "allow":
            # No validation, pass through
            return extraction_result
        
        entities = extraction_result.get('entities', [])
        
        for entity in entities:
            self.validated_count += 1
            
            # Get entity type from metadata
            if hasattr(entity, 'metadata') and isinstance(entity.metadata, dict):
                etype = entity.metadata.get('entity_type', 'concept')
            else:
                continue
            
            # Check against ontology or allowed types
            is_valid = self._is_valid_type(etype)
            
            if not is_valid:
                self._handle_invalid_type(entity, etype)
        
        return extraction_result
    
    def _is_valid_type(self, etype: str) -> bool:
        """Check if entity type is valid."""
        if self.ontology:
            # Validate against ontology
            if hasattr(self.ontology, 'entity_types'):
                return etype in self.ontology.entity_types
            return True  # No entity_types defined, allow all
        else:
            # Validate against allowed types list
            return etype in self.allowed_types
    
    def _handle_invalid_type(self, entity: Any, etype: str):
        """Handle invalid entity type based on mode."""
        entity_name = getattr(entity, 'content', 'unknown')
        
        if self.mode == "warn":
            logger.warning(
                f"Unknown entity_type '{etype}' for entity '{entity_name}' "
                f"(not in allowed types)"
            )
            self.warned_count += 1
        
        elif self.mode == "reject":
            logger.warning(
                f"Rejecting entity_type '{etype}' for entity '{entity_name}', "
                f"defaulting to '{self.default_type}'"
            )
            if hasattr(entity, 'metadata') and isinstance(entity.metadata, dict):
                entity.metadata['entity_type'] = self.default_type
                entity.metadata['original_entity_type'] = etype  # Preserve original
            self.rejected_count += 1
        
        elif self.mode == "strict":
            raise ValueError(
                f"Invalid entity_type '{etype}' for entity '{entity_name}'. "
                f"Allowed types: {self.allowed_types[:10]}..."
            )
    
    def get_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return {
            'validated': self.validated_count,
            'rejected': self.rejected_count,
            'warned': self.warned_count
        }
    
    def reset_stats(self):
        """Reset validation statistics."""
        self.validated_count = 0
        self.rejected_count = 0
        self.warned_count = 0


class RelationValidationStage:
    """
    Optional pipeline stage for relationship type validation.
    
    Similar to EntityValidationStage but for relationships/edges.
    """
    
    def __init__(
        self,
        mode: str = "allow",
        allowed_types: Optional[List[str]] = None,
        ontology: Optional[Any] = None,
        default_type: str = "RELATED"
    ):
        self.mode = mode
        self.allowed_types = allowed_types
        self.ontology = ontology
        self.default_type = default_type
    
    def process(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate relationship types in extraction result."""
        if self.mode == "allow":
            return extraction_result
        
        relations = extraction_result.get('relations', [])
        
        for relation in relations:
            rel_type = relation.get('relation_type') or relation.get('type')
            
            if self.ontology and hasattr(self.ontology, 'relationship_types'):
                if rel_type not in self.ontology.relationship_types:
                    self._handle_invalid_type(relation, rel_type)
            elif self.allowed_types and rel_type not in self.allowed_types:
                self._handle_invalid_type(relation, rel_type)
        
        return extraction_result
    
    def _handle_invalid_type(self, relation: Dict, rel_type: str):
        """Handle invalid relationship type."""
        if self.mode == "warn":
            logger.warning(f"Unknown relation_type '{rel_type}'")
        elif self.mode == "reject":
            logger.warning(f"Rejecting relation_type '{rel_type}', using '{self.default_type}'")
            relation['relation_type'] = self.default_type
            relation['original_relation_type'] = rel_type
        elif self.mode == "strict":
            raise ValueError(f"Invalid relation_type '{rel_type}'")
