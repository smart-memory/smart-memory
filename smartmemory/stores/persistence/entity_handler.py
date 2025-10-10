"""
Generic EntityHandler - Simple repository for any AgenticBaseModel type.

Takes type name as constructor arg, provides repo operations for that type.
Extendable by inheritance.
"""

import importlib
from typing import TypeVar, Type, List, Optional, Any

from smartmemory.models.base import MemoryBaseModel

T = TypeVar('T', bound=MemoryBaseModel)


class EntityHandler:
    """
    Generic repository for any AgenticBaseModel type.
    
    Usage:
        concept_handler = EntityHandler('Concept')
        concept = concept_handler.get('123')
        concepts = concept_handler.list()
        saved = concept_handler.save(concept)
    """

    def __init__(self, type_name: str):
        self.type_name = type_name
        self.agentic_class = self._get_agentic_class(type_name)
        self.beanie_class = self._get_beanie_class(type_name)

    def _get_agentic_class(self, type_name: str) -> Type[T]:
        """Get AgenticBaseModel class by type name."""
        # Try common IR models locations
        import_paths = [
            'smartmemory.ontology.ir_models',
            'smartmemory.extraction.models',
            'smartmemory.models',
        ]

        class_name = type_name
        if not class_name.endswith('IR'):
            class_name = type_name + 'IR'  # Try IR suffix

        for import_path in import_paths:
            try:
                module = importlib.import_module(import_path)
                if hasattr(module, class_name):
                    return getattr(module, class_name)
                # Try without IR suffix
                if hasattr(module, type_name):
                    return getattr(module, type_name)
            except ImportError:
                continue

        # For ontology entities that don't have IR models, use the Pydantic models directly
        # This handles cases like OntologyRegistry, OntologySnapshot, etc.
        beanie_class = self._get_beanie_class_safe(type_name)
        if beanie_class:
            return beanie_class

        raise ValueError(f"Could not find AgenticBaseModel class for type '{type_name}'")

    def _get_beanie_class(self, type_name: str) -> Type:
        """Get Beanie Document class by type name."""
        return self._get_beanie_class_safe(type_name) or self._raise_beanie_error(type_name)

    def _get_beanie_class_safe(self, type_name: str) -> Type:
        """Safely get Beanie Document class by type name."""
        # Try common Beanie models locations
        import_paths = [
            'memory_studio.models.ontology',
            'service_common.models.prompts',
            'memory_studio.models.enriched',
            'service_common.models.models'
        ]

        # Try different naming patterns
        class_names = [
            type_name + 'Model',  # ConceptModel
            type_name,  # OntologyRegistry (direct)
        ]

        for import_path in import_paths:
            try:
                module = importlib.import_module(import_path)
                for class_name in class_names:
                    if hasattr(module, class_name):
                        return getattr(module, class_name)
            except ImportError:
                continue

        return None

    def _raise_beanie_error(self, type_name: str):
        """Raise error for missing Beanie class."""
        raise ValueError(f"Could not find Beanie Document class for type '{type_name}'")

    def save(self, entity) -> Any:
        """Save entity and return entity."""
        # Handle both AgenticBaseModel and dict inputs
        if isinstance(entity, dict):
            # Create Beanie models from dict
            beanie_model = self.beanie_class(**entity)
        elif hasattr(entity, 'to_pydantic'):
            # AgenticBaseModel with conversion
            beanie_model = entity.to_pydantic(self.beanie_class)
        else:
            # Assume it's already a Pydantic models
            beanie_model = entity

        saved_beanie = beanie_model.save()

        # Return appropriate type based on what we have
        if self.agentic_class == self.beanie_class:
            return saved_beanie
        elif hasattr(self.agentic_class, 'from_pydantic'):
            return self.agentic_class.from_pydantic(saved_beanie)
        else:
            return saved_beanie

    def get(self, entity_id: str) -> Optional[Any]:
        """Get by ID and return entity."""
        beanie_model = self.beanie_class.get(entity_id)
        if not beanie_model:
            return None

        # Return appropriate type
        if self.agentic_class == self.beanie_class:
            return beanie_model
        elif hasattr(self.agentic_class, 'from_pydantic'):
            return self.agentic_class.from_pydantic(beanie_model)
        else:
            return beanie_model

    def list(self, **filters) -> List[Any]:
        """List entities and return List of entities."""
        beanie_models = self.beanie_class.find_many(**filters)

        # Return appropriate type
        if self.agentic_class == self.beanie_class:
            return beanie_models
        elif hasattr(self.agentic_class, 'from_pydantic'):
            return [self.agentic_class.from_pydantic(model) for model in beanie_models]
        else:
            return beanie_models

    def update(self, entity: T) -> T:
        """Update AgenticBaseModel and return AgenticBaseModel."""
        beanie_model = entity.to_pydantic(self.beanie_class)
        updated_beanie = beanie_model.save()
        return self.agentic_class.from_pydantic(updated_beanie)

    def delete(self, entity_id: str) -> bool:
        """Delete by ID and return success."""
        beanie_model = self.beanie_class.get(entity_id)
        if not beanie_model:
            return False
        beanie_model.delete()
        return True

    def find_one(self, **filters) -> Optional[T]:
        """Find one entity and return AgenticBaseModel."""
        beanie_model = self.beanie_class.find_one(**filters)
        if not beanie_model:
            return None
        if isinstance(beanie_model, MemoryBaseModel):
            return beanie_model
        return self.agentic_class.from_pydantic(beanie_model)

    def count(self, **filters) -> int:
        """Count entities matching filters."""
        return self.beanie_class.count(**filters)

    def delete_all(self, **filters) -> int:
        """Delete all entities matching filters and return count of deleted entities."""
        if filters:
            # Delete with filters
            result = self.beanie_class.find(**filters).delete()
        else:
            # Delete all entities of this type
            result = self.beanie_class.find().delete()

        # Return count of deleted documents
        return result.deleted_count if hasattr(result, 'deleted_count') else 0


# Extendable by inheritance
class ConceptHandler(EntityHandler):
    """Specialized handler for Concepts with additional methods."""

    def __init__(self):
        super().__init__('Concept')

    def find_by_label(self, label: str) -> Optional[T]:
        """Find concept by label."""
        return self.find_one(label=label)

    def find_active(self) -> List[T]:
        """Find active concepts."""
        return self.list(status="active")


class RelationHandler(EntityHandler):
    """Specialized handler for Relations with additional methods."""

    def __init__(self):
        super().__init__('Relation')

    def find_by_domain(self, domain: str) -> List[T]:
        """Find relations by domain."""
        return self.list(domain=domain)


class EvidenceHandler(EntityHandler):
    """Specialized handler for Evidence with additional methods."""

    def __init__(self):
        super().__init__('Evidence')

    def find_by_doc(self, doc: str) -> List[T]:
        """Find evidence by document."""
        return self.list(doc=doc)


# Usage examples
"""
# Generic usage
concept_handler = EntityHandler('Concept')
relation_handler = EntityHandler('Relation')
evidence_handler = EntityHandler('Evidence')

# Operations return AgenticBaseModels
concept = concept_handler.get('123')
concepts = concept_handler.list(status='active')
saved = concept_handler.save(concept)

# Specialized handlers with extra methods
concept_handler = ConceptHandler()
active_concepts = concept_handler.find_active()
concept = concept_handler.find_by_label('Doctor')

relation_handler = RelationHandler()
domain_relations = relation_handler.find_by_domain('medical')
"""
