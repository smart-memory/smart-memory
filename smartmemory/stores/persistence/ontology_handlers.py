"""
Ontology Entity Handlers

Official production handlers for ontology entities.
Extends EntityHandler with domain-specific methods for ontology operations.
"""

from datetime import datetime
from typing import List

from .entity_handler import EntityHandler


class OntologyRegistryHandler(EntityHandler):
    """Handler for OntologyRegistry with registry-specific methods."""

    def __init__(self):
        super().__init__('OntologyRegistry')

    def find_by_registry_id(self, registry_id: str):
        """Find registry by registry_id field."""
        return self.find_one(registry_id=registry_id)

    def find_by_domain(self, domain: str) -> List:
        """Find registries by domain."""
        return self.list(domain=domain)

    def get_active_registries(self) -> List:
        """Get all active registries."""
        return self.list(status="active")


class OntologySnapshotHandler(EntityHandler):
    """Handler for OntologySnapshot with versioning methods."""

    def __init__(self):
        super().__init__('OntologySnapshot')

    def find_by_registry(self, registry_id: str) -> List:
        """Find all snapshots for a registry."""
        return self.list(registry_id=registry_id)

    def find_by_version(self, registry_id: str, version: str):
        """Find specific version of a registry."""
        return self.find_one(registry_id=registry_id, version=version)

    def get_latest(self, registry_id: str):
        """Get latest snapshot for a registry."""
        snapshots = self.find_by_registry(registry_id)
        if not snapshots:
            return None

        # Sort by built_at descending
        latest = max(snapshots, key=lambda s: getattr(s, 'built_at', datetime.min))
        return latest

    def get_versions(self, registry_id: str, limit: int = 50) -> List:
        """Get version history for a registry."""
        snapshots = self.find_by_registry(registry_id)

        # Sort by built_at descending and limit
        sorted_snapshots = sorted(
            snapshots,
            key=lambda s: getattr(s, 'built_at', datetime.min),
            reverse=True
        )
        return sorted_snapshots[:limit]


class OntologyChangeLogHandler(EntityHandler):
    """Handler for OntologyChangeLog with audit methods."""

    def __init__(self):
        super().__init__('OntologyChangeLog')

    def find_by_registry(self, registry_id: str) -> List:
        """Find all changes for a registry."""
        return self.list(registry_id=registry_id)

    def find_by_operation(self, operation: str) -> List:
        """Find changes by operation type."""
        return self.list(operation=operation)

    def get_recent_changes(self, registry_id: str, limit: int = 20) -> List:
        """Get recent changes for a registry."""
        changes = self.find_by_registry(registry_id)

        # Sort by timestamp descending
        sorted_changes = sorted(
            changes,
            key=lambda c: getattr(c, 'timestamp', datetime.min),
            reverse=True
        )
        return sorted_changes[:limit]


class ConceptHandler(EntityHandler):
    """Handler for Concepts with ontology methods."""

    def __init__(self):
        super().__init__('Concept')

    def find_by_label(self, label: str):
        """Find concept by label."""
        return self.find_one(label=label)

    def find_by_confidence(self, min_confidence: float) -> List:
        """Find concepts above confidence threshold."""
        concepts = self.list()
        return [c for c in concepts if getattr(c, 'confidence', 0) >= min_confidence]

    def find_active(self) -> List:
        """Find active concepts."""
        return self.list(status="active")

    def search_by_text(self, text: str) -> List:
        """Search concepts by text content."""
        concepts = self.list()
        text_lower = text.lower()
        return [
            c for c in concepts
            if text_lower in getattr(c, 'label', '').lower()
               or text_lower in getattr(c, 'description', '').lower()
        ]


class RelationHandler(EntityHandler):
    """Handler for Relations with graph methods."""

    def __init__(self):
        super().__init__('Relation')

    def find_by_domain(self, domain: str) -> List:
        """Find relations by domain."""
        return self.list(domain=domain)

    def find_by_source(self, source_id: str) -> List:
        """Find relations from a source concept."""
        return self.list(source_id=source_id)

    def find_by_target(self, target_id: str) -> List:
        """Find relations to a target concept."""
        return self.list(target_id=target_id)

    def find_by_type(self, relation_type: str) -> List:
        """Find relations by type."""
        return self.list(relation_type=relation_type)


class OntologyHandlers:
    """Factory for all ontology handlers."""

    def __init__(self):
        self.registry = OntologyRegistryHandler()
        self.snapshot = OntologySnapshotHandler()
        self.changelog = OntologyChangeLogHandler()
        self.concept = ConceptHandler()
        self.relation = RelationHandler()

    def get_handler(self, entity_type: str) -> EntityHandler:
        """Get handler by entity type."""
        handlers = {
            'registry': self.registry,
            'snapshot': self.snapshot,
            'changelog': self.changelog,
            'concept': self.concept,
            'relation': self.relation
        }

        handler = handlers.get(entity_type.lower())
        if not handler:
            # Fallback to generic handler
            return EntityHandler(entity_type)

        return handler
