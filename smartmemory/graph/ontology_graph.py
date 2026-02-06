"""OntologyGraph — manages entity type definitions in a separate FalkorDB graph.

Stores entity types with three-tier status: seed → provisional → confirmed.
Separate from the data graph so ontology reads never contend with data writes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Seed entity types shipped with SmartMemory
SEED_TYPES: List[str] = [
    "Person",
    "Organization",
    "Technology",
    "Concept",
    "Event",
    "Location",
    "Document",
    "Tool",
    "Skill",
    "Decision",
    "Claim",
    "Action",
    "Metric",
    "Process",
]

VALID_STATUSES = {"seed", "provisional", "confirmed"}


class OntologyGraph:
    """Manage entity type definitions in a dedicated FalkorDB graph.

    The ontology graph is named ``ws_{workspace_id}_ontology`` and stores
    ``EntityType`` nodes with ``name`` and ``status`` properties.

    Usage::

        og = OntologyGraph(workspace_id="acme")
        og.seed_types()   # idempotent — seeds 14 base types
        og.add_provisional("Metric")
        og.promote("Metric")
        types = og.get_entity_types()
    """

    def __init__(self, workspace_id: str = "default", backend=None):
        """
        Args:
            workspace_id: Workspace identifier for graph naming.
            backend: Optional FalkorDB backend instance. If None, resolved from config.
        """
        self.workspace_id = workspace_id
        self._graph_name = f"ws_{workspace_id}_ontology"
        self._backend = backend

    def _get_backend(self):
        """Lazy-load or return injected backend."""
        if self._backend is not None:
            return self._backend
        try:
            from smartmemory.graph.backends.falkordb import FalkorDBBackend

            self._backend = FalkorDBBackend(graph_name=self._graph_name)
            return self._backend
        except Exception as e:
            logger.error("Failed to initialize ontology graph backend: %s", e)
            raise

    def seed_types(self, types: Optional[List[str]] = None) -> int:
        """Seed initial entity types with status='seed'. Idempotent.

        Returns:
            Number of types actually created (skips existing).
        """
        types_to_seed = types or SEED_TYPES
        backend = self._get_backend()
        created = 0

        for type_name in types_to_seed:
            try:
                existing = self._query_type(backend, type_name)
                if existing:
                    continue
                self._create_type(backend, type_name, "seed")
                created += 1
            except Exception as e:
                logger.warning("Failed to seed type '%s': %s", type_name, e)

        logger.info("Seeded %d/%d entity types in %s", created, len(types_to_seed), self._graph_name)
        return created

    def get_entity_types(self) -> List[Dict[str, Any]]:
        """Return all entity types with their status."""
        backend = self._get_backend()
        try:
            result = backend.query(
                "MATCH (t:EntityType) RETURN t.name AS name, t.status AS status ORDER BY t.name",
                graph_name=self._graph_name,
            )
            return [{"name": row[0], "status": row[1]} for row in (result or [])]
        except Exception as e:
            logger.warning("Failed to query entity types: %s", e)
            return []

    def get_type_status(self, name: str) -> Optional[str]:
        """Return the status of a type, or None if not found."""
        backend = self._get_backend()
        existing = self._query_type(backend, name)
        if existing:
            return existing[0][1]  # status column
        return None

    def add_provisional(self, name: str) -> bool:
        """Add a new type with status='provisional'. No-op if already exists.

        Returns:
            True if created, False if already existed.
        """
        backend = self._get_backend()
        existing = self._query_type(backend, name)
        if existing:
            return False
        self._create_type(backend, name, "provisional")
        return True

    def promote(self, name: str) -> bool:
        """Promote a type to 'confirmed'. Returns False if type not found."""
        backend = self._get_backend()
        try:
            backend.query(
                "MATCH (t:EntityType {name: $name}) SET t.status = 'confirmed'",
                params={"name": name},
                graph_name=self._graph_name,
            )
            return True
        except Exception as e:
            logger.warning("Failed to promote type '%s': %s", name, e)
            return False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _query_type(self, backend, name: str):
        """Query a single type by name."""
        try:
            return backend.query(
                "MATCH (t:EntityType {name: $name}) RETURN t.name, t.status",
                params={"name": name},
                graph_name=self._graph_name,
            )
        except Exception:
            return None

    def _create_type(self, backend, name: str, status: str):
        """Create an EntityType node."""
        backend.query(
            "CREATE (t:EntityType {name: $name, status: $status})",
            params={"name": name, "status": status},
            graph_name=self._graph_name,
        )
