"""FalkorDB index utilities for batch ontology update queries."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def ensure_extraction_indexes(graph_backend) -> None:
    """Create FalkorDB index on extraction_status for batch ontology update queries.

    Follows the pattern in ontology/registry.py — try-except to handle "already exists" gracefully.
    Callable from both tests and the Phase 2 service worker without modifying SmartGraph init.

    Args:
        graph_backend: A FalkorDB backend with a ``query()`` method (e.g. ``smart_graph.backend``).
    """
    indexes = [
        "CREATE INDEX FOR (m:Memory) ON (m.extraction_status)",
    ]
    for query in indexes:
        try:
            graph_backend.query(query)
        except Exception:
            pass  # Index already exists
