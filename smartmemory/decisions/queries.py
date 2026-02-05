"""Decision query patterns.

Provides high-level query methods for decisions: filtering, provenance chains,
causal chain traversal. All operations are synchronous.
"""

import logging
from typing import Any, Literal

from smartmemory.models.decision import Decision

logger = logging.getLogger(__name__)


class DecisionQueries:
    """Query patterns for decisions - filtered retrieval, provenance, causal chains.

    Usage:
        queries = DecisionQueries(smart_memory)
        active = queries.get_active_decisions(domain="preferences")
        chain = queries.get_causal_chain(decision_id, direction="causes")
    """

    def __init__(self, memory: Any, graph: Any = None):
        """Initialize with SmartMemory instance.

        Args:
            memory: SmartMemory instance for search operations.
            graph: SmartGraph instance (defaults to memory._graph if available).
        """
        self.memory = memory
        self.graph = graph or getattr(memory, '_graph', None)

    def get_active_decisions(
        self,
        domain: str | None = None,
        decision_type: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[Decision]:
        """Get all active decisions, optionally filtered.

        Args:
            domain: Filter by domain (e.g., "preferences", "facts").
            decision_type: Filter by type (e.g., "inference", "preference").
            min_confidence: Minimum confidence threshold.
            limit: Maximum number of results.

        Returns:
            List of active Decision objects matching filters.
        """
        # Use broad search to get decision-type memories
        try:
            items = self.memory.search(query="*", memory_type="decision", top_k=limit * 2)
        except Exception:
            # Fallback: some backends don't support wildcard search
            try:
                items = self.memory.search(query="decision", memory_type="decision", top_k=limit * 2)
            except Exception as e:
                logger.warning(f"Failed to search for decisions: {e}")
                return []

        decisions = []
        for item in items:
            d = self._to_decision(item)
            if d is None or not d.is_active:
                continue
            if domain and d.domain != domain:
                continue
            if decision_type and d.decision_type != decision_type:
                continue
            if d.confidence < min_confidence:
                continue
            decisions.append(d)
            if len(decisions) >= limit:
                break

        return decisions

    def get_decisions_about(self, topic: str, limit: int = 20) -> list[Decision]:
        """Get active decisions related to a topic via semantic search.

        Args:
            topic: Search query for semantic matching.
            limit: Maximum number of results.

        Returns:
            List of active decisions relevant to the topic.
        """
        try:
            items = self.memory.search(query=topic, memory_type="decision", top_k=limit)
        except Exception as e:
            logger.warning(f"Failed to search decisions for topic '{topic}': {e}")
            return []

        decisions = []
        for item in items:
            d = self._to_decision(item)
            if d is not None and d.is_active:
                decisions.append(d)

        return decisions

    def get_decision_provenance(self, decision_id: str) -> dict[str, Any]:
        """Get full provenance chain for a decision.

        Returns the decision, its source reasoning trace, supporting evidence,
        and any decisions it superseded.

        Args:
            decision_id: The decision to trace provenance for.

        Returns:
            Dict with keys: decision, reasoning_trace, evidence, superseded.
        """
        result: dict[str, Any] = {
            "decision": None,
            "reasoning_trace": None,
            "evidence": [],
            "superseded": [],
        }

        # Get the decision itself
        item = self.memory.get(decision_id)
        if item is None:
            return result
        result["decision"] = self._to_decision(item)

        if not self.graph:
            return result

        # Get source reasoning trace (PRODUCED edge pointing to this decision)
        incoming = self.graph.get_incoming_neighbors(decision_id, edge_type="PRODUCED")
        if incoming:
            trace_node = incoming[0]
            result["reasoning_trace"] = trace_node

        # Get evidence (DERIVED_FROM edges from this decision)
        neighbors = self.graph.get_neighbors(decision_id, edge_type="DERIVED_FROM")
        if neighbors:
            for node in neighbors:
                node_id = node.get('item_id') if isinstance(node, dict) else getattr(node, 'item_id', None)
                result["evidence"].append({
                    "memory": node,
                    "memory_id": node_id,
                })

        # Get superseded decisions (SUPERSEDES edges from this decision)
        superseded = self.graph.get_neighbors(decision_id, edge_type="SUPERSEDES")
        if superseded:
            for node in superseded:
                d = self._to_decision(node)
                if d:
                    result["superseded"].append(d)

        return result

    def get_causal_chain(
        self,
        decision_id: str,
        direction: Literal["causes", "effects", "both"] = "both",
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """Trace causal chain from a decision.

        Args:
            decision_id: Starting decision.
            direction: "causes" (what led to this), "effects" (what it influenced), or "both".
            max_depth: Maximum traversal depth.

        Returns:
            Dict with keys: decision, causes, effects.
        """
        item = self.memory.get(decision_id)
        result: dict[str, Any] = {
            "decision": self._to_decision(item) if item else None,
            "causes": [],
            "effects": [],
        }

        if not self.graph or item is None:
            return result

        if direction in ("causes", "both"):
            result["causes"] = self._traverse_causes(decision_id, max_depth)

        if direction in ("effects", "both"):
            result["effects"] = self._traverse_effects(decision_id, max_depth)

        return result

    def _traverse_causes(self, node_id: str, max_depth: int, current_depth: int = 0) -> list[dict[str, Any]]:
        """Recursively traverse cause edges (DERIVED_FROM, CAUSED_BY)."""
        if current_depth >= max_depth or not self.graph:
            return []

        causes: list[dict[str, Any]] = []

        for edge_type in ["DERIVED_FROM", "CAUSED_BY"]:
            neighbors = self.graph.get_neighbors(node_id, edge_type=edge_type)
            if not neighbors:
                continue
            for node in neighbors:
                node_id_val = node.get('item_id') if isinstance(node, dict) else getattr(node, 'item_id', None)
                causes.append({
                    "node": node,
                    "node_id": node_id_val,
                    "relationship": edge_type,
                    "depth": current_depth + 1,
                    "causes": self._traverse_causes(node_id_val, max_depth, current_depth + 1) if node_id_val else [],
                })

        return causes

    def _traverse_effects(self, node_id: str, max_depth: int, current_depth: int = 0) -> list[dict[str, Any]]:
        """Recursively traverse effect edges (CAUSES, INFLUENCES, PRODUCED)."""
        if current_depth >= max_depth or not self.graph:
            return []

        effects: list[dict[str, Any]] = []

        # Effects are nodes that this node points to via causal edges
        # Use incoming neighbors for edges where this node is the source
        for edge_type in ["CAUSES", "INFLUENCES", "PRODUCED"]:
            # get_neighbors follows outgoing edges from the node
            neighbors = self.graph.get_neighbors(node_id, edge_type=edge_type)
            if not neighbors:
                continue
            for node in neighbors:
                node_id_val = node.get('item_id') if isinstance(node, dict) else getattr(node, 'item_id', None)
                effects.append({
                    "node": node,
                    "node_id": node_id_val,
                    "relationship": edge_type,
                    "depth": current_depth + 1,
                    "effects": self._traverse_effects(node_id_val, max_depth, current_depth + 1) if node_id_val else [],
                })

        return effects

    def _to_decision(self, item: Any) -> Decision | None:
        """Convert a MemoryItem (or dict) to a Decision."""
        if item is None:
            return None
        if isinstance(item, Decision):
            return item
        metadata = getattr(item, 'metadata', None)
        if isinstance(item, dict):
            metadata = item.get('metadata', item)
        if not isinstance(metadata, dict):
            return None
        if 'content' not in metadata or not metadata['content']:
            content = getattr(item, 'content', None) or (item.get('content') if isinstance(item, dict) else '')
            metadata['content'] = content
        return Decision.from_dict(metadata)
