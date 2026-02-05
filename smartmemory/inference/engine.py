"""Inference engine that applies rules to enrich the graph.

Executes pattern-matching Cypher queries from InferenceRules and creates
inferred edges with provenance metadata.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from smartmemory.models.base import MemoryBaseModel

from .rules import InferenceRule, get_default_rules

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult(MemoryBaseModel):
    """Result of running inference rules."""

    edges_created: int = 0
    rules_applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges_created": self.edges_created,
            "rules_applied": self.rules_applied,
            "errors": self.errors,
        }


class InferenceEngine:
    """Applies inference rules to automatically enrich the graph.

    Usage:
        engine = InferenceEngine(smart_memory)
        result = engine.run()
        print(f"Created {result.edges_created} inferred edges")
    """

    def __init__(self, memory: Any, graph: Any = None, rules: list[InferenceRule] | None = None):
        self.memory = memory
        self.graph = graph or getattr(memory, "_graph", None)
        self.rules = rules if rules is not None else get_default_rules()

    def run(self, dry_run: bool = False) -> InferenceResult:
        """Execute all enabled inference rules.

        Args:
            dry_run: If True, find matches but don't create edges.

        Returns:
            InferenceResult with counts and any errors.
        """
        result = InferenceResult()

        for rule in self.rules:
            if not rule.enabled:
                continue
            try:
                count = self._apply_rule(rule, dry_run=dry_run)
                if count > 0:
                    result.edges_created += count
                    result.rules_applied.append(rule.name)
            except Exception as e:
                logger.warning("Inference rule '%s' failed: %s", rule.name, e)
                result.errors.append(f"{rule.name}: {e}")

        return result

    def _apply_rule(self, rule: InferenceRule, dry_run: bool = False) -> int:
        """Apply a single inference rule."""
        if not self.graph:
            return 0

        rows = self._cypher(rule.pattern_cypher)
        if not rows:
            return 0

        if dry_run:
            return len(rows)

        count = 0
        now = datetime.now(timezone.utc).isoformat()

        for row in rows:
            source_id = row[0]
            target_id = row[1]

            confidence = rule.confidence
            if len(row) > 3 and row[2] is not None and row[3] is not None:
                confidence = min(rule.confidence, float(row[2]) * float(row[3]))

            try:
                self._create_inferred_edge(
                    source_id,
                    target_id,
                    rule.edge_type,
                    rule_name=rule.name,
                    confidence=confidence,
                    inferred_at=now,
                )
                count += 1
            except Exception as e:
                logger.debug("Failed to create inferred edge %s->%s: %s", source_id, target_id, e)

        return count

    def _create_inferred_edge(self, source_id: str, target_id: str, edge_type: str, **properties: Any) -> None:
        """Create an inferred edge with provenance metadata."""
        cypher = f"MATCH (a {{item_id: $source_id}}), (b {{item_id: $target_id}}) CREATE (a)-[:{edge_type} $props]->(b)"
        self._cypher(
            cypher,
            {
                "source_id": source_id,
                "target_id": target_id,
                "props": properties,
            },
        )

    def _cypher(self, query: str, params: dict | None = None) -> list:
        if not self.graph:
            return []
        try:
            return self.graph.backend.execute_cypher(query, params or {})
        except Exception as e:
            logger.debug("Cypher query failed: %s", e)
            return []
