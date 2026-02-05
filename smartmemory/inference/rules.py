"""Built-in inference rules for graph enrichment.

Each rule defines a Cypher pattern to match and an edge type to create.
Rules follow the INFERRED_FROM edge schema registered in schema_validator.py.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class InferenceRule:
    """Definition of a single inference rule."""

    name: str = ""
    description: str = ""
    pattern_cypher: str = ""
    edge_type: str = ""
    confidence: float = 0.7
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "edge_type": self.edge_type,
            "confidence": self.confidence,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceRule":
        known_keys = {"name", "description", "pattern_cypher", "edge_type", "confidence", "enabled"}
        return cls(**{k: data[k] for k in known_keys if k in data})


def get_default_rules() -> list[InferenceRule]:
    """Return built-in inference rules.

    Three core rules:
    1. Causal transitivity - A CAUSES B and B CAUSES C implies A CAUSES C.
    2. Contradiction symmetry - A CONTRADICTS B implies B CONTRADICTS A.
    3. Topic inheritance - decision DERIVED_FROM semantic creates INFLUENCES.
    """
    return [
        InferenceRule(
            name="causal_transitivity",
            description="If A CAUSES B and B CAUSES C, infer A CAUSES C with decayed confidence",
            pattern_cypher=(
                "MATCH (a)-[r1:CAUSES]->(b)-[r2:CAUSES]->(c) "
                "WHERE a.item_id <> c.item_id "
                "AND NOT (a)-[:CAUSES]->(c) "
                "RETURN a.item_id AS source_id, c.item_id AS target_id, "
                "r1.confidence AS conf1, r2.confidence AS conf2"
            ),
            edge_type="CAUSES",
            confidence=0.7,
        ),
        InferenceRule(
            name="contradiction_symmetry",
            description="If A CONTRADICTS B, ensure B CONTRADICTS A",
            pattern_cypher=(
                "MATCH (a)-[:CONTRADICTS]->(b) "
                "WHERE NOT (b)-[:CONTRADICTS]->(a) "
                "RETURN b.item_id AS source_id, a.item_id AS target_id"
            ),
            edge_type="CONTRADICTS",
            confidence=1.0,
        ),
        InferenceRule(
            name="topic_inheritance",
            description="If decision DERIVED_FROM semantic, create INFLUENCES edge",
            pattern_cypher=(
                "MATCH (d {memory_type: 'decision'})-[:DERIVED_FROM]->"
                "(s {memory_type: 'semantic'}) "
                "WHERE NOT (s)-[:INFLUENCES]->(d) "
                "RETURN s.item_id AS source_id, d.item_id AS target_id"
            ),
            edge_type="INFLUENCES",
            confidence=0.6,
        ),
    ]
