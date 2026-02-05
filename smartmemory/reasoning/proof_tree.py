"""Proof tree builder for auditable reasoning chains.

Traverses DERIVED_FROM, CAUSED_BY, PRODUCED edges to build a tree
showing how a decision was reached.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from smartmemory.models.base import MemoryBaseModel

logger = logging.getLogger(__name__)

EVIDENCE_EDGE_TYPES = {"DERIVED_FROM", "CAUSED_BY", "PRODUCED"}


@dataclass
class ProofNode(MemoryBaseModel):
    """A node in a proof tree."""

    node_id: str = ""
    content: str = ""
    node_type: str = ""
    confidence: float = 0.0
    edge_type: str = ""
    children: list["ProofNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "content": self.content[:200],
            "node_type": self.node_type,
            "confidence": self.confidence,
            "edge_type": self.edge_type,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProofNode":
        children = [cls.from_dict(c) for c in d.get("children", [])]
        return cls(
            node_id=d.get("node_id", ""),
            content=d.get("content", ""),
            node_type=d.get("node_type", ""),
            confidence=d.get("confidence", 0.0),
            edge_type=d.get("edge_type", ""),
            children=children,
        )


@dataclass
class ProofTree(MemoryBaseModel):
    """Complete proof tree for a decision."""

    root: ProofNode | None = None
    decision_id: str = ""
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "depth": self.depth,
            "root": self.root.to_dict() if self.root else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProofTree":
        root = ProofNode.from_dict(d["root"]) if d.get("root") else None
        return cls(root=root, decision_id=d.get("decision_id", ""), depth=d.get("depth", 0))

    def render_text(self, indent: int = 0) -> str:
        """Render the tree as indented text."""
        if not self.root:
            return "(empty proof tree)"
        return self._render_node(self.root, indent)

    def _render_node(self, node: ProofNode, indent: int) -> str:
        prefix = "  " * indent
        edge_label = f" [{node.edge_type}]" if node.edge_type else ""
        line = f"{prefix}{node.node_type}: {node.content[:100]} (conf={node.confidence:.2f}){edge_label}"
        lines = [line]
        for child in node.children:
            lines.append(self._render_node(child, indent + 1))
        return "\n".join(lines)


class ProofTreeBuilder:
    """Builds proof trees by traversing evidence edges in the graph.

    Usage:
        builder = ProofTreeBuilder(smart_memory._graph)
        tree = builder.build_proof("dec_123")
        print(tree.render_text())
    """

    def __init__(self, graph: Any):
        self.graph = graph

    def build_proof(self, decision_id: str, max_depth: int = 5) -> ProofTree | None:
        """Build a proof tree for a decision.

        Args:
            decision_id: Root decision to trace.
            max_depth: Maximum traversal depth.

        Returns:
            ProofTree if decision found, None otherwise.
        """
        root_node = self.graph.get_node(decision_id)
        if root_node is None:
            return None

        root = self._build_node(root_node, decision_id, depth=0, max_depth=max_depth, visited=set())
        return ProofTree(root=root, decision_id=decision_id, depth=max_depth)

    def _build_node(
        self,
        graph_node: Any,
        node_id: str,
        depth: int,
        max_depth: int,
        visited: set[str],
    ) -> ProofNode:
        """Recursively build a proof node from a graph node."""
        if node_id in visited:
            return ProofNode(node_id=node_id, content="(circular reference)", node_type="cycle")
        visited.add(node_id)

        content = getattr(graph_node, "content", "") or ""
        node_type = getattr(graph_node, "memory_type", "") or ""
        metadata = getattr(graph_node, "metadata", {}) or {}
        confidence = metadata.get("confidence", 0.0) if isinstance(metadata, dict) else 0.0

        children = []
        if depth < max_depth:
            edges = self.graph.get_edges_for_node(node_id) or []
            for edge in edges:
                if edge.get("source") == node_id and edge.get("type") in EVIDENCE_EDGE_TYPES:
                    target_id = edge.get("target")
                    if target_id and target_id not in visited:
                        child_node = self.graph.get_node(target_id)
                        if child_node:
                            child = self._build_node(child_node, target_id, depth + 1, max_depth, visited)
                            child.edge_type = edge.get("type", "")
                            children.append(child)

        return ProofNode(
            node_id=node_id,
            content=content,
            node_type=node_type,
            confidence=confidence,
            children=children,
        )
