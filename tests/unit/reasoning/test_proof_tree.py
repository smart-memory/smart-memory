"""Unit tests for ProofTreeBuilder."""

from unittest.mock import MagicMock

import pytest

from smartmemory.reasoning.proof_tree import ProofTreeBuilder, ProofTree, ProofNode


@pytest.fixture
def mock_graph():
    return MagicMock()


@pytest.fixture
def builder(mock_graph):
    return ProofTreeBuilder(mock_graph)


class TestProofNode:
    def test_to_dict(self):
        node = ProofNode(node_id="dec_1", content="Test decision", node_type="decision", confidence=0.9)
        d = node.to_dict()
        assert d["node_id"] == "dec_1"
        assert d["confidence"] == 0.9

    def test_with_children(self):
        child = ProofNode(node_id="sem_1", content="Evidence", node_type="semantic", confidence=0.8)
        parent = ProofNode(
            node_id="dec_1", content="Decision", node_type="decision", confidence=0.9, children=[child],
        )
        assert len(parent.children) == 1


class TestProofTree:
    def test_to_dict(self):
        root = ProofNode(node_id="dec_1", content="Root", node_type="decision", confidence=0.9)
        tree = ProofTree(root=root, decision_id="dec_1")
        d = tree.to_dict()
        assert d["decision_id"] == "dec_1"
        assert "root" in d

    def test_render_text(self):
        child = ProofNode(
            node_id="sem_1", content="Supporting fact", node_type="semantic",
            confidence=0.8, edge_type="DERIVED_FROM",
        )
        root = ProofNode(
            node_id="dec_1", content="Main decision", node_type="decision",
            confidence=0.9, children=[child],
        )
        tree = ProofTree(root=root, decision_id="dec_1")
        text = tree.render_text()
        assert "Main decision" in text
        assert "Supporting fact" in text


class TestBuildProof:
    def test_builds_simple_tree(self, builder, mock_graph):
        root_node = MagicMock()
        root_node.item_id = "dec_1"
        root_node.content = "User prefers Python"
        root_node.memory_type = "decision"
        root_node.metadata = {"confidence": 0.9}

        evidence_node = MagicMock()
        evidence_node.item_id = "sem_1"
        evidence_node.content = "User mentioned Python multiple times"
        evidence_node.memory_type = "semantic"
        evidence_node.metadata = {"confidence": 0.95}

        mock_graph.get_node.side_effect = lambda item_id: root_node if item_id == "dec_1" else evidence_node
        mock_graph.get_edges_for_node.side_effect = lambda item_id: (
            [{"source": "dec_1", "target": "sem_1", "type": "DERIVED_FROM"}]
            if item_id == "dec_1"
            else []
        )

        tree = builder.build_proof("dec_1")
        assert tree is not None
        assert tree.root.node_id == "dec_1"
        assert len(tree.root.children) == 1

    def test_returns_none_for_missing_node(self, builder, mock_graph):
        mock_graph.get_node.return_value = None
        tree = builder.build_proof("nonexistent")
        assert tree is None

    def test_respects_max_depth(self, builder, mock_graph):
        node = MagicMock()
        node.item_id = "dec_1"
        node.content = "Test"
        node.memory_type = "decision"
        node.metadata = {"confidence": 0.9}
        mock_graph.get_node.return_value = node
        mock_graph.get_edges_for_node.return_value = [
            {"source": "dec_1", "target": "dec_2", "type": "DERIVED_FROM"}
        ]
        tree = builder.build_proof("dec_1", max_depth=1)
        assert tree is not None
