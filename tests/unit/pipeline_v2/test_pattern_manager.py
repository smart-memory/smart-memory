"""Unit tests for PatternManager and EntityRulerStage learned pattern scan."""

import pytest

from smartmemory.ontology.pattern_manager import PatternManager
from smartmemory.graph.ontology_graph import OntologyGraph
from smartmemory.pipeline.stages.entity_ruler import _ngram_scan

from tests.unit.pipeline_v2.test_ontology_graph_extended import ExtendedMockBackend


@pytest.fixture
def backend():
    return ExtendedMockBackend()


@pytest.fixture
def graph(backend):
    return OntologyGraph(workspace_id="test", backend=backend)


# ------------------------------------------------------------------ #
# PatternManager
# ------------------------------------------------------------------ #


def test_pattern_manager_loads_patterns(graph):
    graph.add_entity_pattern("python", "Technology", 0.9, workspace_id="test")
    graph.add_entity_pattern("react", "Technology", 0.85, workspace_id="test")

    pm = PatternManager(graph, workspace_id="test")
    patterns = pm.get_patterns()

    assert "python" in patterns
    assert "react" in patterns
    assert patterns["python"] == "Technology"


def test_pattern_manager_filters_blocklist_words(graph):
    graph.add_entity_pattern("the", "Concept", 0.5, workspace_id="test")
    graph.add_entity_pattern("python", "Technology", 0.9, workspace_id="test")

    pm = PatternManager(graph, workspace_id="test")
    patterns = pm.get_patterns()

    assert "the" not in patterns
    assert "python" in patterns


def test_pattern_manager_filters_short_names(graph):
    graph.add_entity_pattern("a", "Concept", 0.5, workspace_id="test")
    graph.add_entity_pattern("qt", "Technology", 0.9, workspace_id="test")

    pm = PatternManager(graph, workspace_id="test")
    patterns = pm.get_patterns()

    assert "a" not in patterns
    assert "qt" in patterns  # length 2 passes


def test_pattern_manager_reload(graph):
    pm = PatternManager(graph, workspace_id="test")
    assert pm.pattern_count == 0

    graph.add_entity_pattern("docker", "Technology", 0.9, workspace_id="test")
    pm.reload()

    assert pm.pattern_count == 1
    assert pm.version == 2  # initial load + reload


def test_pattern_manager_includes_global_patterns(graph):
    graph.add_entity_pattern("python", "Technology", 0.9, is_global=True, source="seed")
    graph.add_entity_pattern("mylib", "Technology", 0.7, workspace_id="test", is_global=False)

    pm = PatternManager(graph, workspace_id="test")
    patterns = pm.get_patterns()

    assert "python" in patterns
    assert "mylib" in patterns


# ------------------------------------------------------------------ #
# _ngram_scan
# ------------------------------------------------------------------ #


def test_ngram_scan_finds_single_word():
    patterns = {"python": "Technology", "react": "Technology"}
    text = "I use Python for web development"
    matches = _ngram_scan(text, patterns)

    assert len(matches) == 1
    assert matches[0] == ("Python", "Technology")


def test_ngram_scan_finds_multi_word():
    patterns = {"machine learning": "Concept", "python": "Technology"}
    text = "We use machine learning with Python"
    matches = _ngram_scan(text, patterns)

    names = {m[0].lower() for m in matches}
    assert "machine learning" in names
    assert "python" in names


def test_ngram_scan_no_duplicate_subspan():
    """If 'machine learning' matches as 2-gram, 'machine' alone shouldn't match."""
    patterns = {"machine learning": "Concept", "machine": "Tool"}
    text = "We use machine learning every day"
    matches = _ngram_scan(text, patterns)

    # Should only get the 2-gram match, not the 1-gram
    assert len(matches) == 1
    assert matches[0][0].lower() == "machine learning"


def test_ngram_scan_empty_patterns():
    matches = _ngram_scan("some text here", {})
    assert matches == []


# ------------------------------------------------------------------ #
# EntityRulerStage with PatternManager
# ------------------------------------------------------------------ #


def test_entity_ruler_stage_uses_learned_patterns(graph):
    """EntityRulerStage should find entities from learned patterns."""
    from smartmemory.pipeline.stages.entity_ruler import EntityRulerStage
    from smartmemory.pipeline.state import PipelineState
    from smartmemory.pipeline.config import PipelineConfig
    from unittest.mock import MagicMock

    graph.add_entity_pattern("fastapi", "Technology", 0.9, workspace_id="test")
    pm = PatternManager(graph, workspace_id="test")

    # Mock spaCy to avoid requiring the model
    mock_nlp = MagicMock()
    mock_doc = MagicMock()
    mock_doc.ents = []
    mock_nlp.return_value = mock_doc

    stage = EntityRulerStage(nlp=mock_nlp, pattern_manager=pm)
    state = PipelineState(text="We built our API with FastAPI and Python")
    config = PipelineConfig()

    result = stage.execute(state, config)

    learned_entities = [e for e in result.ruler_entities if e.get("source") == "entity_ruler_learned"]
    assert len(learned_entities) >= 1
    names = {e["name"].lower() for e in learned_entities}
    assert "fastapi" in names
