"""Unit tests for grounding executors."""

import pytest

pytestmark = pytest.mark.unit

from smartmemory.grounding.executors import (
    SCHEMA_VERSION,
    _result_stub,
    run_ontology_grounding,
    run_knowledge_base_grounding,
    run_commonsense_grounding,
    run_causal_grounding,
    run_wikipedia_grounding,
)
from smartmemory.grounding.schemas import (
    OntologyGroundingConfig,
    KnowledgeBaseGroundingConfig,
    CommonsenseGroundingConfig,
    CausalGroundingConfig,
    WikipediaGroundingConfig,
)


class TestResultStub:
    def test_structure(self):
        result = _result_stub("test_stage")
        assert result["success"] is True
        assert result["artifacts"] == []
        assert result["metrics"]["stage"] == "test_stage"
        assert result["metrics"]["processed"] == 0
        assert result["schema_version"] == SCHEMA_VERSION

    def test_schema_version_format(self):
        assert SCHEMA_VERSION.startswith("grounding@")


class TestRunOntologyGrounding:
    def test_returns_stub(self):
        cfg = OntologyGroundingConfig()
        result = run_ontology_grounding({"item_id": "test"}, cfg)
        assert result["success"] is True
        assert result["metrics"]["stage"] == "ontology_grounding"

    def test_accepts_custom_config(self):
        cfg = OntologyGroundingConfig(registry_id="custom", confidence_threshold=0.9)
        result = run_ontology_grounding({}, cfg)
        assert result["success"] is True


class TestRunKnowledgeBaseGrounding:
    def test_returns_stub(self):
        cfg = KnowledgeBaseGroundingConfig()
        result = run_knowledge_base_grounding({"item_id": "test"}, cfg)
        assert result["success"] is True
        assert result["metrics"]["stage"] == "knowledge_base"


class TestRunCommonsenseGrounding:
    def test_returns_stub(self):
        cfg = CommonsenseGroundingConfig()
        result = run_commonsense_grounding({}, cfg)
        assert result["success"] is True
        assert result["metrics"]["stage"] == "commonsense_grounding"


class TestRunCausalGrounding:
    def test_returns_stub(self):
        cfg = CausalGroundingConfig()
        result = run_causal_grounding({}, cfg)
        assert result["success"] is True
        assert result["metrics"]["stage"] == "causal_grounding"


class TestRunWikipediaGrounding:
    def test_returns_stub(self):
        cfg = WikipediaGroundingConfig()
        result = run_wikipedia_grounding({}, cfg)
        assert result["success"] is True
        assert result["metrics"]["stage"] == "wikipedia_grounding"
