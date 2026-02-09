"""Unit tests for grounding registry."""

import pytest

pytestmark = pytest.mark.unit

from smartmemory.grounding.registry import GROUNDING_REGISTRY
from smartmemory.grounding.schemas import (
    OntologyGroundingConfig,
    KnowledgeBaseGroundingConfig,
    CommonsenseGroundingConfig,
    CausalGroundingConfig,
    WikipediaGroundingConfig,
)
from smartmemory.grounding.executors import (
    run_ontology_grounding,
    run_knowledge_base_grounding,
    run_commonsense_grounding,
    run_causal_grounding,
    run_wikipedia_grounding,
)


class TestGroundingRegistry:
    def test_contains_all_five_entries(self):
        expected_keys = {
            "ontology_grounding",
            "knowledge_base",
            "commonsense_grounding",
            "causal_grounding",
            "wikipedia_grounding",
        }
        assert set(GROUNDING_REGISTRY.keys()) == expected_keys

    def test_each_entry_is_config_executor_tuple(self):
        for key, (config_cls, executor_fn) in GROUNDING_REGISTRY.items():
            assert callable(executor_fn), f"{key} executor is not callable"
            # Config class should be instantiable with defaults
            cfg = config_cls()
            assert cfg is not None

    def test_ontology_entry(self):
        config_cls, executor = GROUNDING_REGISTRY["ontology_grounding"]
        assert config_cls is OntologyGroundingConfig
        assert executor is run_ontology_grounding

    def test_knowledge_base_entry(self):
        config_cls, executor = GROUNDING_REGISTRY["knowledge_base"]
        assert config_cls is KnowledgeBaseGroundingConfig
        assert executor is run_knowledge_base_grounding

    def test_commonsense_entry(self):
        config_cls, executor = GROUNDING_REGISTRY["commonsense_grounding"]
        assert config_cls is CommonsenseGroundingConfig
        assert executor is run_commonsense_grounding

    def test_causal_entry(self):
        config_cls, executor = GROUNDING_REGISTRY["causal_grounding"]
        assert config_cls is CausalGroundingConfig
        assert executor is run_causal_grounding

    def test_wikipedia_entry(self):
        config_cls, executor = GROUNDING_REGISTRY["wikipedia_grounding"]
        assert config_cls is WikipediaGroundingConfig
        assert executor is run_wikipedia_grounding

    def test_executor_returns_valid_result(self):
        """Each executor should return a dict with success, artifacts, metrics, schema_version."""
        for key, (config_cls, executor_fn) in GROUNDING_REGISTRY.items():
            cfg = config_cls()
            result = executor_fn({}, cfg)
            assert isinstance(result, dict), f"{key} executor did not return dict"
            assert "success" in result
            assert "artifacts" in result
            assert "metrics" in result
            assert "schema_version" in result
