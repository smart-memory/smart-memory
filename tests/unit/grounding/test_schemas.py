"""Unit tests for grounding config schemas."""

import pytest

pytestmark = pytest.mark.unit

from smartmemory.grounding.schemas import (
    OntologyGroundingConfig,
    KnowledgeBaseGroundingConfig,
    CommonsenseGroundingConfig,
    CausalGroundingConfig,
    WikipediaGroundingConfig,
)


class TestOntologyGroundingConfig:
    def test_defaults(self):
        cfg = OntologyGroundingConfig()
        assert cfg.registry_id == "bfo"
        assert cfg.ontology_id == "default"
        assert cfg.confidence_threshold == 0.7
        assert cfg.max_results == 10

    def test_custom_values(self):
        cfg = OntologyGroundingConfig(registry_id="custom", ontology_id="bio", confidence_threshold=0.9, max_results=5)
        assert cfg.registry_id == "custom"
        assert cfg.ontology_id == "bio"
        assert cfg.confidence_threshold == 0.9
        assert cfg.max_results == 5

    def test_confidence_threshold_bounds(self):
        with pytest.raises(Exception):
            OntologyGroundingConfig(confidence_threshold=1.5)
        with pytest.raises(Exception):
            OntologyGroundingConfig(confidence_threshold=-0.1)

    def test_max_results_bounds(self):
        with pytest.raises(Exception):
            OntologyGroundingConfig(max_results=0)
        with pytest.raises(Exception):
            OntologyGroundingConfig(max_results=51)


class TestKnowledgeBaseGroundingConfig:
    def test_defaults(self):
        cfg = KnowledgeBaseGroundingConfig()
        assert cfg.sources == ["wikidata"]
        assert cfg.max_results == 10
        assert cfg.rerank is False

    def test_custom_sources(self):
        cfg = KnowledgeBaseGroundingConfig(sources=["wikidata", "dbpedia"], rerank=True)
        assert cfg.sources == ["wikidata", "dbpedia"]
        assert cfg.rerank is True

    def test_max_results_bounds(self):
        with pytest.raises(Exception):
            KnowledgeBaseGroundingConfig(max_results=0)
        with pytest.raises(Exception):
            KnowledgeBaseGroundingConfig(max_results=101)


class TestCommonsenseGroundingConfig:
    def test_defaults(self):
        cfg = CommonsenseGroundingConfig()
        assert cfg.reasoning_level == "medium"
        assert cfg.confidence_threshold == 0.7

    def test_valid_reasoning_levels(self):
        for level in ("low", "medium", "high"):
            cfg = CommonsenseGroundingConfig(reasoning_level=level)
            assert cfg.reasoning_level == level

    def test_invalid_reasoning_level(self):
        with pytest.raises(Exception):
            CommonsenseGroundingConfig(reasoning_level="ultra")


class TestCausalGroundingConfig:
    def test_defaults(self):
        cfg = CausalGroundingConfig()
        assert cfg.model_name == "gpt-5"
        assert cfg.inference_method == "llm"
        assert cfg.confidence_threshold == 0.7
        assert cfg.max_paths == 10
        assert cfg.max_hops == 3
        assert cfg.edge_weight_threshold == 0.5
        assert cfg.allow_cycles is False
        assert cfg.temporal_window_days == 365
        assert cfg.enforce_temporal_ordering is True
        assert cfg.enable_counterfactuals is False
        assert cfg.rerank_by_causal_strength is True
        assert cfg.knowledge_sources == []

    def test_valid_inference_methods(self):
        for method in ("pattern", "bayesian", "do_calculus", "llm"):
            cfg = CausalGroundingConfig(inference_method=method)
            assert cfg.inference_method == method

    def test_invalid_inference_method(self):
        with pytest.raises(Exception):
            CausalGroundingConfig(inference_method="magic")

    def test_max_hops_bounds(self):
        CausalGroundingConfig(max_hops=1)  # min
        CausalGroundingConfig(max_hops=6)  # max
        with pytest.raises(Exception):
            CausalGroundingConfig(max_hops=0)
        with pytest.raises(Exception):
            CausalGroundingConfig(max_hops=7)

    def test_counterfactuals_and_cycles(self):
        cfg = CausalGroundingConfig(enable_counterfactuals=True, allow_cycles=True)
        assert cfg.enable_counterfactuals is True
        assert cfg.allow_cycles is True


class TestWikipediaGroundingConfig:
    def test_defaults(self):
        cfg = WikipediaGroundingConfig()
        assert cfg.language == "en"
        assert cfg.confidence_threshold == 0.7
        assert cfg.max_results == 10

    def test_custom_language(self):
        cfg = WikipediaGroundingConfig(language="de")
        assert cfg.language == "de"

    def test_confidence_bounds(self):
        WikipediaGroundingConfig(confidence_threshold=0.0)
        WikipediaGroundingConfig(confidence_threshold=1.0)
        with pytest.raises(Exception):
            WikipediaGroundingConfig(confidence_threshold=-0.1)
