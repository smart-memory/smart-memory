"""Unit tests for OntologyConstrainStage."""

from unittest.mock import MagicMock, call

from smartmemory.pipeline.config import PipelineConfig, ConstrainConfig, PromotionConfig, ExtractionConfig
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.ontology_constrain import OntologyConstrainStage


def _mock_ontology(type_statuses=None):
    """Build a mock OntologyGraph.

    Args:
        type_statuses: dict mapping type name (title-case) to status string.
            Returns None for types not in the dict.
    """
    og = MagicMock()
    statuses = type_statuses or {}

    def get_status(name):
        return statuses.get(name)

    og.get_type_status.side_effect = get_status
    og.add_provisional.return_value = True
    og.promote.return_value = True
    return og


class TestOntologyConstrainStage:
    """Tests for the ontology constrain pipeline stage."""

    def test_merge_ruler_and_llm_entities(self):
        """Ruler + LLM entities are merged by name, ruler type preferred."""
        og = _mock_ontology({"Person": "seed", "Organization": "seed"})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            ruler_entities=[{"name": "John", "entity_type": "person", "confidence": 0.9}],
            llm_entities=[
                {"name": "John", "entity_type": "person", "confidence": 0.95},
                {"name": "Google", "entity_type": "organization", "confidence": 0.85},
            ],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        names = [e["name"] for e in result.entities]
        assert "John" in names
        assert "Google" in names
        # John should have higher confidence from LLM
        john = next(e for e in result.entities if e["name"] == "John")
        assert john["confidence"] == 0.95

    def test_seed_types_accepted(self):
        """Entities with seed type status are accepted."""
        og = _mock_ontology({"Person": "seed"})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            ruler_entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Alice"

    def test_confirmed_types_accepted(self):
        """Entities with confirmed type status are accepted."""
        og = _mock_ontology({"Technology": "confirmed"})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Python", "entity_type": "technology", "confidence": 0.9}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.entities) == 1

    def test_provisional_types_accepted(self):
        """Entities with provisional type status are accepted."""
        og = _mock_ontology({"Framework": "provisional"})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Django", "entity_type": "framework", "confidence": 0.8}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.entities) == 1

    def test_unknown_type_high_confidence_becomes_provisional(self):
        """Unknown type with confidence >= threshold creates provisional type."""
        og = _mock_ontology({})  # No known types
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "React", "entity_type": "library", "confidence": 0.8}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        og.add_provisional.assert_called_once_with("Library")
        assert len(result.entities) == 1
        assert len(result.promotion_candidates) == 1

    def test_unknown_type_low_confidence_rejected(self):
        """Unknown type with confidence < threshold is rejected."""
        og = _mock_ontology({})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Unknown", "entity_type": "mystery", "confidence": 0.3}],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.entities) == 0
        assert len(result.rejected) == 1

    def test_relation_filtering(self):
        """Only relations with both endpoints in accepted entities are kept."""
        og = _mock_ontology({"Person": "seed", "Organization": "seed"})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[
                {"name": "Alice", "entity_type": "person", "confidence": 0.9, "item_id": "a1"},
                {"name": "Google", "entity_type": "organization", "confidence": 0.9, "item_id": "g1"},
            ],
            llm_relations=[
                {"source_id": "a1", "target_id": "g1", "relation_type": "works_at"},
                {"source_id": "a1", "target_id": "unknown_id", "relation_type": "knows"},
            ],
        )
        config = PipelineConfig()

        result = stage.execute(state, config)

        assert len(result.relations) == 1
        assert result.relations[0]["relation_type"] == "works_at"

    def test_max_entities_limit(self):
        """Entities are truncated to max_entities."""
        og = _mock_ontology({"Person": "seed"})
        stage = OntologyConstrainStage(og)
        entities = [{"name": f"Person{i}", "entity_type": "person", "confidence": 0.9} for i in range(30)]
        state = PipelineState(text="Test.", llm_entities=entities)
        config = PipelineConfig()
        config.extraction.constrain = ConstrainConfig(max_entities=5)

        result = stage.execute(state, config)

        assert len(result.entities) == 5

    def test_max_relations_limit(self):
        """Relations are truncated to max_relations."""
        og = _mock_ontology({"Person": "seed"})
        stage = OntologyConstrainStage(og)
        entities = [
            {"name": "A", "entity_type": "person", "confidence": 0.9, "item_id": "a"},
            {"name": "B", "entity_type": "person", "confidence": 0.9, "item_id": "b"},
        ]
        relations = [{"source_id": "a", "target_id": "b", "relation_type": f"r{i}"} for i in range(50)]
        state = PipelineState(text="Test.", llm_entities=entities, llm_relations=relations)
        config = PipelineConfig()
        config.extraction.constrain = ConstrainConfig(max_relations=3)

        result = stage.execute(state, config)

        assert len(result.relations) == 3

    def test_auto_promote_without_approval(self):
        """When require_approval=False, provisional types are auto-promoted."""
        og = _mock_ontology({})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Django", "entity_type": "framework", "confidence": 0.8}],
        )
        config = PipelineConfig()
        config.extraction.promotion = PromotionConfig(require_approval=False)

        stage.execute(state, config)

        og.promote.assert_called_once_with("Framework")

    def test_require_approval_skips_promote(self):
        """When require_approval=True, promote() is not called."""
        og = _mock_ontology({})
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            llm_entities=[{"name": "Django", "entity_type": "framework", "confidence": 0.8}],
        )
        config = PipelineConfig()
        config.extraction.promotion = PromotionConfig(require_approval=True)

        stage.execute(state, config)

        og.promote.assert_not_called()

    def test_undo_clears_all_outputs(self):
        """Undo resets entities, relations, rejected, promotion_candidates."""
        og = _mock_ontology()
        stage = OntologyConstrainStage(og)
        state = PipelineState(
            text="Test.",
            entities=[{"name": "A"}],
            relations=[{"source_id": "a"}],
            rejected=[{"name": "B"}],
            promotion_candidates=[{"name": "C"}],
        )

        result = stage.undo(state)

        assert result.entities == []
        assert result.relations == []
        assert result.rejected == []
        assert result.promotion_candidates == []
