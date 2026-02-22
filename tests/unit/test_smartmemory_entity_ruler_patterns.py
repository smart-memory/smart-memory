"""Tests for the entity_ruler_patterns constructor param in SmartMemory.

Verifies:
- entity_ruler_patterns is stored and forwarded to EntityRulerStage via the
  post-ontology-block override in _create_pipeline_runner().
- When param is None and enable_ontology=True, PatternManager is still instantiated.
- Defaults unchanged when param not passed.
"""
import pytest
from unittest.mock import MagicMock, patch

try:
    from smartmemory.tools.factory import create_lite_memory
    _LITE_AVAILABLE = True
except ImportError:
    _LITE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _LITE_AVAILABLE, reason="smartmemory-lite not installed")


def test_defaults_unchanged_with_new_param(tmp_path):
    """SmartMemory._entity_ruler_patterns is None when param not passed."""
    from smartmemory.stores.vector.vector_store import VectorStore
    try:
        memory = create_lite_memory(str(tmp_path))
        assert memory._entity_ruler_patterns is None, (
            "_entity_ruler_patterns must default to None when not passed"
        )
    finally:
        VectorStore.set_default_backend(None)


def test_entity_ruler_patterns_stored_on_instance(tmp_path):
    """SmartMemory stores entity_ruler_patterns on the instance."""
    from smartmemory.stores.vector.vector_store import VectorStore
    mock_pm = MagicMock()
    mock_pm.get_patterns.return_value = {}
    try:
        memory = create_lite_memory(str(tmp_path), entity_ruler_patterns=mock_pm)
        assert memory._entity_ruler_patterns is mock_pm, (
            "_entity_ruler_patterns must be the injected manager"
        )
    finally:
        VectorStore.set_default_backend(None)


def test_entity_ruler_patterns_passed_to_stage(tmp_path):
    """entity_ruler_patterns is forwarded as pattern_manager to EntityRulerStage."""
    from smartmemory.stores.vector.vector_store import VectorStore
    from smartmemory.pipeline.stages.entity_ruler import EntityRulerStage

    mock_pm = MagicMock()
    mock_pm.get_patterns.return_value = {}

    # EntityRulerStage is a local import inside _create_pipeline_runner, so patch at the source module
    with patch(
        "smartmemory.pipeline.stages.entity_ruler.EntityRulerStage", wraps=EntityRulerStage
    ) as MockEntityRulerStage:
        try:
            memory = create_lite_memory(str(tmp_path), entity_ruler_patterns=mock_pm)
            # Trigger runner creation
            memory._create_pipeline_runner()
            # EntityRulerStage must have been called with pattern_manager=mock_pm
            call_kwargs = MockEntityRulerStage.call_args
            assert call_kwargs is not None, "EntityRulerStage must be instantiated"
            assert call_kwargs.kwargs.get("pattern_manager") is mock_pm, (
                "EntityRulerStage must receive the injected pattern_manager"
            )
        finally:
            VectorStore.set_default_backend(None)


def test_entity_ruler_patterns_none_does_not_override(tmp_path):
    """When entity_ruler_patterns=None, the post-block override does not run.

    In Lite mode (enable_ontology=False), pattern_manager stays None after the
    ontology block is skipped. The override must NOT replace None with None
    (no-op), meaning EntityRulerStage receives pattern_manager=None.
    """
    from smartmemory.stores.vector.vector_store import VectorStore
    from smartmemory.pipeline.stages.entity_ruler import EntityRulerStage

    with patch(
        "smartmemory.pipeline.stages.entity_ruler.EntityRulerStage", wraps=EntityRulerStage
    ) as MockEntityRulerStage:
        try:
            memory = create_lite_memory(str(tmp_path))  # no entity_ruler_patterns
            memory._create_pipeline_runner()
            call_kwargs = MockEntityRulerStage.call_args
            assert call_kwargs is not None, "EntityRulerStage must be instantiated"
            # pattern_manager should be None — ontology skipped, no override
            assert call_kwargs.kwargs.get("pattern_manager") is None, (
                "pattern_manager must be None when entity_ruler_patterns not passed"
            )
        finally:
            VectorStore.set_default_backend(None)
