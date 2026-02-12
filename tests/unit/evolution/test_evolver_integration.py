"""Unit tests for evolver integration with evolution tracking."""

import pytest
from unittest.mock import MagicMock

from smartmemory.plugins.evolvers.working_to_procedural import (
    WorkingToProceduralEvolver,
    WorkingToProceduralConfig,
)
from smartmemory.evolution.tracker import EvolutionTracker


class TestWorkingToProceduralEvolverTracking:
    """Tests for WorkingToProceduralEvolver with evolution tracking."""

    @pytest.fixture
    def mock_tracker(self):
        """Create a mock EvolutionTracker."""
        tracker = MagicMock(spec=EvolutionTracker)
        return tracker

    @pytest.fixture
    def mock_memory(self):
        """Create a mock SmartMemory instance."""
        memory = MagicMock()
        # Set up working memory to return patterns
        memory.working.detect_skill_patterns.return_value = [
            {"name": "Test Pattern", "skills": ["skill1"], "tools": ["tool1"]}
        ]
        # Set up procedural memory
        memory.procedural.add_macro.return_value = {"item_id": "proc-123"}
        memory.procedural.search.return_value = []  # No existing procedures
        # Set up scope provider
        memory.scope_provider.get_write_context.return_value = {
            "workspace_id": "ws-789",
            "user_id": "user-abc",
        }
        return memory

    def test_creation_with_tracker(self, mock_tracker, mock_memory):
        """Test that creation events are tracked when tracker is provided."""
        config = WorkingToProceduralConfig(k=3, track_evolution=True)
        evolver = WorkingToProceduralEvolver(
            config=config,
            evolution_tracker=mock_tracker,
        )

        evolver.evolve(mock_memory)

        # Verify creation was tracked
        mock_tracker.track_creation.assert_called_once()
        call_kwargs = mock_tracker.track_creation.call_args[1]
        assert call_kwargs["procedure_id"] == "proc-123"
        assert call_kwargs["workspace_id"] == "ws-789"
        assert call_kwargs["user_id"] == "user-abc"
        assert call_kwargs["source"]["type"] == "working_memory"
        assert call_kwargs["source"]["pattern_count"] == 3

    def test_creation_without_tracker(self, mock_memory):
        """Test that evolution works without tracker."""
        config = WorkingToProceduralConfig(k=3)
        evolver = WorkingToProceduralEvolver(config=config)

        # Should not raise even without tracker
        evolver.evolve(mock_memory)

        # Macro should still be added
        mock_memory.procedural.add_macro.assert_called_once()

    def test_tracking_disabled_in_config(self, mock_tracker, mock_memory):
        """Test that tracking can be disabled in config."""
        config = WorkingToProceduralConfig(k=3, track_evolution=False)
        evolver = WorkingToProceduralEvolver(
            config=config,
            evolution_tracker=mock_tracker,
        )

        evolver.evolve(mock_memory)

        # Macro should be added but no tracking
        mock_memory.procedural.add_macro.assert_called_once()
        mock_tracker.track_creation.assert_not_called()

    def test_refinement_tracking(self, mock_tracker, mock_memory):
        """Test that refinement events are tracked for existing procedures."""
        # Set up existing procedure
        existing_proc = MagicMock()
        existing_proc.item_id = "proc-existing"
        existing_proc.content = "test pattern skills tools"  # Similar content
        existing_proc.metadata = {}
        mock_memory.procedural.search.return_value = [existing_proc]

        config = WorkingToProceduralConfig(k=3, track_evolution=True)
        evolver = WorkingToProceduralEvolver(
            config=config,
            evolution_tracker=mock_tracker,
        )

        # Pattern that's similar to existing
        mock_memory.working.detect_skill_patterns.return_value = ["test pattern skills tools refined"]

        evolver.evolve(mock_memory)

        # Verify refinement was tracked (not creation)
        mock_tracker.track_refinement.assert_called_once()
        mock_tracker.track_creation.assert_not_called()

    def test_no_workspace_no_tracking(self, mock_tracker, mock_memory):
        """Test that tracking is skipped when workspace_id is not available."""
        mock_memory.scope_provider.get_write_context.return_value = {
            "user_id": "user-abc",
            # No workspace_id
        }

        config = WorkingToProceduralConfig(k=3, track_evolution=True)
        evolver = WorkingToProceduralEvolver(
            config=config,
            evolution_tracker=mock_tracker,
        )

        evolver.evolve(mock_memory)

        # Macro should be added but no tracking (no workspace)
        mock_memory.procedural.add_macro.assert_called_once()
        mock_tracker.track_creation.assert_not_called()

    def test_tracking_error_does_not_fail_evolve(self, mock_tracker, mock_memory):
        """Test that tracking errors don't prevent evolution."""
        mock_tracker.track_creation.side_effect = Exception("Tracking failed")

        config = WorkingToProceduralConfig(k=3, track_evolution=True)
        evolver = WorkingToProceduralEvolver(
            config=config,
            evolution_tracker=mock_tracker,
        )

        # Should not raise
        logger = MagicMock()
        evolver.evolve(mock_memory, logger=logger)

        # Macro should still be added
        mock_memory.procedural.add_macro.assert_called_once()
        # Warning should be logged
        logger.warning.assert_called()


class TestPatternMetadataExtraction:
    """Tests for pattern metadata extraction."""

    @pytest.fixture
    def evolver(self):
        return WorkingToProceduralEvolver()

    def test_extract_dict_pattern(self, evolver):
        """Test metadata extraction from dict pattern."""
        pattern = {
            "name": "Test Pattern",
            "description": "A test pattern",
            "skills": ["skill1", "skill2"],
            "tools": ["tool1"],
            "steps": ["step1", "step2"],
        }
        metadata = evolver._extract_pattern_metadata(pattern)
        assert metadata["name"] == "Test Pattern"
        assert metadata["skills"] == ["skill1", "skill2"]
        assert metadata["tools"] == ["tool1"]
        assert len(metadata["steps"]) == 2

    def test_extract_string_pattern(self, evolver):
        """Test metadata extraction from string pattern."""
        pattern = "Simple string pattern"
        metadata = evolver._extract_pattern_metadata(pattern)
        assert metadata["name"] == "Simple string pattern"
        assert metadata["skills"] == []
        assert metadata["tools"] == []

    def test_extract_long_string_truncated(self, evolver):
        """Test that long string names are truncated."""
        pattern = "a" * 100
        metadata = evolver._extract_pattern_metadata(pattern)
        assert len(metadata["name"]) == 50


class TestPatternSimilarity:
    """Tests for pattern similarity checking."""

    @pytest.fixture
    def evolver(self):
        return WorkingToProceduralEvolver()

    def test_identical_patterns(self, evolver):
        """Test that identical patterns are similar."""
        assert evolver._is_similar_pattern("hello world", "hello world") is True

    def test_similar_patterns(self, evolver):
        """Test that similar patterns are detected."""
        # High word overlap
        assert (
            evolver._is_similar_pattern(
                "use tool to process data",
                "use tool to process data quickly",
            )
            is True
        )

    def test_different_patterns(self, evolver):
        """Test that different patterns are not similar."""
        assert (
            evolver._is_similar_pattern(
                "completely different content",
                "nothing in common here",
            )
            is False
        )

    def test_empty_patterns(self, evolver):
        """Test handling of empty patterns."""
        assert evolver._is_similar_pattern("", "") is False
        assert evolver._is_similar_pattern("hello", "") is False
