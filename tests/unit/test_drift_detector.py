"""Unit tests for CFS-4 DriftDetector."""

import pytest

from smartmemory.drift_detector import DriftDetector, DriftDetectorConfig
from smartmemory.models.schema_snapshot import SchemaSnapshot


class TestDriftDetector:
    """Tests for DriftDetector.check_procedure() and apply_confidence()."""

    def _make_snapshot(self, schemas: dict, procedure_id: str = "proc-1", workspace_id: str = "ws-1") -> SchemaSnapshot:
        """Create a SchemaSnapshot for testing."""
        snap = SchemaSnapshot(
            snapshot_id="snap-1",
            procedure_id=procedure_id,
            workspace_id=workspace_id,
            source_type="static",
            schemas=schemas,
        )
        snap.schema_hash = snap.compute_hash()
        return snap

    def test_disabled_returns_none(self):
        """When disabled, check_procedure returns None without checking."""
        detector = DriftDetector(config=DriftDetectorConfig(enabled=False))
        snapshot = self._make_snapshot({"tool_a": {"properties": {"x": {"type": "string"}}}})
        current = {"tool_a": {"properties": {"x": {"type": "integer"}}}}  # Breaking change
        result = detector.check_procedure(snapshot, current)
        assert result is None

    def test_no_drift_returns_none(self):
        """When schemas match exactly, returns None."""
        detector = DriftDetector(config=DriftDetectorConfig(enabled=True))
        schemas = {"tool_a": {"properties": {"x": {"type": "string"}}}}
        snapshot = self._make_snapshot(schemas)
        result = detector.check_procedure(snapshot, schemas)
        assert result is None

    def test_breaking_drift_returns_event(self):
        """Breaking drift returns a DriftEvent with action_taken='confidence_reduced'."""
        detector = DriftDetector(config=DriftDetectorConfig(enabled=True))
        old_schemas = {"tool_a": {"properties": {"x": {"type": "string"}}, "required": ["x"]}}
        new_schemas = {"tool_a": {"properties": {}}}  # Required property removed
        snapshot = self._make_snapshot(old_schemas)
        event = detector.check_procedure(snapshot, new_schemas)
        assert event is not None
        assert event.breaking_count >= 1
        assert event.action_taken == "confidence_reduced"
        assert event.procedure_id == "proc-1"
        assert event.workspace_id == "ws-1"
        assert event.snapshot_id == "snap-1"

    def test_non_breaking_drift_returns_event_flagged(self):
        """Non-breaking-only drift returns a DriftEvent with action_taken='flagged'."""
        detector = DriftDetector(config=DriftDetectorConfig(enabled=True))
        old_schemas = {"tool_a": {"properties": {"x": {"type": "string"}}}}
        new_schemas = {"tool_a": {"properties": {"x": {"type": "string"}, "y": {"type": "string"}}}}
        snapshot = self._make_snapshot(old_schemas)
        event = detector.check_procedure(snapshot, new_schemas)
        assert event is not None
        assert event.breaking_count == 0
        assert event.non_breaking_count >= 1
        assert event.action_taken == "flagged"

    def test_breaking_only_mode_skips_non_breaking(self):
        """With breaking_only=True, non-breaking changes are ignored."""
        detector = DriftDetector(config=DriftDetectorConfig(enabled=True, breaking_only=True))
        old_schemas = {"tool_a": {"properties": {"x": {"type": "string"}}}}
        new_schemas = {"tool_a": {"properties": {"x": {"type": "string"}, "y": {"type": "string"}}}}
        snapshot = self._make_snapshot(old_schemas)
        result = detector.check_procedure(snapshot, new_schemas)
        assert result is None

    def test_breaking_only_mode_detects_breaking(self):
        """With breaking_only=True, breaking changes are still detected."""
        detector = DriftDetector(config=DriftDetectorConfig(enabled=True, breaking_only=True))
        old_schemas = {"tool_a": {"properties": {"x": {"type": "string"}}, "required": ["x"]}}
        new_schemas = {"tool_a": {"properties": {}}}
        snapshot = self._make_snapshot(old_schemas)
        event = detector.check_procedure(snapshot, new_schemas)
        assert event is not None
        assert event.breaking_count >= 1

    def test_apply_confidence(self):
        """apply_confidence multiplies by configured multiplier."""
        detector = DriftDetector(config=DriftDetectorConfig(enabled=True, confidence_multiplier=0.5))
        assert detector.apply_confidence(0.92) == pytest.approx(0.46)

    def test_apply_confidence_custom_multiplier(self):
        """apply_confidence with custom multiplier."""
        detector = DriftDetector(config=DriftDetectorConfig(enabled=True, confidence_multiplier=0.3))
        assert detector.apply_confidence(1.0) == pytest.approx(0.3)

    def test_multiple_tools_mixed_drift(self):
        """Multiple tools: one breaking, one clean -> event with correct counts."""
        detector = DriftDetector(config=DriftDetectorConfig(enabled=True))
        old_schemas = {
            "tool_a": {"properties": {"x": {"type": "string"}}, "required": ["x"]},
            "tool_b": {"properties": {"y": {"type": "integer"}}},
        }
        new_schemas = {
            "tool_a": {"properties": {}},  # Breaking: required property removed
            "tool_b": {"properties": {"y": {"type": "integer"}}},  # No change
        }
        snapshot = self._make_snapshot(old_schemas)
        event = detector.check_procedure(snapshot, new_schemas)
        assert event is not None
        assert event.breaking_count >= 1
        assert event.action_taken == "confidence_reduced"

    def test_empty_snapshot_schemas_no_crash(self):
        """Empty snapshot schemas don't crash."""
        detector = DriftDetector(config=DriftDetectorConfig(enabled=True))
        snapshot = self._make_snapshot({})
        current = {"tool_a": {"properties": {"x": {"type": "string"}}}}
        event = detector.check_procedure(snapshot, current)
        # tool_a added = non-breaking
        assert event is not None
        assert event.breaking_count == 0

    def test_empty_current_schemas_no_crash(self):
        """Empty current schemas (all tools removed) -> breaking drift."""
        detector = DriftDetector(config=DriftDetectorConfig(enabled=True))
        old_schemas = {"tool_a": {"properties": {"x": {"type": "string"}}}}
        snapshot = self._make_snapshot(old_schemas)
        event = detector.check_procedure(snapshot, {})
        assert event is not None
        assert event.breaking_count >= 1
        assert event.action_taken == "confidence_reduced"

    def test_event_has_valid_event_id(self):
        """Returned DriftEvent has a non-empty event_id (UUID)."""
        detector = DriftDetector(config=DriftDetectorConfig(enabled=True))
        old_schemas = {"tool_a": {"properties": {"x": {"type": "string"}}}}
        new_schemas = {"tool_a": {"properties": {"x": {"type": "string"}, "y": {"type": "integer"}}}}
        snapshot = self._make_snapshot(old_schemas)
        event = detector.check_procedure(snapshot, new_schemas)
        assert event is not None
        assert len(event.event_id) > 0

    def test_event_diff_summary_contains_tool_name(self):
        """DriftEvent summary references the tool that drifted."""
        detector = DriftDetector(config=DriftDetectorConfig(enabled=True))
        old_schemas = {"my_special_tool": {"properties": {"x": {"type": "string"}}}}
        new_schemas = {"my_special_tool": {"properties": {"x": {"type": "integer"}}}}  # Type change
        snapshot = self._make_snapshot(old_schemas)
        event = detector.check_procedure(snapshot, new_schemas)
        assert event is not None
        assert "my_special_tool" in event.diff_summary

    def test_config_property(self):
        """config property returns the configured DriftDetectorConfig."""
        cfg = DriftDetectorConfig(enabled=True, confidence_multiplier=0.7)
        detector = DriftDetector(config=cfg)
        assert detector.config is cfg
        assert detector.config.enabled is True
        assert detector.config.confidence_multiplier == 0.7

    def test_default_config(self):
        """Default config has enabled=False, multiplier=0.5, breaking_only=False."""
        detector = DriftDetector()
        assert detector.config.enabled is False
        assert detector.config.confidence_multiplier == 0.5
        assert detector.config.breaking_only is False
