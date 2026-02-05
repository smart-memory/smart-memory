"""Unit tests for MemoryValidator."""

from unittest.mock import MagicMock

import pytest

from smartmemory.models.memory_item import MemoryItem
from smartmemory.validation.memory_validator import MemoryValidator, ValidationResult, ValidationIssue


@pytest.fixture
def mock_memory():
    memory = MagicMock()
    memory._graph = MagicMock()
    return memory


@pytest.fixture
def validator(mock_memory):
    return MemoryValidator(mock_memory)


class TestValidateItem:
    def test_valid_semantic_item(self, validator):
        item = MemoryItem(content="The sky is blue", memory_type="semantic")
        result = validator.validate_item(item)
        assert result.is_valid

    def test_empty_content_fails(self, validator):
        item = MemoryItem(content="", memory_type="semantic")
        result = validator.validate_item(item)
        assert not result.is_valid
        assert any(i.field == "content" for i in result.issues)

    def test_unknown_memory_type_warns(self, validator):
        item = MemoryItem(content="test", memory_type="unknown_type")
        result = validator.validate_item(item)
        assert any(i.severity == "warning" for i in result.issues)

    def test_valid_decision_item(self, validator):
        item = MemoryItem(
            content="User prefers Python",
            memory_type="decision",
            metadata={"decision_id": "dec_123", "confidence": 0.8},
        )
        result = validator.validate_item(item)
        assert result.is_valid

    def test_decision_missing_id_warns(self, validator):
        item = MemoryItem(content="test", memory_type="decision", metadata={})
        result = validator.validate_item(item)
        assert any(i.field == "decision_id" for i in result.issues)

    def test_confidence_out_of_range(self, validator):
        item = MemoryItem(
            content="test",
            memory_type="semantic",
            metadata={"confidence": 1.5},
        )
        result = validator.validate_item(item)
        assert any(i.field == "confidence" for i in result.issues)


class TestValidationResult:
    def test_errors_only(self):
        result = ValidationResult(issues=[
            ValidationIssue(severity="error", field="content", message="empty"),
            ValidationIssue(severity="warning", field="type", message="unknown"),
        ])
        assert len(result.errors) == 1
        assert len(result.warnings) == 1

    def test_is_valid_with_only_warnings(self):
        result = ValidationResult(issues=[
            ValidationIssue(severity="warning", field="type", message="unknown"),
        ])
        assert result.is_valid  # Warnings don't make it invalid

    def test_to_dict(self):
        result = ValidationResult(issues=[
            ValidationIssue(severity="error", field="content", message="empty"),
        ])
        d = result.to_dict()
        assert "issues" in d
        assert d["is_valid"] is False
