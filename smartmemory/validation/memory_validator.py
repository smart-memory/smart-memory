"""Runtime memory validation.

Validates MemoryItems before storage, catching data quality issues early.
Builds on existing GraphSchemaValidator but adds semantic checks.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from smartmemory.graph.models.schema_validator import get_validator
from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.memory_item import MEMORY_TYPES, MemoryItem

logger = logging.getLogger(__name__)

# Memory types that require specific metadata
DECISION_REQUIRED_METADATA = {"decision_id"}
REASONING_REQUIRED_METADATA = {"trace_id"}


@dataclass
class ValidationIssue(MemoryBaseModel):
    """A single validation finding."""

    severity: str = "error"  # error, warning, info
    field: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"severity": self.severity, "field": self.field, "message": self.message}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ValidationIssue":
        return cls(severity=d.get("severity", "error"), field=d.get("field", ""), message=d.get("message", ""))


@dataclass
class ValidationResult(MemoryBaseModel):
    """Result of validating a memory item."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Valid if no errors (warnings are OK)."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {"is_valid": self.is_valid, "issues": [i.to_dict() for i in self.issues]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ValidationResult":
        return cls(issues=[ValidationIssue.from_dict(i) for i in d.get("issues", [])])


class MemoryValidator:
    """Validates MemoryItems for data quality before storage.

    Usage:
        validator = MemoryValidator(smart_memory)
        result = validator.validate_item(item)
        if not result.is_valid:
            logger.warning(f"Validation failed: {result.errors}")
    """

    def __init__(self, memory: Any, graph: Any = None):
        self.memory = memory
        self.graph = graph or getattr(memory, "_graph", None)
        self._schema_validator = get_validator()

    def validate_item(self, item: MemoryItem) -> ValidationResult:
        """Validate a MemoryItem against schema and semantic rules.

        Args:
            item: The memory item to validate.

        Returns:
            ValidationResult with any issues found.
        """
        issues: list[ValidationIssue] = []

        # Content checks
        if not item.content or not item.content.strip():
            issues.append(ValidationIssue(severity="error", field="content", message="Content is empty"))

        # Memory type checks
        if item.memory_type not in MEMORY_TYPES:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    field="memory_type",
                    message=f"Unknown memory type: {item.memory_type}",
                )
            )

        # Metadata confidence range
        confidence = item.metadata.get("confidence")
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="confidence",
                    message=f"Confidence {confidence} out of range [0.0, 1.0]",
                )
            )

        # Type-specific metadata checks
        if item.memory_type == "decision":
            issues.extend(self._validate_decision_metadata(item))
        elif item.memory_type == "reasoning":
            issues.extend(self._validate_reasoning_metadata(item))

        return ValidationResult(issues=issues)

    def _validate_decision_metadata(self, item: MemoryItem) -> list[ValidationIssue]:
        """Check decision-specific metadata requirements."""
        issues = []
        if not item.metadata.get("decision_id"):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    field="decision_id",
                    message="Decision missing decision_id in metadata",
                )
            )
        return issues

    def _validate_reasoning_metadata(self, item: MemoryItem) -> list[ValidationIssue]:
        """Check reasoning-specific metadata requirements."""
        issues = []
        if not item.metadata.get("trace_id"):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    field="trace_id",
                    message="Reasoning trace missing trace_id in metadata",
                )
            )
        return issues
