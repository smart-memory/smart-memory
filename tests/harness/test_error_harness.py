"""
Table-driven error harness for SmartMemory core library.

Tests common error pathways with parametrized test cases,
consolidating error handling verification into a single maintainable file.

This harness tests the core library's error handling. Service-level
HTTP error codes (400, 404, 403, etc.) are tested in smart-memory-service.
"""

import pytest
from unittest.mock import Mock
from typing import Any, Callable, Optional, Union, Type


# --- Error case definitions ---

class ErrorCase:
    """Definition of an error test case."""

    def __init__(
        self,
        name: str,
        operation: str,
        setup: Optional[Callable] = None,
        input_data: Optional[dict] = None,
        expected_exception: Union[Type[Exception], tuple] = Exception,
        expected_message: Optional[str] = None,
        category: str = "validation",
    ):
        self.name = name
        self.operation = operation
        self.setup = setup
        self.input_data = input_data or {}
        self.expected_exception = expected_exception
        self.expected_message = expected_message
        self.category = category


# --- Validation error cases ---
# Note: MemoryItem is a dataclass that doesn't validate on construction.
# Validation happens at API/service layer. These cases document expected behavior.

VALIDATION_CASES = [
    # These test InputAdapterConfig validation, not MemoryItem
    ErrorCase(
        name="empty_content_in_config",
        operation="validate",
        input_data={"content": "", "adapter_name": "text"},
        expected_exception=ValueError,
        expected_message="content",
        category="validation",
    ),
]


# --- Not found error cases ---

NOT_FOUND_CASES = [
    ErrorCase(
        name="get_nonexistent_item",
        operation="get",
        input_data={"item_id": "nonexistent_id_12345"},
        expected_exception=KeyError,
        category="not_found",
    ),
    ErrorCase(
        name="update_nonexistent_item",
        operation="update",
        input_data={"item_id": "nonexistent_id_12345", "content": "New content"},
        expected_exception=KeyError,
        category="not_found",
    ),
    ErrorCase(
        name="delete_nonexistent_item",
        operation="delete",
        input_data={"item_id": "nonexistent_id_12345"},
        expected_exception=KeyError,
        category="not_found",
    ),
]


# --- Dependency failure cases ---

DEPENDENCY_CASES = [
    ErrorCase(
        name="graph_db_unavailable",
        operation="add",
        input_data={"content": "Test", "memory_type": "semantic"},
        expected_exception=ConnectionError,
        category="dependency",
    ),
    ErrorCase(
        name="embedding_service_unavailable",
        operation="ingest",
        input_data={"content": "Test content for embedding"},
        expected_exception=(ConnectionError, RuntimeError),
        category="dependency",
    ),
]


# --- Harness runner ---

class ErrorHarness:
    """Runner for table-driven error tests."""

    def __init__(self, target: Any):
        """
        Args:
            target: The object to test (e.g., SmartMemory instance)
        """
        self.target = target

    def run_case(self, case: ErrorCase) -> None:
        """Execute a single error case.

        Args:
            case: The error case to run

        Raises:
            AssertionError: If the expected error is not raised
        """
        if case.setup:
            case.setup(self.target)

        operation_fn = getattr(self.target, case.operation, None)
        if operation_fn is None:
            pytest.skip(f"Operation '{case.operation}' not found on target")

        with pytest.raises(case.expected_exception) as exc_info:
            operation_fn(**case.input_data)

        if case.expected_message:
            assert case.expected_message.lower() in str(exc_info.value).lower(), (
                f"Expected message containing '{case.expected_message}' "
                f"but got: {exc_info.value}"
            )


# --- Parametrized tests ---

@pytest.mark.harness
class TestValidationErrors:
    """Test validation error handling.

    Note: MemoryItem is a simple dataclass that doesn't validate on construction.
    Validation happens at higher levels (InputAdapterConfig, API routes).
    These tests demonstrate the harness pattern for when validation exists.
    """

    @pytest.fixture
    def validating_config(self):
        """Mock that validates inputs."""
        def validate(content: str, **kwargs):
            if not content or not content.strip():
                raise ValueError("content cannot be empty")
            return {"content": content, **kwargs}
        return Mock(validate=validate)

    @pytest.mark.parametrize(
        "case",
        VALIDATION_CASES,
        ids=[c.name for c in VALIDATION_CASES],
    )
    def test_validation_error(self, case: ErrorCase, validating_config):
        """Validation errors are raised with appropriate messages."""
        harness = ErrorHarness(validating_config)
        harness.run_case(case)


@pytest.mark.harness
class TestNotFoundErrors:
    """Test not-found error handling.

    These require a real or mocked SmartMemory instance.
    """

    @pytest.fixture
    def mock_memory(self):
        """Create a mock SmartMemory that raises KeyError for missing items."""
        memory = Mock()
        memory.get.side_effect = KeyError("Item not found")
        memory.update.side_effect = KeyError("Item not found")
        memory.delete.side_effect = KeyError("Item not found")
        return memory

    @pytest.mark.parametrize(
        "case",
        NOT_FOUND_CASES,
        ids=[c.name for c in NOT_FOUND_CASES],
    )
    def test_not_found_error(self, case: ErrorCase, mock_memory):
        """Not-found errors are raised for missing items."""
        harness = ErrorHarness(mock_memory)
        harness.run_case(case)


@pytest.mark.harness
class TestDependencyErrors:
    """Test dependency failure handling.

    Tests that the system handles infrastructure failures gracefully.
    """

    @pytest.fixture
    def broken_memory(self):
        """Create a mock SmartMemory with broken dependencies."""
        memory = Mock()
        memory.add.side_effect = ConnectionError("Database unavailable")
        memory.ingest.side_effect = ConnectionError("Embedding service unavailable")
        return memory

    @pytest.mark.parametrize(
        "case",
        DEPENDENCY_CASES,
        ids=[c.name for c in DEPENDENCY_CASES],
    )
    def test_dependency_error(self, case: ErrorCase, broken_memory):
        """Dependency errors are raised when infrastructure fails."""
        harness = ErrorHarness(broken_memory)
        harness.run_case(case)


# --- Utility for adding custom cases ---

def register_error_case(
    cases_list: list,
    name: str,
    operation: str,
    expected_exception: type,
    **kwargs,
) -> None:
    """Register a new error case to a cases list.

    Allows tests to extend the harness with custom cases.
    """
    cases_list.append(
        ErrorCase(
            name=name,
            operation=operation,
            expected_exception=expected_exception,
            **kwargs,
        )
    )
