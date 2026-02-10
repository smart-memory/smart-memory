"""
Test fixtures and factories for SmartMemory.

Provides reusable test data factories and shared fixtures.
"""

from tests.fixtures.factories import (
    unique_id,
    unique_email,
    make_memory_item,
    make_memory_item_model,
    make_decision,
    make_decision_model,
    make_entity,
    make_reasoning_step,
    make_reasoning_trace,
    make_scope_context,
    make_classification_state,
    make_extraction_state,
)

__all__ = [
    "unique_id",
    "unique_email",
    "make_memory_item",
    "make_memory_item_model",
    "make_decision",
    "make_decision_model",
    "make_entity",
    "make_reasoning_step",
    "make_reasoning_trace",
    "make_scope_context",
    "make_classification_state",
    "make_extraction_state",
]
