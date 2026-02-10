"""
Test data factories for SmartMemory.

Provides reusable functions to create test data with unique IDs,
reducing boilerplate across test files and ensuring consistent shapes.

Usage:
    from tests.fixtures.factories import make_memory_item, make_decision, unique_id

    item = make_memory_item(content="Test content", memory_type="semantic")
    decision = make_decision(content="User prefers X", confidence=0.9)
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional


def unique_id(prefix: str = "test") -> str:
    """Generate unique test ID."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def unique_email() -> str:
    """Generate unique test email."""
    return f"test_{uuid.uuid4().hex[:8]}@example.com"


def make_memory_item(
    content: str = "Test content",
    memory_type: str = "semantic",
    item_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    embedding: Optional[list] = None,
    **kwargs: Any,
) -> dict:
    """Create memory item dict with sensible defaults.

    Returns dict for flexibility - can be used with MemoryItem.from_dict()
    or passed directly to APIs.
    """
    return {
        "content": content,
        "memory_type": memory_type,
        "item_id": item_id or unique_id("mem"),
        "metadata": metadata or {},
        "embedding": embedding,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }


def make_memory_item_model(
    content: str = "Test content",
    memory_type: str = "semantic",
    **kwargs: Any,
):
    """Create actual MemoryItem model instance."""
    from smartmemory.models.memory_item import MemoryItem

    return MemoryItem(
        content=content,
        memory_type=memory_type,
        **kwargs,
    )


def make_decision(
    content: str = "Test decision",
    decision_type: str = "inference",
    confidence: float = 0.8,
    source_type: str = "inferred",
    decision_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Create decision dict with sensible defaults."""
    return {
        "decision_id": decision_id or unique_id("dec"),
        "content": content,
        "decision_type": decision_type,
        "confidence": confidence,
        "source_type": source_type,
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }


def make_decision_model(
    content: str = "Test decision",
    decision_type: str = "inference",
    confidence: float = 0.8,
    **kwargs: Any,
):
    """Create actual Decision model instance."""
    from smartmemory.models.decision import Decision

    return Decision(
        content=content,
        decision_type=decision_type,
        confidence=confidence,
        **kwargs,
    )


def make_entity(
    name: str = "Test Entity",
    entity_type: str = "CONCEPT",
    entity_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Create entity dict with sensible defaults."""
    return {
        "entity_id": entity_id or unique_id("ent"),
        "name": name,
        "entity_type": entity_type,
        **kwargs,
    }


def make_reasoning_step(
    step_type: str = "thought",
    content: str = "Test reasoning step",
    **kwargs: Any,
) -> dict:
    """Create reasoning step dict."""
    return {
        "type": step_type,
        "content": content,
        **kwargs,
    }


def make_reasoning_trace(
    trace_id: Optional[str] = None,
    steps: Optional[list] = None,
    goal: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Create reasoning trace dict with sensible defaults."""
    return {
        "trace_id": trace_id or unique_id("trace"),
        "steps": steps or [make_reasoning_step()],
        "task_context": {"goal": goal or "Test goal"} if goal else None,
        **kwargs,
    }


def make_scope_context(
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> dict:
    """Create scope context dict for multi-tenancy tests."""
    return {
        "workspace_id": workspace_id or unique_id("ws"),
        "user_id": user_id or unique_id("user"),
        "tenant_id": tenant_id or unique_id("tenant"),
        "team_id": team_id,
    }


# --- Pipeline state factories ---


def make_classification_state(
    content: str = "Test content",
    memory_type: str = "semantic",
    **kwargs: Any,
) -> dict:
    """Create classification state for pipeline tests."""
    return {
        "item": make_memory_item(content=content, memory_type=memory_type),
        "classification": {"memory_type": memory_type, "confidence": 0.9},
        **kwargs,
    }


def make_extraction_state(
    content: str = "Test content",
    entities: Optional[list] = None,
    relations: Optional[list] = None,
    **kwargs: Any,
) -> dict:
    """Create extraction state for pipeline tests."""
    return {
        "item": make_memory_item(content=content),
        "entities": entities or [make_entity()],
        "relations": relations or [],
        **kwargs,
    }
