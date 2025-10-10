from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

# ---- Change operation model -------------------------------------------------

ChangeOpType = Literal[
    "add_edge",
    "remove_edge",
    "set_properties",
    "restore_properties",
    "add_node",
    "remove_node",
    "vector_upsert",
    "vector_delete",
    "archive",
    "unarchive",
]


@dataclass
class ChangeOp:
    op_type: ChangeOpType
    args: Dict[str, Any] = field(default_factory=dict)
    inverse: Optional["ChangeOp"] = None  # populated at commit time if not provided


# ---- ChangeSet and results --------------------------------------------------

@dataclass
class ChangeSet:
    change_set_id: str
    stage: str  # extraction|enrichment|grounding|evolution
    plugin: str
    run_id: Optional[str] = None
    ops: List[ChangeOp] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(stage: str, plugin: str, run_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "ChangeSet":
        return ChangeSet(
            change_set_id=str(uuid.uuid4()),
            stage=stage,
            plugin=plugin,
            run_id=run_id,
            ops=[],
            metadata=metadata or {},
        )


@dataclass
class PreviewResult:
    result: Dict[str, Any]
    change_set: ChangeSet


@dataclass
class CommitResult:
    change_set_id: str
    applied_ops: int
    errors: List[str] = field(default_factory=list)


@dataclass
class RollbackResult:
    change_set_id: str
    reverted_ops: int
    errors: List[str] = field(default_factory=list)
