from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Tuple, Type, TypeVar

from smartmemory.memory.pipeline.transactions.change_set import (
    ChangeOp,
    ChangeSet,
    CommitResult,
    PreviewResult,
    RollbackResult,
)
from smartmemory.memory.pipeline.transactions.ops import OperationApplier

ConfigT = TypeVar("ConfigT")
RequestT = TypeVar("RequestT")


class PreviewCommitRollbackMixin:
    """
    Mixin that provides uniform preview/commit/rollback using store-level appliers.
    Assumes the concrete class exposes:
      - self.config (typed) and optional self.request
      - self.stage_name: str and self.plugin_name: str
      - a compute_preview(input_obj, context) -> (result: dict, ops: list[ChangeOp])
    """

    stage_name: str = "generic"  # override in concrete class (extraction|enrichment|grounding|evolution)
    plugin_name: str = "plugin"  # override with class name or friendly name

    def preview(self, input_obj: Any, context: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None) -> PreviewResult:
        context = context or {}
        result, ops = self.compute_preview(input_obj, context)
        cs = ChangeSet.new(stage=self.stage_name, plugin=self.plugin_name, run_id=run_id)
        cs.ops.extend(ops or [])
        cs.metadata.update({
            "context": {k: v for k, v in context.items() if k not in {"memory", "vector_store"}},
        })
        return PreviewResult(result=result, change_set=cs)

    def commit(self, change_set: ChangeSet, stores: Optional[Dict[str, Any]] = None) -> CommitResult:
        """
        Apply a ChangeSet to backing stores via OperationApplier.
        Stores: optional dict with explicit store instances, e.g., {"graph": graph_store, "vector": vector_store}
        """
        applier = OperationApplier(stores or {})
        return applier.apply(change_set)

    def rollback(self, change_set_id: str, stores: Optional[Dict[str, Any]] = None) -> RollbackResult:
        applier = OperationApplier(stores or {})
        return applier.rollback(change_set_id)

    # Expected to be implemented by concrete classes
    def compute_preview(self, input_obj: Any, context: Dict[str, Any]) -> Tuple[Dict[str, Any], list[ChangeOp]]:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class BaseTypedPlugin(Generic[ConfigT, RequestT]):
    """
    Lightweight base that enforces typed config/request presence.
    Concrete classes typically also inherit PreviewCommitRollbackMixin.
    """
    config: ConfigT
    request: Optional[RequestT] = None

    def ensure_config(self, expected: Type[ConfigT]) -> ConfigT:
        if not isinstance(self.config, expected):
            raise TypeError(f"{self.__class__.__name__} requires typed config {expected.__name__}")
        return self.config
