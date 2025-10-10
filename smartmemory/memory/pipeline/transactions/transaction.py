from __future__ import annotations

from typing import Any, Dict, List, Optional

from smartmemory.memory.pipeline.transactions.change_set import ChangeOp


class TransactionalGraphProxy:
    """
    Proxy for a graph store that records ChangeOps instead of performing writes.
    Use to capture a preview of mutations.
    """

    def __init__(self, ops: List[ChangeOp]):
        self._ops = ops

    # --- Graph-like methods --------------------------------------------------

    def add_edge(self, source: str, target: str, relation_type: str = "RELATED", properties: Optional[Dict[str, Any]] = None):
        self._ops.append(ChangeOp(op_type="add_edge", args={"source": source, "target": target, "relation_type": relation_type, "properties": properties or {}}))

    def remove_edge(self, source: str, target: str, relation_type: str = "RELATED"):
        self._ops.append(ChangeOp(op_type="remove_edge", args={"source": source, "target": target, "relation_type": relation_type}))

    def set_properties(self, node_id: str, properties: Dict[str, Any], previous_properties: Optional[Dict[str, Any]] = None):
        self._ops.append(ChangeOp(op_type="set_properties",
                                  args={"node_id": node_id, "properties": dict(properties or {}), "previous_properties": dict(previous_properties or {})}))

    def add_node(self, node_id: str, properties: Optional[Dict[str, Any]] = None):
        self._ops.append(ChangeOp(op_type="add_node", args={"node_id": node_id, "properties": dict(properties or {})}))

    def remove_node(self, node_id: str, properties: Optional[Dict[str, Any]] = None):
        self._ops.append(ChangeOp(op_type="remove_node", args={"node_id": node_id, "properties": dict(properties or {})}))

    def archive(self, node_id: str):
        self._ops.append(ChangeOp(op_type="archive", args={"node_id": node_id}))

    def unarchive(self, node_id: str):
        self._ops.append(ChangeOp(op_type="unarchive", args={"node_id": node_id}))


class TransactionalVectorProxy:
    """
    Proxy for a vector store to record upserts/deletes as ChangeOps.
    """

    def __init__(self, ops: List[ChangeOp]):
        self._ops = ops

    def upsert(self, item_id: str, embedding: Any, metadata: Dict[str, Any], node_ids: Optional[list] = None, is_global: bool = True):
        self._ops.append(ChangeOp(op_type="vector_upsert", args={
            "item_id": item_id,
            "embedding": embedding,
            "metadata": dict(metadata or {}),
            "node_ids": list(node_ids or []),
            "is_global": bool(is_global),
        }))

    def delete(self, item_id: str):
        self._ops.append(ChangeOp(op_type="vector_delete", args={"item_id": item_id}))


class TransactionalMemoryProxy:
    """
    High-level proxy returning sub-proxies for graph/vector to capture ops during preview.
    Example usage in a plugin:
        ops: List[ChangeOp] = []
        txn = TransactionalMemoryProxy(ops)
        g = txn.graph()
        g.add_edge("a","b","RELATED")  # captured, not applied
        v = txn.vector()
        v.upsert(item_id, embedding, metadata, node_ids)
    """

    def __init__(self, ops: Optional[List[ChangeOp]] = None):
        self._ops = ops if ops is not None else []
        self._graph = TransactionalGraphProxy(self._ops)
        self._vector = TransactionalVectorProxy(self._ops)

    def ops(self) -> List[ChangeOp]:
        return self._ops

    def graph(self) -> TransactionalGraphProxy:
        return self._graph

    def vector(self) -> TransactionalVectorProxy:
        return self._vector
