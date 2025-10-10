import logging

from smartmemory.graph.smartgraph import SmartGraph

logger = logging.getLogger(__name__)


def prune_by_age(graph: SmartGraph, node_label: str, reference_time_field: str, max_age_days: int, dry_run: bool = True) -> dict:
    """
    Prune nodes by age. Removes from graph if not dry_run.
    """
    from datetime import datetime, timezone, timedelta
    candidates = set()
    now = datetime.now(timezone.utc)
    nodes = graph.search_nodes(node_label)
    cutoff = now - timedelta(days=max_age_days)
    for node in nodes:
        ref_time = node.get(reference_time_field)
        if ref_time:
            if isinstance(ref_time, str):
                try:
                    ref_time = datetime.fromisoformat(ref_time)
                except Exception:
                    continue
            if ref_time < cutoff:
                candidates.add(node["item_id"])
    removed = []
    if not dry_run and candidates:
        for item_id in candidates:
            graph.remove_node(item_id)
            removed.append(item_id)
    return {"candidates": list(candidates), "removed": removed, "dry_run": dry_run}


def prune_by_degree(graph: SmartGraph, node_label: str, min_degree: int, dry_run: bool = True) -> dict:
    """
    Prune nodes by min degree. Removes from graph if not dry_run.
    """
    candidates = set()
    nodes = graph.search_nodes(node_label)
    for node in nodes:
        item_id = node.get("item_id")
        out_edges = graph.get_outgoing_edges(item_id)
        in_edges = graph.get_incoming_edges(item_id)
        if len(out_edges) + len(in_edges) <= min_degree:
            candidates.add(item_id)
    removed = []
    if not dry_run and candidates:
        for item_id in candidates:
            graph.remove_node(item_id)
            removed.append(item_id)
    return {"candidates": list(candidates), "removed": removed, "dry_run": dry_run}
