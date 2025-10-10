import logging
from typing import Optional

from smartmemory.graph.smartgraph import SmartGraph

logger = logging.getLogger(__name__)


def export_graph(graph: SmartGraph, node_label: Optional[str] = None) -> dict:
    """
    Export the graph or a subset as nodes and edges.
    """
    try:
        nodes = graph.search_nodes(node_label) if node_label else graph.get_all_nodes()
        edges = []
        for node in nodes:
            item_id = node.get("item_id")
            outgoing = graph.get_outgoing_edges(item_id)
            for edge in outgoing:
                edges.append({
                    "source": item_id,
                    "target": edge.get("target"),
                    "type": edge.get("type")
                })
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        logger.error(f"Failed to export graph: {e}")
        return {"nodes": [], "edges": []}
