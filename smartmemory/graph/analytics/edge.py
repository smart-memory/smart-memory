from typing import Dict

from smartmemory.graph.smartgraph import SmartGraph


def edge_type_breakdown(graph: SmartGraph) -> Dict[str, int]:
    """
    Count the number of each edge type in the graph.
    """
    # Assuming SmartGraph exposes a method to get all edges
    edge_counts = {}
    all_edges = graph.get_all_edges()  # This method should exist on SmartGraph
    for edge in all_edges:
        edge_type = edge.get("type")
        if edge_type:
            edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
    return edge_counts
