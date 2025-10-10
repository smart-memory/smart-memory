from typing import Optional, Callable, Any, Dict, Set, List


class EvolutionNode:
    """
    Represents a single evolver step in the DAG.
    - evolver_path: dotted import path to an evolver class with evolve(smart_memory, ...)
    - condition: optional callable (smart_memory, context) -> bool
    - params: optional dict passed to evolve as **kwargs
    """

    def __init__(self, node_id: str, evolver_path: str, condition: Optional[Callable[[Any, Optional[Dict[str, Any]]], bool]] = None, params: Optional[Dict[str, Any]] = None):
        self.id = node_id
        self.evolver_path = evolver_path
        self.condition = condition
        self.params = params or {}


class EvolutionFlow:
    """In-memory DAG used to execute evolution workflows dynamically."""

    def __init__(self):
        self.nodes: Dict[str, EvolutionNode] = {}
        self.edges_out: Dict[str, Set[str]] = {}
        self.indegree: Dict[str, int] = {}

    def add_node(self, node: EvolutionNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node id: {node.id}")
        self.nodes[node.id] = node
        self.edges_out.setdefault(node.id, set())
        self.indegree.setdefault(node.id, 0)

    def add_edge(self, src_id: str, dst_id: str) -> None:
        if src_id not in self.nodes or dst_id not in self.nodes:
            raise KeyError("Both src and dst must be added as nodes before adding an edge")
        if dst_id not in self.edges_out[src_id]:
            self.edges_out[src_id].add(dst_id)
            self.indegree[dst_id] = self.indegree.get(dst_id, 0) + 1

    def validate_acyclic(self) -> None:
        # Kahn's algorithm to detect cycles
        indeg = dict(self.indegree)
        queue = [n for n in self.nodes if indeg.get(n, 0) == 0]
        seen = 0
        idx = 0
        # Use a simple list as queue to keep deterministic order
        while idx < len(queue):
            nid = queue[idx]
            idx += 1
            seen += 1
            for m in self.edges_out.get(nid, set()):
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        if seen != len(self.nodes):
            raise ValueError("EvolutionFlow contains a cycle")

    def topological_order(self) -> List[str]:
        # Return topological order using Kahn's algorithm (does not mutate original indegree)
        indeg = dict(self.indegree)
        order: List[str] = []
        queue = [n for n in self.nodes if indeg.get(n, 0) == 0]
        idx = 0
        while idx < len(queue):
            nid = queue[idx]
            idx += 1
            order.append(nid)
            for m in self.edges_out.get(nid, set()):
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        if len(order) != len(self.nodes):
            raise ValueError("EvolutionFlow contains a cycle; cannot order")
        return order
