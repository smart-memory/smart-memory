"""Code indexer — orchestrates parsing and graph writes for a directory."""

import logging
import os
import time
from typing import Any, Optional

from smartmemory.code.models import CodeEntity, CodeRelation, IndexResult
from smartmemory.code.parser import CodeParser, collect_python_files

logger = logging.getLogger(__name__)


class CodeIndexer:
    """Index a Python codebase into the SmartMemory knowledge graph."""

    def __init__(self, graph: Any, repo: str, repo_root: str, exclude_dirs: Optional[set[str]] = None):
        """Initialize the indexer.

        Args:
            graph: SmartGraph instance (with scope provider for multi-tenancy)
            repo: Repository identifier (e.g., "smart-memory-service")
            repo_root: Absolute path to the repository root
            exclude_dirs: Directory names to skip during file collection
        """
        self.graph = graph
        self.repo = repo
        self.repo_root = os.path.abspath(repo_root)
        self.parser = CodeParser(repo=repo, repo_root=repo_root)
        self.exclude_dirs = exclude_dirs

    def index(self) -> IndexResult:
        """Index the entire directory and write to graph."""
        start = time.time()
        result = IndexResult(repo=self.repo)

        # Collect files
        py_files = collect_python_files(self.repo_root, self.exclude_dirs)
        logger.info(f"Found {len(py_files)} Python files in {self.repo_root}")

        # Parse all files
        all_entities: list[CodeEntity] = []
        all_relations: list[CodeRelation] = []

        for file_path in py_files:
            parse_result = self.parser.parse_file(file_path)
            if parse_result.errors and not parse_result.entities:
                result.files_skipped += 1
            else:
                result.files_parsed += 1
            all_entities.extend(parse_result.entities)
            all_relations.extend(parse_result.relations)
            result.errors.extend(parse_result.errors)

        # Build entity ID set for edge validation
        entity_ids = {e.item_id for e in all_entities}

        # Delete existing code nodes for this repo (clean slate)
        self._delete_existing(self.repo)

        # Write nodes
        for entity in all_entities:
            try:
                self.graph.add_node(
                    item_id=entity.item_id,
                    properties=entity.to_properties(),
                    memory_type="code",
                )
                result.entities_created += 1
            except Exception as e:
                result.errors.append(f"Failed to create node {entity.item_id}: {e}")

        # Write edges (only if both endpoints exist)
        for relation in all_relations:
            if relation.source_id not in entity_ids or relation.target_id not in entity_ids:
                continue  # skip dangling edges
            try:
                self.graph.add_edge(
                    source_id=relation.source_id,
                    target_id=relation.target_id,
                    edge_type=relation.relation_type,
                    properties=relation.properties,
                    memory_type="code",
                )
                result.edges_created += 1
            except Exception as e:
                result.errors.append(f"Failed to create edge {relation.relation_type}: {e}")

        result.elapsed_seconds = round(time.time() - start, 2)
        logger.info(
            f"Indexed {self.repo}: {result.files_parsed} files, "
            f"{result.entities_created} entities, {result.edges_created} edges "
            f"in {result.elapsed_seconds}s"
        )
        return result

    def _delete_existing(self, repo: str):
        """Delete all code nodes for a given repo before re-indexing.

        Applies workspace_id scoping from the graph's scope_provider
        to prevent cross-tenant deletion.
        """
        try:
            params: dict[str, Any] = {"repo": repo}
            where_clauses = ["n.repo = $repo"]

            # Inject workspace scope to prevent cross-tenant deletion
            scope_filters = _get_scope_filters(self.graph)
            if scope_filters.get("workspace_id"):
                params["workspace_id"] = scope_filters["workspace_id"]
                where_clauses.append("n.workspace_id = $workspace_id")

            where = " AND ".join(where_clauses)
            query = f"MATCH (n:Code) WHERE {where} DETACH DELETE n"
            self.graph.execute_query(query, params)
            logger.info(f"Deleted existing code nodes for repo: {repo}")
        except Exception as e:
            logger.warning(f"Failed to delete existing code nodes: {e}")


def _get_scope_filters(graph: Any) -> dict[str, Any]:
    """Extract isolation filters from a graph's scope_provider.

    Returns workspace_id (and tenant_id if present) for injecting into
    raw Cypher WHERE clauses. Returns empty dict in OSS/unscoped mode.

    Note: Duplicated in smartmemory_mcp/tools/code_tools.py — keep in sync.
    """
    try:
        backend = getattr(graph, "backend", None)
        if backend is None:
            # SmartGraph wraps backend via nodes manager
            nodes = getattr(graph, "nodes", None)
            backend = getattr(nodes, "backend", None) if nodes else None
        if backend and hasattr(backend, "scope_provider"):
            return backend.scope_provider.get_isolation_filters()
    except Exception:
        pass
    return {}
