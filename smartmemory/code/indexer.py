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
        logger.info("Found %d Python files in %s", len(py_files), self.repo_root)

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

        # Write nodes in bulk
        bulk_nodes = [entity.to_properties() for entity in all_entities]
        try:
            result.entities_created = self.graph.add_nodes_bulk(bulk_nodes)
        except Exception as e:
            logger.error("Bulk node write failed for %s: %s", self.repo, e)
            result.errors.append(f"Bulk node write failed: {e}")

        # Write edges in bulk (only if both endpoints exist)
        bulk_edges = [
            (rel.source_id, rel.target_id, rel.relation_type, rel.properties)
            for rel in all_relations
            if rel.source_id in entity_ids and rel.target_id in entity_ids
        ]
        try:
            result.edges_created = self.graph.add_edges_bulk(bulk_edges)
        except Exception as e:
            logger.error("Bulk edge write failed for %s: %s", self.repo, e)
            result.errors.append(f"Bulk edge write failed: {e}")

        result.elapsed_seconds = round(time.time() - start, 2)
        logger.info(
            "Indexed %s: %d files, %d entities, %d edges in %.2fs",
            self.repo,
            result.files_parsed,
            result.entities_created,
            result.edges_created,
            result.elapsed_seconds,
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
            logger.info("Deleted existing code nodes for repo: %s", repo)
        except Exception as e:
            logger.warning("Failed to delete existing code nodes: %s", e)


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
