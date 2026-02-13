"""Code indexer — orchestrates parsing and graph writes for a directory."""

import logging
import os
import time
from typing import Optional

from smartmemory.code.models import CodeEntity, CodeRelation, IndexResult
from smartmemory.code.parser import CodeParser, collect_python_files

logger = logging.getLogger(__name__)


class CodeIndexer:
    """Index a Python codebase into the SmartMemory knowledge graph."""

    def __init__(self, graph, repo: str, repo_root: str, exclude_dirs: Optional[set[str]] = None):
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
            all_entities.extend(parse_result.entities)
            all_relations.extend(parse_result.relations)
            result.files_parsed += 1
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
        """Delete all code nodes for a given repo before re-indexing."""
        try:
            query = "MATCH (n:Code {repo: $repo}) DETACH DELETE n"
            self.graph.execute_query(query, {"repo": repo})
            logger.info(f"Deleted existing code nodes for repo: {repo}")
        except Exception as e:
            logger.warning(f"Failed to delete existing code nodes: {e}")
