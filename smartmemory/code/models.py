"""Data models for code entities and relationships.

Lightweight dataclasses for code parsing results. These do NOT extend
MemoryBaseModel because they are transient parsing artifacts that never
cross the API boundary or get serialized to MongoDB.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodeEntity:
    """A code entity extracted from a Python source file."""

    name: str
    entity_type: str  # module | class | function | route | test
    file_path: str  # relative to repo root
    line_number: int
    repo: str
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    http_method: str = ""
    http_path: str = ""

    @property
    def item_id(self) -> str:
        """Deterministic ID for graph node."""
        return f"code::{self.repo}::{self.file_path}::{self.name}"

    def to_properties(self) -> dict[str, Any]:
        """Convert to FalkorDB node properties."""
        props: dict[str, Any] = {
            "item_id": self.item_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "repo": self.repo,
            "memory_type": "code",
        }
        if self.docstring:
            props["docstring"] = self.docstring[:500]  # truncate long docstrings
        if self.decorators:
            props["decorators"] = ",".join(self.decorators)
        if self.bases:
            props["bases"] = ",".join(self.bases)
        if self.http_method:
            props["http_method"] = self.http_method
        if self.http_path:
            props["http_path"] = self.http_path
        return props


@dataclass
class CodeRelation:
    """A relationship between two code entities."""

    source_id: str
    target_id: str
    relation_type: str  # DEFINES | IMPORTS | CALLS | INHERITS | TESTS
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    """Result of parsing a single Python file."""

    file_path: str
    entities: list[CodeEntity] = field(default_factory=list)
    relations: list[CodeRelation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class IndexResult:
    """Summary of indexing an entire directory."""

    repo: str
    files_parsed: int = 0
    files_skipped: int = 0
    entities_created: int = 0
    edges_created: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
