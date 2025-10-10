"""
Rich ontological system for SmartMemory with proper node types and semantic relationships.
Replaces the generic Entity models with a proper type hierarchy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.memory_item import MemoryItem


class NodeType(Enum):
    """Enumeration of supported node types in the ontology."""
    PERSON = "person"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    EVENT = "event"
    LOCATION = "location"
    DOCUMENT = "document"
    SKILL = "skill"
    TOOL = "tool"
    PROJECT = "project"
    TASK = "task"
    GOAL = "goal"
    PROCESS = "process"
    ARTIFACT = "artifact"


class RelationType(Enum):
    """Semantic relationship types between nodes."""

    # Professional/Organizational
    WORKS_AT = "works_at"
    MANAGES = "manages"
    REPORTS_TO = "reports_to"
    COLLABORATES_WITH = "collaborates_with"
    MEMBER_OF = "member_of"

    # Spatial/Location
    LIVES_IN = "lives_in"
    LOCATED_IN = "located_in"
    CONTAINS = "contains"
    NEAR = "near"

    # Temporal
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    TRIGGERS = "triggers"

    # Causal
    CAUSES = "causes"
    ENABLES = "enables"
    PREVENTS = "prevents"
    REQUIRES = "requires"
    DEPENDS_ON = "depends_on"

    # Hierarchical
    PART_OF = "part_of"
    BELONGS_TO = "belongs_to"
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"

    # Semantic
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    EXAMPLE_OF = "example_of"
    INSTANCE_OF = "instance_of"
    TYPE_OF = "type_of"

    # Logical
    IMPLIES = "implies"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    REFUTES = "refutes"

    # Knowledge
    KNOWS_ABOUT = "knows_about"
    TEACHES = "teaches"
    LEARNS_FROM = "learns_from"
    USES = "uses"
    CREATES = "creates"

    # Document/Information
    AUTHORED_BY = "authored_by"
    REFERENCES = "references"
    CITES = "cites"
    MENTIONS = "mentions"
    DESCRIBES = "describes"

    # Project/Task
    ASSIGNED_TO = "assigned_to"
    CONTRIBUTES_TO = "contributes_to"
    BLOCKS = "blocks"
    DEPENDS_ON_TASK = "depends_on_task"

    # Generic (use sparingly!)
    RELATED_TO = "related_to"  # Only when no specific relation applies


@dataclass
class OntologyNode(MemoryBaseModel, ABC):
    """Abstract base class for all ontology nodes."""

    item_id: Optional[str] = None
    name: str = ""
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    confidence: float = 1.0
    source: Optional[str] = None  # Where this node came from

    def __post_init__(self):
        # Basic validation similar to Pydantic constraints
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

    @property
    @abstractmethod
    def node_type(self) -> NodeType:
        """Return the specific node type."""
        pass

    @abstractmethod
    def get_searchable_content(self) -> str:
        """Return content that should be indexed for search."""
        pass

    def to_memory_item(self) -> MemoryItem:
        """Convert to MemoryItem for storage."""
        # Include name in metadata - don't exclude it!
        metadata = self.to_dict()
        if 'item_id' in metadata:
            metadata.pop('item_id', None)
        # Handle both enum and string node types
        node_type_value = self.node_type.value if hasattr(self.node_type, 'value') else self.node_type
        metadata['node_type'] = node_type_value

        kwargs = dict(
            content=self.get_searchable_content(),
            type=node_type_value,
            metadata=metadata,
        )
        if self.item_id:
            kwargs["item_id"] = self.item_id
        return MemoryItem(**kwargs)

    @classmethod
    def from_memory_item(cls, item: MemoryItem) -> 'OntologyNode':
        """Create node from MemoryItem."""
        # This will be implemented by subclasses
        raise NotImplementedError("Subclasses must implement from_memory_item")


@dataclass
class Person(OntologyNode):
    """Represents a person/individual."""

    email: Optional[str] = None
    phone: Optional[str] = None
    age: Optional[int] = None
    occupation: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    location: Optional[str] = None

    @property
    def node_type(self) -> NodeType:
        return NodeType.PERSON

    def get_searchable_content(self) -> str:
        content_parts = [self.name]
        if self.occupation:
            content_parts.append(self.occupation)
        if self.skills:
            content_parts.extend(self.skills)
        if self.interests:
            content_parts.extend(self.interests)
        if self.description:
            content_parts.append(self.description)
        return " ".join(content_parts)


@dataclass
class Organization(OntologyNode):
    """Represents an organization/company."""

    industry: Optional[str] = None
    size: Optional[int] = None
    founded: Optional[datetime] = None
    headquarters: Optional[str] = None
    website: Optional[str] = None
    type: Optional[str] = None  # company, nonprofit, government, etc.

    @property
    def node_type(self) -> NodeType:
        return NodeType.ORGANIZATION

    def get_searchable_content(self) -> str:
        content_parts = [self.name]
        if self.industry:
            content_parts.append(self.industry)
        if self.type:
            content_parts.append(self.type)
        if self.description:
            content_parts.append(self.description)
        return " ".join(content_parts)


@dataclass
class Concept(OntologyNode):
    """Represents an abstract concept or idea."""

    domain: Optional[str] = None  # field/domain this concept belongs to
    definition: Optional[str] = None
    complexity: Optional[str] = None  # basic, intermediate, advanced
    prerequisites: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)

    @property
    def node_type(self) -> NodeType:
        return NodeType.CONCEPT

    def get_searchable_content(self) -> str:
        content_parts = [self.name]
        if self.domain:
            content_parts.append(self.domain)
        if self.definition:
            content_parts.append(self.definition)
        if self.description:
            content_parts.append(self.description)
        return " ".join(content_parts)


@dataclass
class Event(OntologyNode):
    """Represents an event or occurrence."""

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    location: Optional[str] = None
    participants: List[str] = field(default_factory=list)
    outcome: Optional[str] = None
    event_type: Optional[str] = None  # meeting, conference, incident, etc.

    @property
    def node_type(self) -> NodeType:
        return NodeType.EVENT

    def get_searchable_content(self) -> str:
        content_parts = [self.name]
        if self.event_type:
            content_parts.append(self.event_type)
        if self.participants:
            content_parts.extend(self.participants)
        if self.outcome:
            content_parts.append(self.outcome)
        if self.description:
            content_parts.append(self.description)
        return " ".join(content_parts)


@dataclass
class Location(OntologyNode):
    """Represents a physical or virtual location."""

    address: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None  # lat, lng
    location_type: Optional[str] = None  # city, building, room, etc.
    capacity: Optional[int] = None
    timezone: Optional[str] = None

    @property
    def node_type(self) -> NodeType:
        return NodeType.LOCATION

    def get_searchable_content(self) -> str:
        content_parts = [self.name]
        if self.address:
            content_parts.append(self.address)
        if self.location_type:
            content_parts.append(self.location_type)
        if self.description:
            content_parts.append(self.description)
        return " ".join(content_parts)


@dataclass
class Document(OntologyNode):
    """Represents a document or information artifact."""

    author: Optional[str] = None
    document_type: Optional[str] = None  # paper, report, email, etc.
    format: Optional[str] = None  # pdf, docx, html, etc.
    url: Optional[str] = None
    file_path: Optional[str] = None
    word_count: Optional[int] = None
    language: Optional[str] = None

    @property
    def node_type(self) -> NodeType:
        return NodeType.DOCUMENT

    def get_searchable_content(self) -> str:
        content_parts = [self.name]
        if self.author:
            content_parts.append(self.author)
        if self.document_type:
            content_parts.append(self.document_type)
        if self.description:
            content_parts.append(self.description)
        return " ".join(content_parts)


@dataclass
class Skill(OntologyNode):
    """Represents a skill or capability."""

    category: Optional[str] = None  # technical, soft, domain-specific
    proficiency_level: Optional[str] = None  # beginner, intermediate, expert
    prerequisites: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)

    @property
    def node_type(self) -> NodeType:
        return NodeType.SKILL

    def get_searchable_content(self) -> str:
        content_parts = [self.name]
        if self.category:
            content_parts.append(self.category)
        if self.applications:
            content_parts.extend(self.applications)
        if self.description:
            content_parts.append(self.description)
        return " ".join(content_parts)


@dataclass
class Tool(OntologyNode):
    """Represents a tool, technology, or system."""

    category: Optional[str] = None  # software, hardware, framework, etc.
    version: Optional[str] = None
    vendor: Optional[str] = None
    use_cases: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)

    @property
    def node_type(self) -> NodeType:
        return NodeType.TOOL

    def get_searchable_content(self) -> str:
        content_parts = [self.name]
        if self.category:
            content_parts.append(self.category)
        if self.vendor:
            content_parts.append(self.vendor)
        if self.use_cases:
            content_parts.extend(self.use_cases)
        if self.description:
            content_parts.append(self.description)
        return " ".join(content_parts)


# Node type registry for dynamic creation
NODE_TYPE_REGISTRY = {
    NodeType.PERSON: Person,
    NodeType.ORGANIZATION: Organization,
    NodeType.CONCEPT: Concept,
    NodeType.EVENT: Event,
    NodeType.LOCATION: Location,
    NodeType.DOCUMENT: Document,
    NodeType.SKILL: Skill,
    NodeType.TOOL: Tool,
}


def create_node(node_type: Union[NodeType, str], **kwargs) -> OntologyNode:
    """Factory function to create nodes of the appropriate type."""
    if isinstance(node_type, str):
        node_type = NodeType(node_type)

    node_class = NODE_TYPE_REGISTRY.get(node_type)
    if not node_class:
        raise ValueError(f"Unknown node type: {node_type}")

    return node_class(**kwargs)


def get_valid_relations(source_type: NodeType, target_type: NodeType) -> List[RelationType]:
    """Get valid relationship types between two node types."""

    # Define valid relationship patterns
    valid_patterns = {
        # Person relationships
        (NodeType.PERSON, NodeType.ORGANIZATION): [
            RelationType.WORKS_AT, RelationType.MEMBER_OF, RelationType.MANAGES
        ],
        (NodeType.PERSON, NodeType.PERSON): [
            RelationType.MANAGES, RelationType.REPORTS_TO, RelationType.COLLABORATES_WITH
        ],
        (NodeType.PERSON, NodeType.LOCATION): [
            RelationType.LIVES_IN, RelationType.LOCATED_IN
        ],
        (NodeType.PERSON, NodeType.SKILL): [
            RelationType.KNOWS_ABOUT, RelationType.USES
        ],
        (NodeType.PERSON, NodeType.CONCEPT): [
            RelationType.KNOWS_ABOUT, RelationType.TEACHES, RelationType.LEARNS_FROM
        ],

        # Organization relationships
        (NodeType.ORGANIZATION, NodeType.LOCATION): [
            RelationType.LOCATED_IN, RelationType.CONTAINS
        ],
        (NodeType.ORGANIZATION, NodeType.TOOL): [
            RelationType.USES, RelationType.CREATES
        ],

        # Document relationships
        (NodeType.DOCUMENT, NodeType.PERSON): [
            RelationType.AUTHORED_BY
        ],
        (NodeType.DOCUMENT, NodeType.CONCEPT): [
            RelationType.DESCRIBES, RelationType.REFERENCES
        ],

        # Event relationships
        (NodeType.EVENT, NodeType.PERSON): [
            RelationType.ASSIGNED_TO
        ],
        (NodeType.EVENT, NodeType.LOCATION): [
            RelationType.LOCATED_IN
        ],

        # Concept relationships
        (NodeType.CONCEPT, NodeType.CONCEPT): [
            RelationType.SIMILAR_TO, RelationType.OPPOSITE_OF, RelationType.PART_OF,
            RelationType.REQUIRES, RelationType.ENABLES
        ],
    }

    # Get specific patterns or fall back to generic relations
    specific_relations = valid_patterns.get((source_type, target_type), [])

    # Always allow generic relations as fallback
    generic_relations = [
        RelationType.RELATED_TO, RelationType.REFERENCES, RelationType.MENTIONS
    ]

    return specific_relations + generic_relations
