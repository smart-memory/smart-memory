"""
Ontology IR (Intermediate Representation) Models

LLM-native ontology data structures following the unified plan.
These models represent the authoritative ontology format used across services.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from smartmemory.models.base import MemoryBaseModel, StageRequest


class Status(str, Enum):
    """Status of ontology elements"""
    APPROVED = "approved"
    PROPOSED = "proposed"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"


class Origin(str, Enum):
    """Origin of ontology elements"""
    AI = "ai"
    IMPORT = "import"
    MANUAL = "manual"


@dataclass
class Evidence(MemoryBaseModel):
    """Evidence supporting an ontology element"""
    doc: str = ""
    span: Optional[List[int]] = None  # [start, end] character positions
    quote: str = ""


@dataclass
class Mapping(MemoryBaseModel):
    """External ontology mapping"""
    curie: str = ""  # e.g., "DOID:4"
    source: str = ""  # e.g., "DOID"
    confidence: float = 0.0
    locked: bool = False


@dataclass
class Meta(MemoryBaseModel):
    """Metadata for ontology elements"""
    created_by: str = ""
    created_at: datetime = None
    updated_at: Optional[datetime] = None


@dataclass
class Concept(MemoryBaseModel):
    """Ontology concept definition"""
    id: str = ""  # e.g., "EX:Disease"
    label: str = ""
    synonyms: List[str] = field(default_factory=list)
    status: Status = Status.PROPOSED
    origin: Origin = Origin.AI
    pinned: bool = False
    evidence: List[Evidence] = field(default_factory=list)
    confidence: float = 0.0
    mapped_to: List[Mapping] = field(default_factory=list)
    meta: Optional[Meta] = None


@dataclass
class TaxonomyRelation(MemoryBaseModel):
    """Taxonomic (is-a) relationship"""
    parent: str = ""  # concept ID
    child: str = ""  # concept ID
    status: Status = Status.PROPOSED
    origin: Origin = Origin.AI
    confidence: float = 0.0
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class Signature(MemoryBaseModel):
    """Relation signature with usage statistics"""
    sub: str = ""  # subject concept ID
    obj: str = ""  # object concept ID
    support: int = 0  # number of instances


@dataclass
class Relation(MemoryBaseModel):
    """Ontology relation definition"""
    id: str = ""  # e.g., "EX:treats"
    label: str = ""
    aliases: List[str] = field(default_factory=list)
    domain: str = ""  # concept ID
    range: str = ""  # concept ID
    signatures: List[Signature] = field(default_factory=list)
    status: Status = Status.PROPOSED
    confidence: float = 0.0
    mapped_to: List[Mapping] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class Attribute(MemoryBaseModel):
    """Concept attribute definition"""
    domain: str = ""  # concept ID
    name: str = ""
    datatype: str = ""  # "string|number|date|age|enum"
    examples: List[str] = field(default_factory=list)
    status: Status = Status.PROPOSED
    confidence: float = 0.0
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class Constraint(MemoryBaseModel):
    """Ontology constraint"""
    type: str = ""  # "domain_range|cardinality|disjointness"
    predicate: str = ""
    domain: str = ""
    range: str = ""
    kind: str = "soft"  # "soft|hard"
    confidence: float = 0.0


@dataclass
class DiffSection(MemoryBaseModel):
    """Diff section for audit trail"""
    concepts: List[str] = field(default_factory=list)
    taxonomy: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)


@dataclass
class Diff(MemoryBaseModel):
    """Changes between ontology versions"""
    added: DiffSection = field(default_factory=DiffSection)
    removed: DiffSection = field(default_factory=DiffSection)
    changed: List[str] = field(default_factory=list)


@dataclass
class Audit(MemoryBaseModel):
    """Audit information for ontology snapshot"""
    version: str = ""
    previous: Optional[str] = None
    diff: Optional[Diff] = None
    counts: Dict[str, int] = field(default_factory=dict)
    checksum: str = ""


@dataclass
class ModelMeta(MemoryBaseModel):
    """LLM models metadata"""
    llm: str = ""
    temp: float = 0.0
    prompt_v: str = ""


@dataclass
class OntologyIR(MemoryBaseModel):
    """Complete Ontology Intermediate Representation"""
    ir_version: str = "1.0"
    registry_id: str = "default"
    built_at: datetime = field(default_factory=datetime.now)
    model_meta: Optional[ModelMeta] = None

    concepts: List[Concept] = field(default_factory=list)
    taxonomy: List[TaxonomyRelation] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    attributes: List[Attribute] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)

    audit: Optional[Audit] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""

        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                data[key] = [
                    {k: convert_datetime(v) for k, v in item.__dict__.items()}
                    if hasattr(item, '__dict__') else item
                    for item in value
                ]
            elif hasattr(value, '__dict__'):
                data[key] = {k: convert_datetime(v) for k, v in value.__dict__.items()}
            else:
                data[key] = convert_datetime(value)
        return data

    def compute_checksum(self) -> str:
        """Compute SHA256 checksum of ontology content"""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get_approved_concepts(self, min_confidence: float = 0.0) -> List[Concept]:
        """Get approved concepts above confidence threshold"""
        return [
            c for c in self.concepts
            if c.status == Status.APPROVED and c.confidence >= min_confidence
        ]

    def get_approved_relations(self, min_confidence: float = 0.0) -> List[Relation]:
        """Get approved relations above confidence threshold"""
        return [
            r for r in self.relations
            if r.status == Status.APPROVED and r.confidence >= min_confidence
        ]

    def validate_references(self) -> List[str]:
        """Validate that all concept references exist"""
        errors = []
        concept_ids = {c.id for c in self.concepts}

        # Check taxonomy references
        for tax in self.taxonomy:
            if tax.parent not in concept_ids:
                errors.append(f"Taxonomy parent '{tax.parent}' not found in concepts")
            if tax.child not in concept_ids:
                errors.append(f"Taxonomy child '{tax.child}' not found in concepts")

        # Check relation domain/range
        for rel in self.relations:
            if rel.domain and rel.domain not in concept_ids:
                errors.append(f"Relation '{rel.id}' domain '{rel.domain}' not found in concepts")
            if rel.range and rel.range not in concept_ids:
                errors.append(f"Relation '{rel.id}' range '{rel.range}' not found in concepts")

        # Check attribute domains
        for attr in self.attributes:
            if attr.domain not in concept_ids:
                errors.append(f"Attribute '{attr.name}' domain '{attr.domain}' not found in concepts")

        return errors


@dataclass
class InferenceRequest(StageRequest):
    """Request for LLM ontology inference"""
    registry_id: str = "default"
    raw_chunks: List[Dict[str, str]] = field(default_factory=list)  # [{"doc_id": "d1", "text": "..."}]
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse(MemoryBaseModel):
    """Response from LLM ontology inference"""
    changeset: OntologyIR = None
    metrics: Dict[str, int] = field(default_factory=dict)
    status: str = "completed"


@dataclass
class ApplyRequest(StageRequest):
    """Request to apply changeset to registry"""
    base_version: str = ""
    changeset: OntologyIR = None


@dataclass
class ApplyResponse(MemoryBaseModel):
    """Response from applying changeset"""
    version: str = ""
    diff: Diff = None
    counts: Dict[str, int] = field(default_factory=dict)
