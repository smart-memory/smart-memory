import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Set, List, Any, Optional


@dataclass
class EntityTypeDefinition:
    """Definition of an entity type in the ontology."""
    name: str
    description: str
    properties: Dict[str, str]  # property_name -> property_type
    required_properties: Set[str]
    parent_types: Set[str]  # inheritance hierarchy
    aliases: Set[str]  # alternative names
    examples: List[str]  # example entities of this type
    created_by: str  # "human", "llm", "inferred"
    created_at: datetime
    confidence: float = 1.0


@dataclass
class RelationshipTypeDefinition:
    """Definition of a relationship type in the ontology."""
    name: str
    description: str
    source_types: Set[str]  # allowed source entity types
    target_types: Set[str]  # allowed target entity types
    properties: Dict[str, str]  # relationship properties
    bidirectional: bool = False
    aliases: Set[str] = None
    examples: List[Dict[str, str]] = None  # example relationships
    created_by: str = "human"
    created_at: datetime = None
    confidence: float = 1.0

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = set()
        if self.examples is None:
            self.examples = []
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class OntologyRule:
    """Validation or enrichment rule for the ontology."""
    id: str
    name: str
    description: str
    rule_type: str  # "validation", "enrichment", "inference"
    conditions: Dict[str, Any]  # rule conditions
    actions: Dict[str, Any]  # rule actions
    enabled: bool = True
    created_by: str = "human"
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class Ontology:
    """Complete ontology definition with entities, relationships, and rules."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.version = version
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        self.entity_types: Dict[str, EntityTypeDefinition] = {}
        self.relationship_types: Dict[str, RelationshipTypeDefinition] = {}
        self.rules: Dict[str, OntologyRule] = {}

        # Metadata
        self.description = ""
        self.domain = ""  # e.g., "general", "medical", "legal", "technical"
        self.language = "en"
        self.created_by = "system"

    def add_entity_type(self, entity_type: EntityTypeDefinition) -> None:
        """Add an entity type to the ontology."""
        self.entity_types[entity_type.name.lower()] = entity_type
        self.updated_at = datetime.now()

    def add_relationship_type(self, rel_type: RelationshipTypeDefinition) -> None:
        """Add a relationship type to the ontology."""
        self.relationship_types[rel_type.name.lower()] = rel_type
        self.updated_at = datetime.now()

    def add_rule(self, rule: OntologyRule) -> None:
        """Add a rule to the ontology."""
        self.rules[rule.id] = rule
        self.updated_at = datetime.now()

    def get_entity_type(self, name: str) -> Optional[EntityTypeDefinition]:
        """Get entity type by name (case-insensitive)."""
        return self.entity_types.get(name.lower())

    def get_relationship_type(self, name: str) -> Optional[RelationshipTypeDefinition]:
        """Get relationship type by name (case-insensitive)."""
        return self.relationship_types.get(name.lower())

    def validate_entity(self, entity_type: str, properties: Dict[str, Any]) -> List[str]:
        """Validate an entity against the ontology. Returns list of validation errors."""
        errors = []

        entity_def = self.get_entity_type(entity_type)
        if not entity_def:
            return [f"Unknown entity type: {entity_type}"]

        # Check required properties
        for req_prop in entity_def.required_properties:
            if req_prop not in properties:
                errors.append(f"Missing required property '{req_prop}' for entity type '{entity_type}'")

        # Check property types (basic validation)
        for prop_name, prop_value in properties.items():
            if prop_name in entity_def.properties:
                expected_type = entity_def.properties[prop_name]
                # Add type validation logic here if needed

        return errors

    def validate_relationship(self, rel_type: str, source_type: str, target_type: str) -> List[str]:
        """Validate a relationship against the ontology. Returns list of validation errors."""
        errors = []

        rel_def = self.get_relationship_type(rel_type)
        if not rel_def:
            return [f"Unknown relationship type: {rel_type}"]

        # Check source type constraints
        if rel_def.source_types and source_type.lower() not in {t.lower() for t in rel_def.source_types}:
            errors.append(f"Invalid source type '{source_type}' for relationship '{rel_type}'. Allowed: {rel_def.source_types}")

        # Check target type constraints
        if rel_def.target_types and target_type.lower() not in {t.lower() for t in rel_def.target_types}:
            errors.append(f"Invalid target type '{target_type}' for relationship '{rel_type}'. Allowed: {rel_def.target_types}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ontology to dictionary."""
        # Custom serialization to handle sets properly
        entity_types_dict = {}
        for name, entity_type in self.entity_types.items():
            entity_dict = asdict(entity_type)
            entity_dict['required_properties'] = list(entity_type.required_properties)
            entity_dict['parent_types'] = list(entity_type.parent_types)
            entity_dict['aliases'] = list(entity_type.aliases)
            entity_dict['created_at'] = entity_type.created_at.isoformat()
            entity_types_dict[name] = entity_dict

        relationship_types_dict = {}
        for name, rel_type in self.relationship_types.items():
            rel_dict = asdict(rel_type)
            rel_dict['source_types'] = list(rel_type.source_types)
            rel_dict['target_types'] = list(rel_type.target_types)
            rel_dict['aliases'] = list(rel_type.aliases)
            rel_dict['created_at'] = rel_type.created_at.isoformat()
            relationship_types_dict[name] = rel_dict

        rules_dict = {}
        for rule_id, rule in self.rules.items():
            rule_dict = asdict(rule)
            rule_dict['created_at'] = rule.created_at.isoformat()
            rules_dict[rule_id] = rule_dict

        return {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'description': self.description,
            'domain': self.domain,
            'language': self.language,
            'created_by': self.created_by,
            'entity_types': entity_types_dict,
            'relationship_types': relationship_types_dict,
            'rules': rules_dict
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Ontology':
        """Deserialize ontology from dictionary."""
        ontology = cls(data['name'], data['version'])
        ontology.id = data['id']
        ontology.created_at = datetime.fromisoformat(data['created_at'])
        ontology.updated_at = datetime.fromisoformat(data['updated_at'])
        ontology.description = data.get('description', '')
        ontology.domain = data.get('domain', '')
        ontology.language = data.get('language', 'en')
        ontology.created_by = data.get('created_by', 'system')

        # Load entity types
        for name, entity_data in data.get('entity_types') or {}.items():
            entity_data['created_at'] = datetime.fromisoformat(entity_data['created_at'])
            entity_data['required_properties'] = set(entity_data['required_properties'])
            entity_data['parent_types'] = set(entity_data['parent_types'])
            entity_data['aliases'] = set(entity_data['aliases'])
            entity_type = EntityTypeDefinition(**entity_data)
            ontology.entity_types[name] = entity_type

        # Load relationship types
        for name, rel_data in data.get('relationship_types') or {}.items():
            rel_data['created_at'] = datetime.fromisoformat(rel_data['created_at'])
            rel_data['source_types'] = set(rel_data.get('source_types', []))
            rel_data['target_types'] = set(rel_data.get('target_types', []))
            rel_data['aliases'] = set(rel_data.get('aliases', []))
            rel_type = RelationshipTypeDefinition(**rel_data)
            ontology.relationship_types[name] = rel_type

        # Load rules
        for rule_id, rule_data in data.get('rules') or {}.items():
            rule_data['created_at'] = datetime.fromisoformat(rule_data['created_at'])
            rule = OntologyRule(**rule_data)
            ontology.rules[rule_id] = rule

        return ontology
