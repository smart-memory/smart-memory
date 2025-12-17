"""
Opinion and Observation Models for Synthesis Memory 

Opinions are beliefs with confidence scores, formed from pattern recognition
across episodic memories. They can be reinforced or contradicted over time.

Observations are synthesized entity summaries - coherent descriptions of
entities based on multiple facts scattered across memories.


"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Literal

from smartmemory.models.base import MemoryBaseModel


# Disposition traits for preference-aware synthesis
DispositionTrait = Literal['skepticism', 'literalism', 'empathy']


@dataclass
class Disposition(MemoryBaseModel):
    """
    Disposition parameters that influence opinion formation.
    
    
    - skepticism: How much evidence is required before forming an opinion (0-1)
    - literalism: How literally to interpret statements (0-1)
    - empathy: How much to weight emotional/preference signals (0-1)
    """
    skepticism: float = 0.5  # Higher = more evidence needed
    literalism: float = 0.5  # Higher = more literal interpretation
    empathy: float = 0.5     # Higher = more weight on preferences
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'skepticism': self.skepticism,
            'literalism': self.literalism,
            'empathy': self.empathy,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Disposition':
        return cls(
            skepticism=data.get('skepticism', 0.5),
            literalism=data.get('literalism', 0.5),
            empathy=data.get('empathy', 0.5),
        )


@dataclass
class OpinionMetadata(MemoryBaseModel):
    """
    Metadata specific to Opinion memory type.
    
    Opinions are beliefs with confidence scores that can be
    reinforced or contradicted over time.
    """
    confidence: float  # 0.0-1.0, required for opinions
    formed_from: List[str] = field(default_factory=list)  # item_ids that contributed
    reinforcement_count: int = 0  # times supporting evidence seen
    contradiction_count: int = 0  # times contradicting evidence seen
    
    # When this opinion was formed/updated
    formed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_reinforced_at: Optional[datetime] = None
    last_contradicted_at: Optional[datetime] = None
    
    # Disposition used when forming this opinion
    disposition: Optional[Disposition] = None
    
    # Topic/entity this opinion is about
    subject: Optional[str] = None  # e.g., "user preferences", "Alice", "Python"
    subject_type: Optional[str] = None  # e.g., "preference", "person", "technology"
    
    @property
    def net_reinforcement(self) -> int:
        """Net reinforcement score (positive = supported, negative = contradicted)."""
        return self.reinforcement_count - self.contradiction_count
    
    @property
    def stability(self) -> float:
        """How stable this opinion is (0-1). Higher = more stable."""
        total = self.reinforcement_count + self.contradiction_count
        if total == 0:
            return 0.5  # Neutral stability for new opinions
        return self.reinforcement_count / total
    
    def reinforce(self, evidence_id: str):
        """Record supporting evidence."""
        self.reinforcement_count += 1
        self.last_reinforced_at = datetime.now(timezone.utc)
        if evidence_id not in self.formed_from:
            self.formed_from.append(evidence_id)
        # Increase confidence (with diminishing returns)
        self.confidence = min(1.0, self.confidence + (1 - self.confidence) * 0.1)
    
    def contradict(self, evidence_id: str):
        """Record contradicting evidence."""
        self.contradiction_count += 1
        self.last_contradicted_at = datetime.now(timezone.utc)
        if evidence_id not in self.formed_from:
            self.formed_from.append(evidence_id)
        # Decrease confidence
        self.confidence = max(0.0, self.confidence - self.confidence * 0.15)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'confidence': self.confidence,
            'formed_from': self.formed_from,
            'reinforcement_count': self.reinforcement_count,
            'contradiction_count': self.contradiction_count,
            'formed_at': self.formed_at.isoformat() if self.formed_at else None,
            'last_reinforced_at': self.last_reinforced_at.isoformat() if self.last_reinforced_at else None,
            'last_contradicted_at': self.last_contradicted_at.isoformat() if self.last_contradicted_at else None,
            'disposition': self.disposition.to_dict() if self.disposition else None,
            'subject': self.subject,
            'subject_type': self.subject_type,
            'net_reinforcement': self.net_reinforcement,
            'stability': self.stability,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OpinionMetadata':
        formed_at = data.get('formed_at')
        if isinstance(formed_at, str):
            formed_at = datetime.fromisoformat(formed_at)
        elif formed_at is None:
            formed_at = datetime.now(timezone.utc)
            
        last_reinforced = data.get('last_reinforced_at')
        if isinstance(last_reinforced, str):
            last_reinforced = datetime.fromisoformat(last_reinforced)
            
        last_contradicted = data.get('last_contradicted_at')
        if isinstance(last_contradicted, str):
            last_contradicted = datetime.fromisoformat(last_contradicted)
        
        disposition = None
        if data.get('disposition'):
            disposition = Disposition.from_dict(data['disposition'])
        
        return cls(
            confidence=data.get('confidence', 0.5),
            formed_from=data.get('formed_from', []),
            reinforcement_count=data.get('reinforcement_count', 0),
            contradiction_count=data.get('contradiction_count', 0),
            formed_at=formed_at,
            last_reinforced_at=last_reinforced,
            last_contradicted_at=last_contradicted,
            disposition=disposition,
            subject=data.get('subject'),
            subject_type=data.get('subject_type'),
        )


@dataclass
class ObservationMetadata(MemoryBaseModel):
    """
    Metadata specific to Observation memory type.
    
    Observations are synthesized entity summaries - coherent descriptions
    built from multiple facts about an entity.
    """
    entity_id: str  # ID of the entity this observation is about
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None  # person, organization, concept, etc.
    
    # Source facts that contributed to this observation
    source_facts: List[str] = field(default_factory=list)  # item_ids
    
    # When this observation was synthesized/updated
    synthesized_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated_at: Optional[datetime] = None
    
    # Completeness score (0-1) - how much we know about this entity
    completeness: float = 0.0
    
    # Aspects covered in this observation
    aspects_covered: List[str] = field(default_factory=list)  # e.g., ["career", "education", "preferences"]
    
    def add_source(self, fact_id: str, aspect: Optional[str] = None):
        """Add a source fact to this observation."""
        if fact_id not in self.source_facts:
            self.source_facts.append(fact_id)
        if aspect and aspect not in self.aspects_covered:
            self.aspects_covered.append(aspect)
        self.last_updated_at = datetime.now(timezone.utc)
        # Update completeness based on aspects
        self.completeness = min(1.0, len(self.aspects_covered) * 0.2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'entity_name': self.entity_name,
            'entity_type': self.entity_type,
            'source_facts': self.source_facts,
            'synthesized_at': self.synthesized_at.isoformat() if self.synthesized_at else None,
            'last_updated_at': self.last_updated_at.isoformat() if self.last_updated_at else None,
            'completeness': self.completeness,
            'aspects_covered': self.aspects_covered,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ObservationMetadata':
        synthesized_at = data.get('synthesized_at')
        if isinstance(synthesized_at, str):
            synthesized_at = datetime.fromisoformat(synthesized_at)
        elif synthesized_at is None:
            synthesized_at = datetime.now(timezone.utc)
            
        last_updated = data.get('last_updated_at')
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        
        return cls(
            entity_id=data.get('entity_id', ''),
            entity_name=data.get('entity_name'),
            entity_type=data.get('entity_type'),
            source_facts=data.get('source_facts', []),
            synthesized_at=synthesized_at,
            last_updated_at=last_updated,
            completeness=data.get('completeness', 0.0),
            aspects_covered=data.get('aspects_covered', []),
        )
