"""
Simple Boundary Conversion

Enforces the conversion rule without heavy abstraction.
Rule: Always use .to_dict() for AgenticBaseModel → Pydantic, .dict() for Pydantic → AgenticBaseModel
"""

from typing import TypeVar, Type, List

from smartmemory.models.base import MemoryBaseModel

T_Agentic = TypeVar('T_Agentic', bound=MemoryBaseModel)
T_Pydantic = TypeVar('T_Pydantic')


def to_persistence(agentic_model: T_Agentic, pydantic_class: Type[T_Pydantic]) -> T_Pydantic:
    """
    Convert AgenticBaseModel → Pydantic for persistence.
    
    Rule: Always use .to_dict() for dataclass serialization.
    """
    if not hasattr(agentic_model, 'to_dict'):
        raise TypeError(f"{type(agentic_model)} must have .to_dict() method")

    return pydantic_class(**agentic_model.to_dict())


def from_persistence(pydantic_model: T_Pydantic, agentic_class: Type[T_Agentic]) -> T_Agentic:
    """
    Convert Pydantic → AgenticBaseModel for internal logic.
    
    Rule: Always use .dict() for Pydantic serialization, .from_dict() for dataclass creation.
    """
    if not hasattr(pydantic_model, 'dict'):
        raise TypeError(f"{type(pydantic_model)} must have .dict() method")

    if not hasattr(agentic_class, 'from_dict'):
        raise TypeError(f"{agentic_class} must have .from_dict() method")

    return agentic_class.from_dict(pydantic_model.dict())


def to_persistence_list(agentic_models: List[T_Agentic], pydantic_class: Type[T_Pydantic]) -> List[T_Pydantic]:
    """Convert list of AgenticBaseModel → list of Pydantic."""
    return [to_persistence(model, pydantic_class) for model in agentic_models]


def from_persistence_list(pydantic_models: List[T_Pydantic], agentic_class: Type[T_Agentic]) -> List[T_Agentic]:
    """Convert list of Pydantic → list of AgenticBaseModel."""
    return [from_persistence(model, agentic_class) for model in pydantic_models]


# Usage examples - much simpler!
"""
# Convert single models
concept_model = to_persistence(concept_ir, ConceptModel)
concept_ir = from_persistence(concept_model, ConceptIR)

# Convert lists
concept_models = to_persistence_list(concepts_ir, ConceptModel)
concepts_ir = from_persistence_list(concept_models, ConceptIR)
"""
