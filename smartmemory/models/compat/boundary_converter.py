"""
Boundary Conversion Interface

Enforces the rule: Always convert between AgenticBaseModel (internal) and Pydantic (persistence/API) at boundaries.
Provides a clean interface to handle conversions seamlessly.
"""

from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from typing import TypeVar, Type, Dict, Any, List

from smartmemory.models.base import MemoryBaseModel

# Type variables for generic conversion
T_Agentic = TypeVar('T_Agentic', bound=MemoryBaseModel)
T_Pydantic = TypeVar('T_Pydantic')


class BoundaryConverter(ABC):
    """
    Abstract interface for converting between internal AgenticBaseModel 
    and external Pydantic models at service boundaries.
    """

    @abstractmethod
    def to_persistence(self, agentic_model: T_Agentic) -> T_Pydantic:
        """Convert AgenticBaseModel to Pydantic for persistence/API boundaries."""
        pass

    @abstractmethod
    def from_persistence(self, pydantic_model: T_Pydantic) -> T_Agentic:
        """Convert Pydantic models back to AgenticBaseModel for internal logic."""
        pass


class DefaultBoundaryConverter(BoundaryConverter):
    """
    Default implementation using DataclassModelMixin conversion methods.
    Enforces the boundary conversion rule automatically.
    """

    def __init__(self, agentic_class: Type[T_Agentic], pydantic_class: Type[T_Pydantic]):
        self.agentic_class = agentic_class
        self.pydantic_class = pydantic_class

        # Validate that agentic_class is a dataclass
        if not is_dataclass(agentic_class):
            raise ValueError(f"{agentic_class} must be a dataclass (AgenticBaseModel)")

    def to_persistence(self, agentic_model: T_Agentic) -> T_Pydantic:
        """
        Convert AgenticBaseModel → Pydantic for persistence boundary.
        
        Rule: Always use .to_dict() for dataclass serialization.
        """
        if not isinstance(agentic_model, self.agentic_class):
            raise TypeError(f"Expected {self.agentic_class}, got {type(agentic_model)}")

        # Use DataclassModelMixin .to_dict() method
        agentic_data = agentic_model.to_dict()

        # Create Pydantic models from dict
        return self.pydantic_class(**agentic_data)

    def from_persistence(self, pydantic_model: T_Pydantic) -> T_Agentic:
        """
        Convert Pydantic → AgenticBaseModel for internal logic boundary.
        
        Rule: Always use .dict() for Pydantic serialization, .from_dict() for dataclass creation.
        """
        if not isinstance(pydantic_model, self.pydantic_class):
            raise TypeError(f"Expected {self.pydantic_class}, got {type(pydantic_model)}")

        # Use Pydantic .dict() method
        pydantic_data = pydantic_model.dict()

        # Create AgenticBaseModel from dict using DataclassModelMixin
        return self.agentic_class.from_dict(pydantic_data)


class BulkBoundaryConverter:
    """
    Handles bulk conversions for lists of models at boundaries.
    Enforces conversion rules for collections.
    """

    def __init__(self, converter: BoundaryConverter):
        self.converter = converter

    def to_persistence_list(self, agentic_models: List[T_Agentic]) -> List[T_Pydantic]:
        """Convert list of AgenticBaseModel → list of Pydantic models."""
        return [self.converter.to_persistence(model) for model in agentic_models]

    def from_persistence_list(self, pydantic_models: List[T_Pydantic]) -> List[T_Agentic]:
        """Convert list of Pydantic models → list of AgenticBaseModel."""
        return [self.converter.from_persistence(model) for model in pydantic_models]


class BoundaryConversionMixin:
    """
    Mixin for service classes to enforce boundary conversion rules.
    Provides standardized conversion methods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._converters: Dict[str, BoundaryConverter] = {}

    def register_converter(self, name: str, converter: BoundaryConverter) -> None:
        """Register a boundary converter for a specific models type."""
        self._converters[name] = converter

    def convert_to_persistence(self, name: str, agentic_model: T_Agentic) -> T_Pydantic:
        """Convert AgenticBaseModel to Pydantic using registered converter."""
        if name not in self._converters:
            raise ValueError(f"No converter registered for '{name}'")
        return self._converters[name].to_persistence(agentic_model)

    def convert_from_persistence(self, name: str, pydantic_model: T_Pydantic) -> T_Agentic:
        """Convert Pydantic to AgenticBaseModel using registered converter."""
        if name not in self._converters:
            raise ValueError(f"No converter registered for '{name}'")
        return self._converters[name].from_persistence(pydantic_model)

    def bulk_convert_to_persistence(self, name: str, agentic_models: List[T_Agentic]) -> List[T_Pydantic]:
        """Bulk convert AgenticBaseModel list to Pydantic list."""
        if name not in self._converters:
            raise ValueError(f"No converter registered for '{name}'")
        bulk_converter = BulkBoundaryConverter(self._converters[name])
        return bulk_converter.to_persistence_list(agentic_models)

    def bulk_convert_from_persistence(self, name: str, pydantic_models: List[T_Pydantic]) -> List[T_Agentic]:
        """Bulk convert Pydantic list to AgenticBaseModel list."""
        if name not in self._converters:
            raise ValueError(f"No converter registered for '{name}'")
        bulk_converter = BulkBoundaryConverter(self._converters[name])
        return bulk_converter.from_persistence_list(pydantic_models)


def create_converter(agentic_class: Type[T_Agentic], pydantic_class: Type[T_Pydantic]) -> DefaultBoundaryConverter:
    """
    Factory function to create a boundary converter.
    
    Usage:
        converter = create_converter(ConceptIR, ConceptDocument)
        pydantic_model = converter.to_persistence(agentic_model)
        agentic_model = converter.from_persistence(pydantic_model)
    """
    return DefaultBoundaryConverter(agentic_class, pydantic_class)


# Convenience functions for direct conversion (enforces the rule)
def to_persistence_dict(agentic_model: MemoryBaseModel) -> Dict[str, Any]:
    """
    Convert AgenticBaseModel to dict for persistence boundary.
    
    Rule: Always use .to_dict() for dataclass serialization.
    """
    if not hasattr(agentic_model, 'to_dict'):
        raise TypeError(f"{type(agentic_model)} must have .to_dict() method (inherit from DataclassModelMixin)")
    return agentic_model.to_dict()


def from_persistence_dict(agentic_class: Type[T_Agentic], data: Dict[str, Any]) -> T_Agentic:
    """
    Convert dict from persistence to AgenticBaseModel.
    
    Rule: Always use .from_dict() for dataclass creation.
    """
    if not hasattr(agentic_class, 'from_dict'):
        raise TypeError(f"{agentic_class} must have .from_dict() method (inherit from DataclassModelMixin)")
    return agentic_class.from_dict(data)
