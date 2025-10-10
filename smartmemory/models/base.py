from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Optional, List, Any, TypeVar, Type, Dict

from smartmemory.models.compat.dataclass_model import DataclassModelMixin


@dataclass
class User(DataclassModelMixin):
    id: str = ""
    name: str = ""
    email: str = ""
    workspace_id: str = "default_workspace"  # Multi-tenant workspace identifier
    roles: List[str] = field(default_factory=lambda: ["user"])
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional user metadata (tenant_id, etc.)


@dataclass
class MemoryBaseModel(DataclassModelMixin):
    """
    Base models for all internal business logic models.
    
    Provides user context and common functionality while using
    dataclasses for simplicity and performance.
    """
    user: Optional[User] = field(default=None, metadata={"description": "User ID for multi-tenant context"})

    @property
    def user_id(self) -> Optional[str]:
        """Convenience property to get user_id from the authenticated user."""
        return self.user.id if self.user else None

    def set_user(self, user: User) -> None:
        """Set the authenticated user for this request."""
        if user is None:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=401,
                detail="Authentication failed: user cannot be None",
                headers={"WWW-Authenticate": "Bearer"}
            )
        self.user = user

    # ---- Merge helpers on base model so all typed configs inherit them -----
    T = TypeVar("T", bound="MemoryBaseModel")

    @staticmethod
    def _merge_values(base: Any, override: Any) -> Any:
        """Merge two values with sensible defaults:
        - If override is None, keep base
        - If base is None, use override
        - If both are dataclasses of the same type, merge recursively
        - If both are dicts, perform shallow merge (base | override)
        - Otherwise, use override
        """
        if override is None:
            return base
        if base is None:
            return override
        # Recursive dataclass merge (same type)
        if is_dataclass(base) and is_dataclass(override) and type(base) is type(override):
            return MemoryBaseModel._merge_dataclasses(type(base), base, override)
        # Shallow dict merge
        if isinstance(base, dict) and isinstance(override, dict):
            out = dict(base)
            out.update(override)
            return out
        # Lists/tuples/sets: prefer override when provided
        return override

    @staticmethod
    def _merge_dataclasses(cls: Type[T], base: T, override: T) -> T:
        """Field-wise merge two instances of the same dataclass type.
        Non-None override fields replace base; nested dataclasses are merged recursively.
        """
        values: Dict[str, Any] = {}
        for f in fields(cls):
            b_val = getattr(base, f.name)
            o_val = getattr(override, f.name)
            values[f.name] = MemoryBaseModel._merge_values(b_val, o_val)
        return replace(base, **values)

    def merged_with(self: T, override: Optional[T]) -> T:
        """Return a new instance with override non-None fields applied.
        If override is None, returns a copy of self.
        """
        if override is None:
            return replace(self)
        return self._merge_dataclasses(type(self), self, override)

    def merge_into(self: T, override: Optional[T]) -> T:
        """In-place style merge: updates self's fields and returns self.
        Implemented by replacing self with merged result via replace semantics.
        """
        merged = self.merged_with(override)
        for f in fields(type(self)):
            setattr(self, f.name, getattr(merged, f.name))
        return self


@dataclass
class StageRequest(MemoryBaseModel):
    """Base request model for pipeline stages, enrichers, evolvers, and service-layer DTOs.
    Many typed request models across repos import this from smartmemory.models.base.
    """
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None

    # No merge helpers here; they live on MemoryBaseModel
