# SmartMemory Design Standards

## Data Modeling Standards

### 1. Core Domain Models
- **Standard**: Use `MemoryBaseModel` (based on Python `dataclasses`) for all internal domain models.
- **Rationale**: Dataclasses provide better performance and simplicity for internal logic compared to Pydantic.
- **Base Class**: All domain models should inherit from `smartmemory.models.base.MemoryBaseModel`.

### 2. External Compatibility
- **Standard**: Use Pydantic **only** when required for external interfaces (e.g., FastAPI request/response models).
- **Conversion**: Provide explicit conversion methods (e.g., `to_pydantic()`, `from_pydantic()`) if mapping between internal dataclasses and external Pydantic models is necessary.

### 3. Shared Models
- **Goal**: Core domain models should be lightweight and dependency-free.
- **Location**: Models shared between Client, Service, and Core should reside in a dedicated package (e.g., `smart-memory-models`) to avoid heavy dependency coupling.

## Dependency Management
- **Explicit Dependencies**: All projects must have a `requirements.txt` or `pyproject.toml` defining their direct dependencies.
- **No Implicit Paths**: Do not use `sys.path` hacking to import local projects. Install them in editable mode (`pip install -e .`) during development.
