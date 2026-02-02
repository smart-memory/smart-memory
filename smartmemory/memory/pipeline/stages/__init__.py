"""Pipeline stages for SmartMemory ingestion flow."""
from .coreference import CoreferenceStage, CoreferenceResult

__all__ = [
    "CoreferenceStage",
    "CoreferenceResult",
]
