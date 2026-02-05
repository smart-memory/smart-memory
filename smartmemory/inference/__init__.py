"""Automatic graph inference engine."""

from .engine import InferenceEngine
from .rules import InferenceRule, get_default_rules

__all__ = ["InferenceEngine", "InferenceRule", "get_default_rules"]
