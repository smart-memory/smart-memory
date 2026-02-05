"""Orchestrator that runs detectors in order, returns first hit."""

import logging
from typing import List, Optional

from smartmemory.reasoning.challenger import Conflict

from .base import ContradictionDetector, DetectionContext

logger = logging.getLogger(__name__)


class DetectionCascade:
    """Run a sequence of detectors; return the first conflict found."""

    def __init__(self, detectors: List[ContradictionDetector]):
        self.detectors = detectors

    def detect(self, ctx: DetectionContext) -> Optional[Conflict]:
        for detector in self.detectors:
            conflict = detector.detect(ctx)
            if conflict:
                conflict.explanation = f"[{detector.name.upper()}] {conflict.explanation}"
                return conflict
        return None
