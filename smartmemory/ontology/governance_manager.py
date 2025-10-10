"""
Generic Governance API facade for SmartMemory's ontology governance engine.

This module exposes a minimal, app-agnostic interface over OntologyGovernor so
other applications (e.g., services or assistants) can use governance without
depending on any app-specific UX or integration code.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from smartmemory.ontology.governance import (
    OntologyGovernor,
    GovernanceAction,
    ViolationSeverity,
)
from smartmemory.ontology.llm_manager import LLMOntologyManager
from smartmemory.ontology.manager import OntologyManager


class GovernanceManager:
    """Thin, generic facade over OntologyGovernor.

    Intended to be consumed by smart-memory-service or other apps. This keeps
    SmartMemory library app-agnostic while providing a stable API surface.
    """

    def __init__(
            self,
            ontology_manager: Optional[OntologyManager] = None,
            llm_manager: Optional[LLMOntologyManager] = None,
            smart_memory: Optional[Any] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.ontology_manager = ontology_manager or OntologyManager()
        self.llm_manager = llm_manager or LLMOntologyManager(self.ontology_manager)
        self.governor = OntologyGovernor(
            ontology_manager=self.ontology_manager,
            llm_manager=self.llm_manager,
            smart_memory=smart_memory,
        )

    # Analysis
    def run_analysis(self, memory_items: Optional[List[Any]] = None, ontology_ids: Optional[List[str]] = None) -> int:
        """Analyze data against ontologies. Returns number of violations found."""
        violations = self.governor.analyze_data_against_ontologies(
            memory_items=memory_items, ontology_ids=ontology_ids
        )
        return len(violations)

    # Violations
    def list_violations(
            self,
            severity: Optional[str] = None,
            auto_fixable_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """List violations for review as dictionaries.

        severity: one of ["low","medium","high","critical"] or None
        """
        sev_enum: Optional[ViolationSeverity] = None
        if severity:
            severity_map = {
                "low": ViolationSeverity.LOW,
                "medium": ViolationSeverity.MEDIUM,
                "high": ViolationSeverity.HIGH,
                "critical": ViolationSeverity.CRITICAL,
            }
            sev_enum = severity_map.get(severity.lower())

        items = self.governor.get_violations_for_review(
            severity_filter=sev_enum, auto_fixable_only=auto_fixable_only
        )
        return [v.to_dict() for v in items]

    def get_violation(self, violation_id: str) -> Optional[Dict[str, Any]]:
        v = self.governor.violations.get(violation_id)
        return v.to_dict() if v else None

    # Decisions
    def apply_decision(
            self,
            violation_id: str,
            action: str,
            rationale: str,
            decided_by: str = "human",
    ) -> bool:
        """Apply a decision. action is one of GovernanceAction values in snake_case."""
        action_map = {
            "approve": GovernanceAction.APPROVE,
            "reject": GovernanceAction.REJECT,
            "evolve_ontology": GovernanceAction.EVOLVE_ONTOLOGY,
            "fix_data": GovernanceAction.FIX_DATA,
            "ignore": GovernanceAction.IGNORE,
            "review_later": GovernanceAction.REVIEW_LATER,
        }
        if action not in action_map:
            raise ValueError(f"unknown action: {action}")
        return self.governor.apply_governance_decision(
            violation_id, action_map[action], rationale, decided_by=decided_by
        )

    def auto_fix(self, confidence_threshold: float = 0.8) -> Tuple[int, int]:
        """Run auto-fix for eligible violations. Returns (fixed_count, failed_count)."""
        return self.governor.auto_fix_violations(confidence_threshold=confidence_threshold)

    # Summary
    def summary(self) -> Dict[str, Any]:
        """Return a lightweight summary of current governance state."""
        violations = list(self.governor.violations.values())
        totals = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        auto_fixable = 0
        for v in violations:
            totals[v.severity.value] += 1
            if v.auto_fixable:
                auto_fixable += 1
        return {
            "totals_by_severity": totals,
            "auto_fixable_count": auto_fixable,
            "pending_count": len(violations),
            "decisions_total": len(self.governor.decisions),
        }
