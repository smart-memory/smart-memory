"""
Maya Integration with SmartMemory for Ontology Governance

This module integrates Maya's conversational HITL interface directly into SmartMemory,
allowing users to interact with ontology governance through natural language.
"""

import logging

from smartmemory.integration.chat.hitl import MayaHITLAssistant
from smartmemory.ontology.governance import OntologyGovernor
from smartmemory.ontology.llm_manager import LLMOntologyManager
from smartmemory.ontology.manager import OntologyManager


class SmartMemoryMayaIntegration:
    """Integration layer between SmartMemory and Maya for ontology governance."""

    def __init__(self, smart_memory):
        self.smart_memory = smart_memory
        self.logger = logging.getLogger(__name__)

        # Initialize ontology governance stages
        self.ontology_manager = OntologyManager()
        self.llm_manager = LLMOntologyManager(self.ontology_manager)
        self.governor = OntologyGovernor(
            ontology_manager=self.ontology_manager,
            llm_manager=self.llm_manager,
            smart_memory=smart_memory
        )

        # Initialize Maya
        self.maya = MayaHITLAssistant(self.governor)

        # Conversation state
        self.active_conversation = False
        self.current_violation = None
        self.pending_confirmation = None

    def maya_chat(self, user_message: str) -> str:
        """Main entry point for Maya conversations about ontology governance."""

        message_lower = user_message.lower().strip()

        # Handle conversation starters
        if any(word in message_lower for word in ['maya', 'governance', 'ontology', 'violations', 'review']):
            return self._start_governance_conversation()

        # Handle help requests
        elif any(word in message_lower for word in ['help', 'what can you do', 'how does this work']):
            return self.maya.maya_help_message()

        # Handle summary requests
        elif any(word in message_lower for word in ['summary', 'dashboard', 'status', 'overview']):
            return self.maya.maya_governance_summary()

        # Handle active conversation
        elif self.active_conversation:
            return self._handle_active_conversation(user_message)

        # Default response
        else:
            return self._suggest_governance_check()

    def _start_governance_conversation(self) -> str:
        """Start a governance conversation with Maya."""

        # Run fresh analysis
        self._run_governance_analysis()

        # Start conversation
        self.active_conversation = True
        return self.maya.maya_greeting()

    def _run_governance_analysis(self) -> None:
        """Run governance analysis on recent memory items."""

        try:
            # Get recent memory items
            recent_items = self.smart_memory.search("*", top_k=100)

            if recent_items:
                # Analyze against ontologies
                violations = self.governor.analyze_data_against_ontologies(
                    memory_items=recent_items
                )

                # Auto-fix high-confidence violations
                self.governor.auto_fix_violations(confidence_threshold=0.8)

                self.logger.info(f"Governance analysis complete: {len(violations)} violations found")

        except Exception as e:
            self.logger.error(f"Error running governance analysis: {e}")

    def _handle_active_conversation(self, user_message: str) -> str:
        """Handle ongoing governance conversation."""

        # If we're waiting for confirmation
        if self.pending_confirmation:
            return self._handle_confirmation(user_message)

        # If we're reviewing a specific violation
        elif self.current_violation:
            return self._handle_violation_decision(user_message)

        # Otherwise, start reviewing violations
        else:
            return self._start_violation_review()

    def _start_violation_review(self) -> str:
        """Start reviewing violations with Maya."""

        violations = self.governor.get_violations_for_review()

        if not violations:
            self.active_conversation = False
            return """
🎉 Excellent! No violations found that need your attention.

Your ontology governance is all caught up. I'll keep monitoring 
and let you know if anything new comes up.

Feel free to ask me for a summary anytime! ✨
            """.strip()

        # Check for batch opportunities
        violation_groups = {}
        for violation in violations:
            key = violation.violation_type
            if key not in violation_groups:
                violation_groups[key] = []
            violation_groups[key].append(violation)

        # Find largest group for potential batching
        largest_group = max(violation_groups.values(), key=len)

        if len(largest_group) >= 3:
            batch_suggestion = self.maya.maya_batch_suggestion(largest_group)
            if batch_suggestion:
                return batch_suggestion

        # Start with first violation
        self.current_violation = violations[0]
        return self.maya.maya_ask_for_decision(self.current_violation)

    def _handle_violation_decision(self, user_message: str) -> str:
        """Handle user's decision on a violation."""

        action, rationale = self.maya.maya_interpret_response(
            user_message, self.current_violation
        )

        if action is None:
            # Maya couldn't understand, ask for clarification
            return self.maya.maya_ask_clarification(user_message)

        # Store pending confirmation
        self.pending_confirmation = {
            'action': action,
            'rationale': rationale,
            'violation': self.current_violation
        }

        return self.maya.maya_confirm_decision(action, rationale, self.current_violation)

    def _handle_confirmation(self, user_message: str) -> str:
        """Handle user's confirmation of a decision."""

        message_lower = user_message.lower().strip()

        if any(word in message_lower for word in ['yes', 'confirm', 'correct', 'proceed', 'do it']):
            # Apply the decision
            action = self.pending_confirmation['action']
            rationale = self.pending_confirmation['rationale']
            violation = self.pending_confirmation['violation']

            success = self.governor.apply_governance_decision(
                violation.id, action, rationale, decided_by="human"
            )

            if success:
                response = self.maya.maya_decision_applied(action, violation)

                # Clear state and move to next violation
                self.current_violation = None
                self.pending_confirmation = None

                # Check if there are more violations
                remaining_violations = self.governor.get_violations_for_review()
                if remaining_violations:
                    # Continue with next violation
                    self.current_violation = remaining_violations[0]
                    response += "\n\n" + "=" * 40 + "\n\n"
                    response += self.maya.maya_ask_for_decision(self.current_violation)
                else:
                    # All done!
                    self.active_conversation = False
                    response += "\n\n🌟 All violations resolved! Great teamwork!"

                return response
            else:
                return "❌ Sorry, I couldn't apply that decision. Let's try again."

        elif any(word in message_lower for word in ['no', 'change', 'different', 'wrong']):
            # Go back to decision making
            violation = self.pending_confirmation['violation']
            self.pending_confirmation = None

            return "No problem! Let's choose a different action.\n\n" + \
                self.maya.maya_ask_for_decision(violation)

        else:
            return """
I need a clear yes or no. 

• Say "yes" or "confirm" to proceed with this decision
• Say "no" or "change" to choose differently

What would you like to do? 😊
            """.strip()

    def _suggest_governance_check(self) -> str:
        """Suggest running a governance check."""

        return """
👋 Hi! I'm Maya, your ontology governance assistant.

I can help you review and manage your data quality and ontology evolution.
Would you like me to check for any ontology issues that need attention?

Just say "review" or "check governance" and I'll take a look! 😊

You can also ask me for:
• "summary" - Current governance status
• "help" - What I can do for you
        """.strip()


# Integration with SmartMemory
def add_maya_to_smart_memory(smart_memory_class):
    """Add Maya integration to SmartMemory class."""

    def maya_governance_chat(self, user_message: str) -> str:
        """Chat with Maya about ontology governance."""

        if not hasattr(self, '_maya_integration'):
            self._maya_integration = SmartMemoryMayaIntegration(self)

        return self._maya_integration.maya_chat(user_message)

    def maya_governance_status(self) -> str:
        """Get governance status from Maya."""

        if not hasattr(self, '_maya_integration'):
            self._maya_integration = SmartMemoryMayaIntegration(self)

        return self._maya_integration.maya.maya_governance_summary()

    def maya_run_governance_check(self) -> str:
        """Run governance analysis and get Maya's summary."""

        if not hasattr(self, '_maya_integration'):
            self._maya_integration = SmartMemoryMayaIntegration(self)

        self._maya_integration._run_governance_analysis()
        return self._maya_integration.maya.maya_governance_summary()

    # Add methods to the class
    smart_memory_class.maya_governance_chat = maya_governance_chat
    smart_memory_class.maya_governance_status = maya_governance_status
    smart_memory_class.maya_run_governance_check = maya_run_governance_check

    return smart_memory_class
