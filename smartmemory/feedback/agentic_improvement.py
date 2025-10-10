from typing import List, Any, Dict, Callable


class AgenticImprover:
    """
    Automates agentic improvement/learning based on feedback.
    Can be used to update memory, re-plan, or adjust strategy.
    """

    def __init__(self, update_fn: Callable = None):
        self.update_fn = update_fn
        self.history: List[Dict[str, Any]] = []

    def record(self, plan: List[Any], feedbacks: List[Any]):
        self.history.append({"plan": plan, "feedbacks": feedbacks})

    def aggregate_feedback(self) -> float:
        # Example: mean of numeric feedbacks (extend as needed)
        scores = []
        for entry in self.history:
            for fb in entry["feedbacks"]:
                if isinstance(fb, (int, float)):
                    scores.append(fb)
                elif isinstance(fb, str):
                    if "good" in fb.lower():
                        scores.append(1)
                    elif "bad" in fb.lower() or "stop" in fb.lower():
                        scores.append(-1)
                    else:
                        scores.append(0)
        return sum(scores) / len(scores) if scores else 0.0

    def improve(self, agent, memory_store, last_plan: List[Any], feedbacks: List[Any]):
        """
        Example: If feedback is negative, trigger replanning or memory update.
        """
        agg = self.aggregate_feedback()
        if agg < 0:
            if self.update_fn:
                self.update_fn(agent, memory_store, last_plan, feedbacks)
            # Could trigger agentic reflection, pruning, or new exploration
            return "Triggered improvement"
        return "No improvement needed"


def fully_automatic_agentic_loop(agent, memory_store, initial_goal, feedback_mgr, improver, planner_fn, max_cycles=10, planner_kwargs=None, stop_on_positive=True):
    """
    Runs a fully automatic agentic improvement loop.
    - agent: agent instance (or None)
    - memory_store: memory store with search
    - initial_goal: starting goal/query
    - feedback_mgr: FeedbackManager instance
    - improver: AgenticImprover instance
    - planner_fn: planning routine (e.g., agentic_goal_directed_planning)
    - max_cycles: maximum improvement cycles
    - planner_kwargs: extra kwargs for planner_fn
    - stop_on_positive: stop if feedback is positive/neutral
    Returns: (final_plan, feedback_history, improvement_history)
    """
    planner_kwargs = planner_kwargs or {}
    goal = initial_goal
    improvement_history = []
    for cycle in range(max_cycles):
        plan, monitor = planner_fn(
            agent=agent,
            memory_store=memory_store,
            goal=goal,
            feedback_fn=feedback_mgr.get_channel('cli'),  # or 'slack', 'web', etc.
            **planner_kwargs
        )
        feedbacks = [fb['feedback'] for fb in monitor.feedback_log]
        improver.record(plan, feedbacks)
        result = improver.improve(agent, memory_store, plan, feedbacks)
        improvement_history.append(result)
        print(f"Cycle {cycle}: {result}")
        if stop_on_positive and result == "No improvement needed":
            break
        # Optionally update goal/strategy here for next cycle
    return plan, improver.history, improvement_history
