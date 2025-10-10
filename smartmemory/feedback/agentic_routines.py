import logging
import numpy as np
from typing import Dict, Any


class AgenticSelfMonitor:
    """
    Tracks memory usage, retrievals, and agentic actions for reflection and optimization.
    """

    def __init__(self):
        self.usage_log = []
        self.feedback_log = []
        self.iteration_log = []
        self.logger = logging.getLogger(__name__)

    def log_usage(self, action: str, details: Dict[str, Any]):
        self.usage_log.append({"action": action, "details": details})
        self.logger.info(f"Usage logged: {action} | {details}")

    def log_feedback(self, feedback: str, context: Dict[str, Any]):
        self.feedback_log.append({"feedback": feedback, "context": context})
        self.logger.info(f"Feedback logged: {feedback}")

    def log_iteration(self, iteration_num: int, summary: str):
        self.iteration_log.append({"iteration": iteration_num, "summary": summary})
        self.logger.info(f"Iteration {iteration_num} summary: {summary}")

    def summarize_usage(self) -> str:
        return f"Total actions: {len(self.usage_log)} | Actions: {[u['action'] for u in self.usage_log]}"

    def summarize_feedback(self) -> str:
        return f"Feedback count: {len(self.feedback_log)} | Last: {self.feedback_log[-1]['feedback'] if self.feedback_log else None}"

    def summarize_iterations(self) -> str:
        return f"Iterations: {len(self.iteration_log)} | Last: {self.iteration_log[-1]['summary'] if self.iteration_log else None}"

    def reset(self):
        self.usage_log.clear()
        self.feedback_log.clear()
        self.iteration_log.clear()
        self.logger.info("AgenticSelfMonitor state reset.")


# Advanced agentic routines
import random


def agentic_mcts_planning(memory_store, start_goal, max_depth=5, simulations=10, scorer=None):
    """
    Monte Carlo Tree Search for agentic planning.
    Explores multiple plans in parallel, simulates outcomes, and scores them.
    Returns best plan and score.
    """
    scorer = scorer or PlanScorer()
    best_plan = []
    best_score = -float('inf')
    for _ in range(simulations):
        plan = [start_goal]
        current_goal = start_goal
        for d in range(max_depth):
            results = memory_store.search(current_goal)
            if not results:
                break
            next_goal = random.choice(results).get('key')
            if not next_goal or next_goal in plan:
                break
            plan.append(next_goal)
            current_goal = next_goal
        score = scorer.score_plan(plan)
        if score > best_score:
            best_score = score
            best_plan = plan
    return best_plan, best_score


class PlanScorer:
    """
    Scores plans based on length, diversity, feedback, and custom criteria.
    """

    def __init__(self):
        self.scores = []
        self.feedback = []

    def score_plan(self, plan, feedbacks=None):
        score = 0
        if not plan:
            return 0
        # Reward longer plans (to a point)
        score += min(len(plan), 10)
        # Reward diversity (unique nodes)
        score += len(set(plan))
        # Penalize cycles
        if len(plan) != len(set(plan)):
            score -= 2
        # Incorporate feedback (if any)
        if feedbacks:
            for fb in feedbacks:
                if fb and "good" in fb.lower():
                    score += 2
                elif fb and "bad" in fb.lower():
                    score -= 2
        self.scores.append(score)
        if feedbacks:
            self.feedback.extend(feedbacks)
        return score


def agentic_goal_directed_planning(agent, memory_store, goal: str, max_steps: int = 5, feedback_fn=None, strategy="greedy", scorer=None):
    """
    Plan and execute a sequence of reasoning steps to achieve a goal, incorporating feedback at each step.
    Supports different planning strategies: 'greedy', 'breadth', 'depth'.
    feedback_fn: function taking (step, context, results) and returning feedback string or None.
    scorer: PlanScorer instance for automated scoring.
    """
    monitor = AgenticSelfMonitor()
    plan = [goal]
    context = {}
    feedbacks = []
    explored = set()
    for step in range(max_steps):
        current_goal = plan[-1]
        monitor.log_usage("plan_step", {"goal": current_goal, "step": step})
        # Chain-of-thought: retrieve relevant facts
        results = memory_store.search(current_goal)
        monitor.log_usage("retrieval", {"results": results})
        explored.add(current_goal)
        # Planning strategies
        if strategy == "greedy":
            # Pick the top result
            next_goal = results[0]["key"] if results and "key" in results[0] else None
            if next_goal and next_goal not in explored:
                plan.append(next_goal)
        elif strategy == "breadth":
            # Add all new results at this level
            for r in results:
                k = r.get("key")
                if k and k not in explored:
                    plan.append(k)
        elif strategy == "depth":
            # Recursively go deeper with the first unexplored result
            for r in results:
                k = r.get("key")
                if k and k not in explored:
                    plan.append(k)
                    break
        # Human-in-the-loop feedback
        feedback = None
        if feedback_fn:
            feedback = feedback_fn(step, context, results)
            feedbacks.append(feedback)
            monitor.log_feedback(feedback or "", {"results": results, "plan": plan})
        # Simulate agentic learning/summary
        summary = f"Step {step}: goal={current_goal}, results={len(results)}, feedback={feedback}"
        monitor.log_iteration(step, summary)
        if not results or (feedback and "stop" in (feedback or "").lower()):
            break
    # Automated scoring
    if scorer:
        score = scorer.score_plan(plan, feedbacks)
        monitor.log_usage("plan_score", {"score": score, "plan": plan})
    return plan, monitor


def agentic_weighted_chain_of_thought(agent, memory_store, prompt: str, steps: int = 3, weight_fn=None):
    """
    Weighted chain-of-thought: at each step, select next prompt based on a weight_fn (e.g., relevance, score).
    """
    thoughts = []
    for i in range(steps):
        results = memory_store.search(prompt)
        weights = [weight_fn(r) if weight_fn else 1.0 for r in results]
        thought = {
            "step": i,
            "prompt": prompt,
            "results": results,
            "weights": weights
        }
        thoughts.append(thought)
        # Pick next prompt by max weight
        if results and weights:
            idx = int(np.argmax(weights))
            prompt = results[idx]["content"] if "content" in results[idx] else prompt
    return thoughts


def agentic_chain_of_thought(agent, memory_store, prompt: str, steps: int = 3):
    """
    Perform a chain-of-thought reasoning process, collecting intermediate thoughts and results.
    """
    thoughts = []
    for i in range(steps):
        results = memory_store.search(prompt)
        thought = {
            "step": i,
            "prompt": prompt,
            "results": results
        }
        thoughts.append(thought)
        # Optionally, update prompt based on results (simple: use top result as next prompt)
        if results:
            prompt = results[0]["content"] if "content" in results[0] else prompt
    return thoughts
