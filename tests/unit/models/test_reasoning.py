"""Unit tests for Reasoning models."""

from smartmemory.models.reasoning import (
    ReasoningStep,
    TaskContext,
    ReasoningEvaluation,
    ReasoningTrace,
)


class TestReasoningStep:
    def test_creation(self):
        step = ReasoningStep(type="thought", content="Let me analyze this")
        assert step.type == "thought"
        assert step.content == "Let me analyze this"

    def test_to_dict_from_dict_roundtrip(self):
        original = ReasoningStep(type="action", content="Search the database")
        data = original.to_dict()
        restored = ReasoningStep.from_dict(data)
        assert restored.type == "action"
        assert restored.content == "Search the database"

    def test_from_dict_defaults(self):
        step = ReasoningStep.from_dict({})
        assert step.type == "thought"
        assert step.content == ""


class TestTaskContext:
    def test_creation_defaults(self):
        ctx = TaskContext()
        assert ctx.goal is None
        assert ctx.input is None
        assert ctx.task_type is None
        assert ctx.domain is None
        assert ctx.complexity is None

    def test_creation_with_values(self):
        ctx = TaskContext(
            goal="Fix the bug",
            input="user reported crash",
            task_type="debugging",
            domain="backend",
            complexity="high",
        )
        assert ctx.goal == "Fix the bug"
        assert ctx.domain == "backend"
        assert ctx.complexity == "high"

    def test_to_dict_from_dict_roundtrip(self):
        original = TaskContext(
            goal="Create feature",
            task_type="code_generation",
            domain="frontend",
            complexity="medium",
        )
        data = original.to_dict()
        restored = TaskContext.from_dict(data)
        assert restored.goal == "Create feature"
        assert restored.task_type == "code_generation"
        assert restored.domain == "frontend"
        assert restored.complexity == "medium"

    def test_from_dict_with_empty_dict(self):
        ctx = TaskContext.from_dict({})
        assert ctx.goal is None
        assert ctx.input is None


class TestReasoningEvaluation:
    def test_creation(self):
        ev = ReasoningEvaluation(quality_score=0.8)
        assert ev.quality_score == 0.8
        assert ev.has_loops is False
        assert ev.has_redundancy is False
        assert ev.step_diversity == 0.0
        assert ev.issues == []
        assert ev.suggestions == []

    def test_should_store_true_when_quality_high_no_issues(self):
        ev = ReasoningEvaluation(quality_score=0.7)
        assert ev.should_store is True

    def test_should_store_true_at_threshold(self):
        ev = ReasoningEvaluation(quality_score=0.4)
        assert ev.should_store is True

    def test_should_store_false_when_quality_below_threshold(self):
        ev = ReasoningEvaluation(quality_score=0.3)
        assert ev.should_store is False

    def test_should_store_false_with_high_severity_issue(self):
        ev = ReasoningEvaluation(
            quality_score=0.8,
            issues=[{"type": "loop", "description": "Circular reasoning", "severity": "high"}],
        )
        assert ev.should_store is False

    def test_should_store_true_with_low_severity_issue(self):
        ev = ReasoningEvaluation(
            quality_score=0.6,
            issues=[{"type": "minor", "description": "Slight redundancy", "severity": "low"}],
        )
        assert ev.should_store is True

    def test_should_store_false_quality_and_high_issue(self):
        ev = ReasoningEvaluation(
            quality_score=0.2,
            issues=[{"type": "loop", "description": "Bad", "severity": "high"}],
        )
        assert ev.should_store is False

    def test_to_dict_includes_should_store(self):
        ev = ReasoningEvaluation(quality_score=0.5)
        data = ev.to_dict()
        assert "should_store" in data
        assert data["should_store"] is True


class TestReasoningTrace:
    def test_creation(self):
        trace = ReasoningTrace(trace_id="trace-1")
        assert trace.trace_id == "trace-1"
        assert trace.steps == []
        assert trace.task_context is None
        assert trace.evaluation is None
        assert trace.session_id is None
        assert trace.has_explicit_markup is False
        assert trace.artifact_ids == []

    def test_step_count(self):
        steps = [
            ReasoningStep(type="thought", content="thinking"),
            ReasoningStep(type="action", content="doing"),
            ReasoningStep(type="conclusion", content="done"),
        ]
        trace = ReasoningTrace(trace_id="trace-2", steps=steps)
        assert trace.step_count == 3

    def test_step_count_empty(self):
        trace = ReasoningTrace(trace_id="trace-3")
        assert trace.step_count == 0

    def test_content_property_formats_steps(self):
        steps = [
            ReasoningStep(type="thought", content="analyze problem"),
            ReasoningStep(type="action", content="run query"),
        ]
        trace = ReasoningTrace(trace_id="trace-4", steps=steps)
        content = trace.content
        assert "Thought: analyze problem" in content
        assert "Action: run query" in content

    def test_content_includes_task_context(self):
        ctx = TaskContext(goal="Fix auth bug", input="login fails for users")
        trace = ReasoningTrace(
            trace_id="trace-5",
            task_context=ctx,
            steps=[ReasoningStep(type="thought", content="check token")],
        )
        content = trace.content
        assert "Goal: Fix auth bug" in content
        assert "Input: login fails for users" in content

    def test_to_dict_from_dict_roundtrip(self):
        steps = [
            ReasoningStep(type="thought", content="thinking hard"),
            ReasoningStep(type="decision", content="go with plan A"),
        ]
        ctx = TaskContext(goal="Plan migration", domain="backend")
        ev = ReasoningEvaluation(quality_score=0.9)
        original = ReasoningTrace(
            trace_id="trace-6",
            steps=steps,
            task_context=ctx,
            evaluation=ev,
            session_id="session-1",
            has_explicit_markup=True,
            artifact_ids=["art-1", "art-2"],
        )
        data = original.to_dict()
        restored = ReasoningTrace.from_dict(data)

        assert restored.trace_id == "trace-6"
        assert restored.step_count == 2
        assert restored.steps[0].type == "thought"
        assert restored.steps[0].content == "thinking hard"
        assert restored.steps[1].type == "decision"
        assert restored.task_context.goal == "Plan migration"
        assert restored.task_context.domain == "backend"
        assert restored.evaluation.quality_score == 0.9
        assert restored.session_id == "session-1"
        assert restored.has_explicit_markup is True
        assert restored.artifact_ids == ["art-1", "art-2"]

    def test_to_dict_includes_step_count(self):
        trace = ReasoningTrace(
            trace_id="trace-7",
            steps=[ReasoningStep(type="thought", content="x")],
        )
        data = trace.to_dict()
        assert data["step_count"] == 1

    def test_from_dict_with_minimal_data(self):
        trace = ReasoningTrace.from_dict({"trace_id": "trace-8"})
        assert trace.trace_id == "trace-8"
        assert trace.steps == []
        assert trace.task_context is None
        assert trace.evaluation is None
