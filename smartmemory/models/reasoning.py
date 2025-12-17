"""
Reasoning Trace Models for System 2 Memory 

Captures chain-of-thought reasoning traces from agent conversations,
enabling retrieval of "why" decisions were made, not just the outcomes.


"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Literal

from smartmemory.models.base import MemoryBaseModel


# Step types for reasoning traces
ReasoningStepType = Literal[
    'thought',      # Internal reasoning/consideration
    'action',       # Action taken or proposed
    'observation',  # Result observed from action
    'decision',     # Decision point reached
    'conclusion',   # Final conclusion/answer
    'reflection',   # Meta-reasoning about the process
]


@dataclass
class ReasoningStep(MemoryBaseModel):
    """
    A single step in a reasoning trace.
    
    Step types:
    - thought: Internal reasoning ("I think we should...")
    - action: Action taken ("Let me search for...")
    - observation: Result observed ("The search returned...")
    - decision: Decision point ("Based on this, I'll...")
    - conclusion: Final answer ("Therefore, the solution is...")
    - reflection: Meta-reasoning ("This approach worked because...")
    """
    type: ReasoningStepType
    content: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'content': self.content,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningStep':
        return cls(
            type=data.get('type', 'thought'),
            content=data.get('content', ''),
        )


@dataclass
class TaskContext(MemoryBaseModel):
    """
    Context about the task that prompted the reasoning.
    
    Used for filtered retrieval - find reasoning traces
    for similar tasks/domains/complexity levels.
    """
    goal: Optional[str] = None           # "Fix or debug issue", "Create or write code"
    input: Optional[str] = None          # Original user input that triggered reasoning
    task_type: Optional[str] = None      # code_generation, analysis, problem_solving, debugging
    domain: Optional[str] = None         # javascript, python, frontend, backend, database
    complexity: Optional[Literal['low', 'medium', 'high']] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'goal': self.goal,
            'input': self.input,
            'task_type': self.task_type,
            'domain': self.domain,
            'complexity': self.complexity,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskContext':
        return cls(
            goal=data.get('goal'),
            input=data.get('input'),
            task_type=data.get('task_type'),
            domain=data.get('domain'),
            complexity=data.get('complexity'),
        )


@dataclass
class ReasoningEvaluation(MemoryBaseModel):
    """
    Quality evaluation of a reasoning trace.
    
    Used as a gate before storage - prevents low-quality
    traces from polluting memory.
    
    Threshold: quality_score >= 0.4 and no high-severity issues
    """
    quality_score: float  # 0.0-1.0
    has_loops: bool = False  # Detected repetitive reasoning
    has_redundancy: bool = False  # Multiple steps say the same thing
    step_diversity: float = 0.0  # Ratio of unique step types
    issues: List[Dict[str, Any]] = field(default_factory=list)  # {type, description, severity}
    suggestions: List[str] = field(default_factory=list)
    
    @property
    def should_store(self) -> bool:
        """
        Determine if this trace should be stored based on quality.
        
        Criteria:
        - quality_score >= 0.4
        - No high-severity issues
        """
        if self.quality_score < 0.4:
            return False
        high_severity_issues = [i for i in self.issues if i.get('severity') == 'high']
        return len(high_severity_issues) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'quality_score': self.quality_score,
            'has_loops': self.has_loops,
            'has_redundancy': self.has_redundancy,
            'step_diversity': self.step_diversity,
            'issues': self.issues,
            'suggestions': self.suggestions,
            'should_store': self.should_store,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningEvaluation':
        return cls(
            quality_score=data.get('quality_score', 0.0),
            has_loops=data.get('has_loops', False),
            has_redundancy=data.get('has_redundancy', False),
            step_diversity=data.get('step_diversity', 0.0),
            issues=data.get('issues', []),
            suggestions=data.get('suggestions', []),
        )


@dataclass
class ReasoningTrace(MemoryBaseModel):
    """
    A complete reasoning trace capturing the thought process
    that led to a decision or artifact.
    
    This is the main data structure for System 2 memory.
    Stored as MemoryType.REASONING and linked to resulting
    artifacts via CAUSES relation edges.
    """
    trace_id: str
    steps: List[ReasoningStep] = field(default_factory=list)
    task_context: Optional[TaskContext] = None
    evaluation: Optional[ReasoningEvaluation] = None
    
    # Metadata
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    has_explicit_markup: bool = False  # True if parsed from Thought:/Action: markers
    
    # Links to artifacts this reasoning produced
    artifact_ids: List[str] = field(default_factory=list)  # item_ids of resulting memories
    
    @property
    def step_count(self) -> int:
        return len(self.steps)
    
    @property
    def content(self) -> str:
        """
        Generate content string for vector embedding.
        Combines all steps into searchable text.
        """
        parts = []
        if self.task_context and self.task_context.goal:
            parts.append(f"Goal: {self.task_context.goal}")
        if self.task_context and self.task_context.input:
            parts.append(f"Input: {self.task_context.input[:200]}")  # Truncate long inputs
        for step in self.steps:
            parts.append(f"{step.type.capitalize()}: {step.content}")
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trace_id': self.trace_id,
            'steps': [s.to_dict() for s in self.steps],
            'task_context': self.task_context.to_dict() if self.task_context else None,
            'evaluation': self.evaluation.to_dict() if self.evaluation else None,
            'extracted_at': self.extracted_at.isoformat() if self.extracted_at else None,
            'session_id': self.session_id,
            'has_explicit_markup': self.has_explicit_markup,
            'artifact_ids': self.artifact_ids,
            'step_count': self.step_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningTrace':
        steps = [ReasoningStep.from_dict(s) for s in data.get('steps', [])]
        task_context = TaskContext.from_dict(data['task_context']) if data.get('task_context') else None
        evaluation = ReasoningEvaluation.from_dict(data['evaluation']) if data.get('evaluation') else None
        
        extracted_at = data.get('extracted_at')
        if isinstance(extracted_at, str):
            extracted_at = datetime.fromisoformat(extracted_at)
        elif extracted_at is None:
            extracted_at = datetime.now(timezone.utc)
            
        return cls(
            trace_id=data.get('trace_id', ''),
            steps=steps,
            task_context=task_context,
            evaluation=evaluation,
            extracted_at=extracted_at,
            session_id=data.get('session_id'),
            has_explicit_markup=data.get('has_explicit_markup', False),
            artifact_ids=data.get('artifact_ids', []),
        )
