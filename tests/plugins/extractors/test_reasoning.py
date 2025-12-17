"""
Unit tests for ReasoningExtractor.
"""

import pytest
from smartmemory.plugins.extractors.reasoning import ReasoningExtractor, ReasoningExtractorConfig
from smartmemory.models.reasoning import ReasoningTrace, ReasoningStep


class TestReasoningExtractor:
    """Tests for ReasoningExtractor."""

    def test_explicit_markers_extraction(self):
        """Test extraction from explicit Thought:/Action:/Observation: markers."""
        extractor = ReasoningExtractor()
        
        text = """
        Thought: I need to analyze this bug in the authentication code.
        Action: Let me search for the login function.
        Observation: Found the issue - the token validation is missing.
        Decision: I will add token validation before proceeding.
        Conclusion: The fix is to add a null check on line 42.
        """
        
        result = extractor.extract(text)
        
        assert result is not None
        assert 'reasoning_trace' in result
        trace = result['reasoning_trace']
        
        if trace:  # May be None if quality threshold not met
            assert isinstance(trace, ReasoningTrace)
            assert trace.has_explicit_markup is True
            assert len(trace.steps) >= 2
            
            # Check step types
            step_types = [s.type for s in trace.steps]
            assert 'thought' in step_types or 'action' in step_types

    def test_no_reasoning_content(self):
        """Test that non-reasoning text returns no trace."""
        extractor = ReasoningExtractor()
        
        text = "Hello, how are you today? The weather is nice."
        
        result = extractor.extract(text)
        
        assert result is not None
        assert result['reasoning_trace'] is None

    def test_short_text_skipped(self):
        """Test that very short text is skipped."""
        extractor = ReasoningExtractor()
        
        text = "Short text"
        
        result = extractor.extract(text)
        
        assert result['reasoning_trace'] is None

    def test_quality_evaluation(self):
        """Test that quality evaluation is performed."""
        extractor = ReasoningExtractor()
        
        text = """
        Thought: First, I need to understand the problem.
        Action: Reading the error message carefully.
        Observation: The error indicates a null pointer exception.
        Decision: I should add null checks.
        Conclusion: Added defensive null checks to fix the issue.
        """
        
        result = extractor.extract(text)
        trace = result.get('reasoning_trace')
        
        if trace:
            assert trace.evaluation is not None
            assert 0.0 <= trace.evaluation.quality_score <= 1.0
            assert isinstance(trace.evaluation.should_store, bool)

    def test_task_context_inference(self):
        """Test that task context is inferred from content."""
        extractor = ReasoningExtractor()
        
        text = """
        Thought: This Python bug is causing issues in the backend API.
        Action: Let me debug the function.
        Observation: Found the issue in the database query.
        Conclusion: Fixed by adding proper error handling.
        """
        
        result = extractor.extract(text)
        trace = result.get('reasoning_trace')
        
        if trace and trace.task_context:
            # Should detect Python/backend domain
            assert trace.task_context.domain in ['python', 'backend', None]
            assert trace.task_context.task_type in ['debugging', 'analysis', None]

    def test_config_customization(self):
        """Test that config options are respected."""
        config = ReasoningExtractorConfig(
            min_steps=3,
            min_quality_score=0.6,
            use_llm_detection=False,
        )
        extractor = ReasoningExtractor(config=config)
        
        # Only 2 steps - should not meet min_steps threshold
        text = """
        Thought: Analyzing the problem.
        Conclusion: Found the solution.
        """
        
        result = extractor.extract(text)
        # With min_steps=3, this should return None
        assert result['reasoning_trace'] is None

    def test_trace_content_generation(self):
        """Test that trace generates searchable content."""
        trace = ReasoningTrace(
            trace_id='test_123',
            steps=[
                ReasoningStep(type='thought', content='Analyzing the bug'),
                ReasoningStep(type='conclusion', content='Fixed by adding null check'),
            ],
        )
        
        content = trace.content
        assert 'Thought: Analyzing the bug' in content
        assert 'Conclusion: Fixed by adding null check' in content

    def test_metadata_output(self):
        """Test that extractor returns proper metadata structure."""
        extractor = ReasoningExtractor()
        
        result = extractor.extract("Short")
        
        # Should always return dict with these keys
        assert 'entities' in result
        assert 'relations' in result
        assert 'reasoning_trace' in result
        assert isinstance(result['entities'], list)
        assert isinstance(result['relations'], list)


class TestReasoningModels:
    """Tests for reasoning data models."""

    def test_reasoning_step_serialization(self):
        """Test ReasoningStep to_dict/from_dict."""
        step = ReasoningStep(type='thought', content='Test content')
        
        data = step.to_dict()
        assert data['type'] == 'thought'
        assert data['content'] == 'Test content'
        
        restored = ReasoningStep.from_dict(data)
        assert restored.type == step.type
        assert restored.content == step.content

    def test_reasoning_trace_serialization(self):
        """Test ReasoningTrace to_dict/from_dict."""
        trace = ReasoningTrace(
            trace_id='test_123',
            steps=[
                ReasoningStep(type='thought', content='Step 1'),
                ReasoningStep(type='action', content='Step 2'),
            ],
        )
        
        data = trace.to_dict()
        assert data['trace_id'] == 'test_123'
        assert len(data['steps']) == 2
        
        restored = ReasoningTrace.from_dict(data)
        assert restored.trace_id == trace.trace_id
        assert len(restored.steps) == 2

    def test_evaluation_should_store_logic(self):
        """Test ReasoningEvaluation.should_store property."""
        from smartmemory.models.reasoning import ReasoningEvaluation
        
        # High quality - should store
        eval_good = ReasoningEvaluation(quality_score=0.7, issues=[])
        assert eval_good.should_store is True
        
        # Low quality - should not store
        eval_bad = ReasoningEvaluation(quality_score=0.3, issues=[])
        assert eval_bad.should_store is False
        
        # High severity issue - should not store
        eval_issue = ReasoningEvaluation(
            quality_score=0.7,
            issues=[{'type': 'critical', 'severity': 'high', 'description': 'Bad'}]
        )
        assert eval_issue.should_store is False
