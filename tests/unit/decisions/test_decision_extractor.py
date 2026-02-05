"""Unit tests for DecisionExtractor."""

import pytest

from smartmemory.models.reasoning import ReasoningStep, ReasoningTrace
from smartmemory.plugins.extractors.decision import DecisionExtractor, DecisionExtractorConfig


@pytest.fixture
def extractor():
    """Create a DecisionExtractor with default config."""
    return DecisionExtractor()


@pytest.fixture
def custom_extractor():
    """Create a DecisionExtractor with custom config."""
    config = DecisionExtractorConfig(min_confidence=0.6, min_content_length=10)
    return DecisionExtractor(config=config)


class TestMetadata:
    """Test plugin metadata."""

    def test_metadata_name(self):
        meta = DecisionExtractor.metadata()
        assert meta.name == "decision"

    def test_metadata_type(self):
        meta = DecisionExtractor.metadata()
        assert meta.plugin_type == "extractor"

    def test_metadata_tags(self):
        meta = DecisionExtractor.metadata()
        assert "decision" in meta.tags


class TestExtractFromText:
    """Test decision extraction from raw text."""

    def test_empty_text(self, extractor):
        result = extractor.extract("")
        assert result["decisions"] == []
        assert result["entities"] == []

    def test_short_text(self, extractor):
        result = extractor.extract("ok")
        assert result["decisions"] == []

    def test_no_decisions(self, extractor):
        result = extractor.extract("The weather is sunny today and the sky is blue.")
        assert result["decisions"] == []

    def test_explicit_preference(self, extractor):
        text = "I prefer Python over JavaScript for backend development."
        result = extractor.extract(text)
        assert len(result["decisions"]) >= 1
        d = result["decisions"][0]
        assert d.decision_type == "preference"
        assert "Python" in d.content or "prefer" in d.content.lower()

    def test_explicit_decision(self, extractor):
        text = "I decided to use React for the frontend because of its ecosystem."
        result = extractor.extract(text)
        assert len(result["decisions"]) >= 1
        d = result["decisions"][0]
        assert d.decision_type == "choice"

    def test_explicit_belief(self, extractor):
        text = "I believe that microservices are better for large teams."
        result = extractor.extract(text)
        assert len(result["decisions"]) >= 1
        d = result["decisions"][0]
        assert d.decision_type == "belief"

    def test_explicit_conclusion(self, extractor):
        text = "Therefore, the root cause of the bug is a race condition in the queue."
        result = extractor.extract(text)
        assert len(result["decisions"]) >= 1
        d = result["decisions"][0]
        assert d.decision_type == "inference"

    def test_explicit_policy(self, extractor):
        text = "We should always write tests before implementation code."
        result = extractor.extract(text)
        assert len(result["decisions"]) >= 1
        d = result["decisions"][0]
        assert d.decision_type == "policy"

    def test_multiple_decisions(self, extractor):
        text = (
            "I prefer dark mode for all applications. "
            "I decided to use VS Code as my editor. "
            "I believe TypeScript is worth the overhead."
        )
        result = extractor.extract(text)
        assert len(result["decisions"]) >= 2

    def test_decisions_have_ids(self, extractor):
        text = "I prefer Python for data science work."
        result = extractor.extract(text)
        for d in result["decisions"]:
            assert d.decision_id.startswith("dec_")
            assert len(d.decision_id) == 16

    def test_decisions_have_source_type(self, extractor):
        text = "I decided to use PostgreSQL."
        result = extractor.extract(text)
        for d in result["decisions"]:
            assert d.source_type == "inferred"

    def test_returns_standard_keys(self, extractor):
        result = extractor.extract("I prefer dark mode.")
        assert "entities" in result
        assert "relations" in result
        assert "decisions" in result


class TestExtractFromTrace:
    """Test decision extraction from ReasoningTrace."""

    def test_extract_decision_steps(self, extractor):
        trace = ReasoningTrace(
            trace_id="trace_abc123",
            steps=[
                ReasoningStep(type="thought", content="Let me analyze the options"),
                ReasoningStep(type="decision", content="I'll use FastAPI for the API layer"),
                ReasoningStep(type="conclusion", content="FastAPI is the best choice for performance"),
            ],
        )
        decisions = extractor.extract_from_trace(trace)
        assert len(decisions) == 2  # decision + conclusion steps

    def test_skips_non_decision_steps(self, extractor):
        trace = ReasoningTrace(
            trace_id="trace_abc123",
            steps=[
                ReasoningStep(type="thought", content="Let me think about this"),
                ReasoningStep(type="action", content="Running the benchmark"),
                ReasoningStep(type="observation", content="Results show 2x faster"),
            ],
        )
        decisions = extractor.extract_from_trace(trace)
        assert len(decisions) == 0

    def test_links_to_trace(self, extractor):
        trace = ReasoningTrace(
            trace_id="trace_xyz789",
            steps=[
                ReasoningStep(type="decision", content="Use Redis for caching"),
            ],
        )
        decisions = extractor.extract_from_trace(trace)
        assert len(decisions) == 1
        assert decisions[0].source_trace_id == "trace_xyz789"
        assert decisions[0].source_type == "reasoning"

    def test_links_to_session(self, extractor):
        trace = ReasoningTrace(
            trace_id="trace_abc",
            session_id="sess_123",
            steps=[
                ReasoningStep(type="conclusion", content="The bug is in the auth module"),
            ],
        )
        decisions = extractor.extract_from_trace(trace)
        assert decisions[0].source_session_id == "sess_123"

    def test_empty_trace(self, extractor):
        trace = ReasoningTrace(trace_id="trace_empty", steps=[])
        decisions = extractor.extract_from_trace(trace)
        assert decisions == []

    def test_short_content_skipped(self, extractor):
        trace = ReasoningTrace(
            trace_id="trace_short",
            steps=[
                ReasoningStep(type="decision", content="ok"),
            ],
        )
        decisions = extractor.extract_from_trace(trace)
        assert len(decisions) == 0

    def test_decision_type_classified(self, extractor):
        trace = ReasoningTrace(
            trace_id="trace_pref",
            steps=[
                ReasoningStep(type="decision", content="I prefer using dark mode on all devices"),
            ],
        )
        decisions = extractor.extract_from_trace(trace)
        assert len(decisions) == 1
        assert decisions[0].decision_type == "preference"

    def test_decisions_have_generated_ids(self, extractor):
        trace = ReasoningTrace(
            trace_id="trace_ids",
            steps=[
                ReasoningStep(type="decision", content="Use PostgreSQL for data storage"),
            ],
        )
        decisions = extractor.extract_from_trace(trace)
        assert decisions[0].decision_id.startswith("dec_")


class TestClassifyDecisionType:
    """Test keyword-based decision type classification."""

    def test_preference_keywords(self, extractor):
        assert extractor._classify_decision_type("I prefer dark mode") == "preference"
        assert extractor._classify_decision_type("my favorite tool is vim") == "preference"
        assert extractor._classify_decision_type("I like using Python") == "preference"

    def test_choice_keywords(self, extractor):
        assert extractor._classify_decision_type("I decided to use React") == "choice"
        assert extractor._classify_decision_type("I chose PostgreSQL") == "choice"
        assert extractor._classify_decision_type("I'll go with option B") == "choice"

    def test_belief_keywords(self, extractor):
        assert extractor._classify_decision_type("I believe TDD is important") == "belief"
        assert extractor._classify_decision_type("I think microservices work well") == "belief"

    def test_policy_keywords(self, extractor):
        assert extractor._classify_decision_type("We should always test first") == "policy"
        assert extractor._classify_decision_type("Never deploy on Fridays") == "policy"
        assert extractor._classify_decision_type("The rule is to review all PRs") == "policy"

    def test_inference_keywords(self, extractor):
        assert extractor._classify_decision_type("Therefore the bug is here") == "inference"
        assert extractor._classify_decision_type("This means the API is slow") == "inference"
        assert extractor._classify_decision_type("In conclusion, we need caching") == "inference"

    def test_classification_keywords(self, extractor):
        assert extractor._classify_decision_type("This is a performance issue") == "classification"
        assert extractor._classify_decision_type("This falls into the security category") == "classification"

    def test_default_inference(self, extractor):
        assert extractor._classify_decision_type("Some unclassified statement about code") == "inference"


class TestDecisionPatterns:
    """Test the regex patterns for decision detection."""

    def test_preference_pattern(self, extractor):
        patterns = [
            "I prefer tabs over spaces",
            "My preference is for functional components",
            "I favor composition over inheritance",
        ]
        for text in patterns:
            result = extractor.extract(text)
            assert len(result["decisions"]) >= 1, f"Failed to detect: {text}"

    def test_decision_pattern(self, extractor):
        patterns = [
            "I decided to refactor the authentication module",
            "My decision is to use GraphQL instead of REST",
        ]
        for text in patterns:
            result = extractor.extract(text)
            assert len(result["decisions"]) >= 1, f"Failed to detect: {text}"

    def test_conclusion_pattern(self, extractor):
        patterns = [
            "Therefore, the root cause is a memory leak in the cache",
            "In conclusion, we should migrate to Kubernetes",
            "Thus, the best approach is to use event sourcing",
        ]
        for text in patterns:
            result = extractor.extract(text)
            assert len(result["decisions"]) >= 1, f"Failed to detect: {text}"

    def test_belief_pattern(self, extractor):
        patterns = [
            "I believe strongly that testing is essential for quality software",
            "I think that monorepos work better for small teams with shared code",
        ]
        for text in patterns:
            result = extractor.extract(text)
            assert len(result["decisions"]) >= 1, f"Failed to detect: {text}"


class TestConfig:
    """Test configuration options."""

    def test_default_config(self):
        config = DecisionExtractorConfig()
        assert config.min_confidence == 0.7
        assert config.min_content_length == 20

    def test_custom_min_confidence(self):
        config = DecisionExtractorConfig(min_confidence=0.9)
        extractor = DecisionExtractor(config=config)
        assert extractor.cfg.min_confidence == 0.9

    def test_custom_min_content_length(self, custom_extractor):
        assert custom_extractor.cfg.min_content_length == 10
