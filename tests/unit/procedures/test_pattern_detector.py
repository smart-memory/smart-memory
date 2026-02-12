"""
Unit tests for PatternDetector and related components.
"""

import pytest
from datetime import datetime, timezone, timedelta

# Import directly to avoid circular import issues
from smartmemory.models.memory_item import MemoryItem
from smartmemory.procedures.models import (
    ProcedureCandidate,
    CandidateScores,
    PatternDetectorConfig,
)
from smartmemory.procedures.pattern_detector import PatternDetector
from smartmemory.procedures.candidate_scorer import CandidateScorer
from smartmemory.procedures.candidate_namer import CandidateNamer


class TestPatternDetectorConfig:
    """Tests for PatternDetectorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PatternDetectorConfig()

        assert config.cluster_threshold == 0.75
        assert config.min_cluster_size == 3
        assert config.min_score == 0.6
        assert config.days_back == 30

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1.0."""
        config = PatternDetectorConfig(
            frequency_weight=1.0,
            consistency_weight=1.0,
            recency_weight=1.0,
            agent_workflow_weight=1.0,
        )

        total = (
            config.frequency_weight + config.consistency_weight + config.recency_weight + config.agent_workflow_weight
        )
        assert abs(total - 1.0) < 0.01

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PatternDetectorConfig(
            cluster_threshold=0.8,
            min_cluster_size=5,
            min_score=0.7,
            days_back=14,
        )

        assert config.cluster_threshold == 0.8
        assert config.min_cluster_size == 5
        assert config.min_score == 0.7
        assert config.days_back == 14


class TestCandidateScorer:
    """Tests for CandidateScorer."""

    @pytest.fixture
    def scorer(self):
        """Create a scorer with default config."""
        return CandidateScorer()

    @pytest.fixture
    def sample_cluster(self):
        """Create a sample cluster of similar items."""
        now = datetime.now(timezone.utc)
        return [
            MemoryItem(
                content="Process API request and validate input",
                memory_type="working",
                transaction_time=now - timedelta(hours=1),
                metadata={"skills": ["api_handling", "validation"], "tools": ["http_client"]},
            ),
            MemoryItem(
                content="Process API response and handle errors",
                memory_type="working",
                transaction_time=now - timedelta(hours=2),
                metadata={"skills": ["api_handling", "error_handling"], "tools": ["http_client"]},
            ),
            MemoryItem(
                content="Process API data and transform results",
                memory_type="working",
                transaction_time=now - timedelta(hours=3),
                metadata={"skills": ["api_handling", "data_transform"], "tools": ["http_client"]},
            ),
        ]

    def test_score_cluster_basic(self, scorer, sample_cluster):
        """Test basic cluster scoring."""
        scores = scorer.score_cluster(sample_cluster, total_items=10)

        assert isinstance(scores, CandidateScores)
        assert 0 <= scores.recommendation_score <= 1
        assert 0 <= scores.frequency_score <= 1
        assert 0 <= scores.consistency_score <= 1
        assert 0 <= scores.recency_score <= 1
        assert 0 <= scores.agent_workflow_score <= 1

    def test_score_empty_cluster(self, scorer):
        """Test scoring an empty cluster."""
        scores = scorer.score_cluster([], total_items=10)

        assert scores.recommendation_score == 0.0
        assert scores.frequency_score == 0.0

    def test_score_single_item(self, scorer):
        """Test scoring a single-item cluster."""
        item = MemoryItem(
            content="Single item",
            memory_type="working",
        )
        scores = scorer.score_cluster([item], total_items=10)

        # Single item should have perfect consistency
        assert scores.consistency_score == 1.0

    def test_frequency_score_scaling(self, scorer, sample_cluster):
        """Test that frequency score scales with cluster size."""
        scores_small = scorer.score_cluster(sample_cluster[:2], total_items=100)
        scores_large = scorer.score_cluster(sample_cluster, total_items=100)

        # Larger cluster should have higher frequency score
        assert scores_large.frequency_score >= scores_small.frequency_score

    def test_recency_score_decay(self, scorer):
        """Test that recency score decays with older items."""
        now = datetime.now(timezone.utc)

        recent_cluster = [
            MemoryItem(
                content="Recent item",
                memory_type="working",
                transaction_time=now - timedelta(hours=1),
            )
        ]

        old_cluster = [
            MemoryItem(
                content="Old item",
                memory_type="working",
                transaction_time=now - timedelta(days=20),
            )
        ]

        recent_scores = scorer.score_cluster(recent_cluster, total_items=10)
        old_scores = scorer.score_cluster(old_cluster, total_items=10)

        # Recent items should have higher recency score
        assert recent_scores.recency_score > old_scores.recency_score

    def test_agent_workflow_score_with_metadata(self, scorer, sample_cluster):
        """Test agent workflow scoring with skills/tools metadata."""
        scores = scorer.score_cluster(sample_cluster, total_items=10)

        # Items with skills/tools should have positive workflow score
        assert scores.agent_workflow_score > 0

    def test_agent_workflow_score_without_metadata(self, scorer):
        """Test agent workflow scoring without explicit metadata."""
        cluster = [
            MemoryItem(content="Execute API call and process response", memory_type="working"),
            MemoryItem(content="Run workflow step and execute task", memory_type="working"),
        ]
        scores = scorer.score_cluster(cluster, total_items=10)

        # Should still detect workflow patterns from content
        assert scores.agent_workflow_score >= 0


class TestCandidateNamer:
    """Tests for CandidateNamer."""

    @pytest.fixture
    def namer(self):
        """Create a namer instance."""
        return CandidateNamer()

    @pytest.fixture
    def sample_cluster(self):
        """Create a sample cluster for naming."""
        return [
            MemoryItem(
                content="Handle database query errors and retry",
                memory_type="working",
                metadata={"skills": ["error_handling"], "tools": ["database"]},
            ),
            MemoryItem(
                content="Process database response and handle exceptions",
                memory_type="working",
                metadata={"skills": ["error_handling"], "tools": ["database"]},
            ),
            MemoryItem(
                content="Manage database connection errors",
                memory_type="working",
                metadata={"skills": ["error_handling"], "tools": ["database"]},
            ),
        ]

    def test_generate_name_and_description(self, namer, sample_cluster):
        """Test basic name and description generation."""
        name, description = namer.generate_name_and_description(sample_cluster)

        assert isinstance(name, str)
        assert isinstance(description, str)
        assert len(name) > 0
        assert len(description) > 0

    def test_name_length_limit(self, namer, sample_cluster):
        """Test that name respects length limit."""
        name, _ = namer.generate_name_and_description(sample_cluster)

        assert len(name) <= namer.max_name_length

    def test_description_length_limit(self, namer, sample_cluster):
        """Test that description respects length limit."""
        _, description = namer.generate_name_and_description(sample_cluster)

        assert len(description) <= namer.max_description_length

    def test_empty_cluster(self, namer):
        """Test handling of empty cluster."""
        name, description = namer.generate_name_and_description([])

        assert name == "Unnamed Pattern"
        assert description == "No description available"

    def test_get_common_skills(self, namer, sample_cluster):
        """Test extracting common skills."""
        skills = namer.get_common_skills(sample_cluster)

        assert "error_handling" in skills

    def test_get_common_tools(self, namer, sample_cluster):
        """Test extracting common tools."""
        tools = namer.get_common_tools(sample_cluster)

        assert "database" in tools


class TestPatternDetector:
    """Tests for PatternDetector."""

    @pytest.fixture
    def detector(self):
        """Create a detector without SmartMemory (for direct item analysis)."""
        # Use a lower threshold for testing since test items may not be highly similar
        config = PatternDetectorConfig(cluster_threshold=0.5)
        return PatternDetector(smart_memory=None, config=config)

    @pytest.fixture
    def similar_items(self):
        """Create items with similar content that should cluster together."""
        now = datetime.now(timezone.utc)
        return [
            # Cluster 1: API error handling (very similar)
            MemoryItem(
                content="When receiving a 401 authentication error from the API, refresh the token and retry",
                memory_type="working",
                transaction_time=now - timedelta(hours=1),
                metadata={"skills": ["error_handling", "auth"], "tools": ["api_client"]},
            ),
            MemoryItem(
                content="When getting a 401 auth error from API, refresh token and retry the request",
                memory_type="working",
                transaction_time=now - timedelta(hours=2),
                metadata={"skills": ["error_handling", "auth"], "tools": ["api_client"]},
            ),
            MemoryItem(
                content="On 401 authentication error from API endpoint, refresh the auth token and retry",
                memory_type="working",
                transaction_time=now - timedelta(hours=3),
                metadata={"skills": ["error_handling", "auth"], "tools": ["api_client"]},
            ),
            MemoryItem(
                content="Handle 401 error from API by refreshing authentication token and retrying",
                memory_type="working",
                transaction_time=now - timedelta(hours=4),
                metadata={"skills": ["error_handling", "auth"], "tools": ["api_client"]},
            ),
            # Cluster 2: Database validation (different topic)
            MemoryItem(
                content="Validate user input before inserting into database",
                memory_type="working",
                transaction_time=now - timedelta(hours=5),
                metadata={"skills": ["validation"], "tools": ["database"]},
            ),
            MemoryItem(
                content="Check and validate input data before database insertion",
                memory_type="working",
                transaction_time=now - timedelta(hours=6),
                metadata={"skills": ["validation"], "tools": ["database"]},
            ),
            MemoryItem(
                content="Validate input parameters before inserting into the database",
                memory_type="working",
                transaction_time=now - timedelta(hours=7),
                metadata={"skills": ["validation"], "tools": ["database"]},
            ),
        ]

    def test_detect_candidates_with_items(self, detector, similar_items):
        """Test candidate detection with provided items."""
        candidates = detector.detect_candidates(
            items=similar_items,
            min_cluster_size=3,
            min_score=0.0,  # Accept any score for testing
        )

        assert len(candidates) >= 1
        for candidate in candidates:
            assert isinstance(candidate, ProcedureCandidate)
            assert candidate.item_count >= 3

    def test_detect_candidates_empty_items(self, detector):
        """Test detection with empty item list."""
        candidates = detector.detect_candidates(items=[])

        assert len(candidates) == 0

    def test_detect_candidates_sorting(self, detector, similar_items):
        """Test that candidates are sorted by recommendation score."""
        candidates = detector.detect_candidates(
            items=similar_items,
            min_cluster_size=3,
            min_score=0.0,
        )

        if len(candidates) > 1:
            scores = [c.scores.recommendation_score for c in candidates]
            assert scores == sorted(scores, reverse=True)

    def test_detect_candidates_limit(self, detector, similar_items):
        """Test limiting number of candidates."""
        candidates = detector.detect_candidates(
            items=similar_items,
            min_cluster_size=2,
            min_score=0.0,
            limit=1,
        )

        assert len(candidates) <= 1

    def test_detect_candidates_min_score_filter(self, detector, similar_items):
        """Test filtering by minimum score."""
        # Get all candidates
        all_candidates = detector.detect_candidates(
            items=similar_items,
            min_cluster_size=2,
            min_score=0.0,
        )

        # Get filtered candidates
        filtered_candidates = detector.detect_candidates(
            items=similar_items,
            min_cluster_size=2,
            min_score=0.8,  # High threshold
        )

        # Filtered should be subset
        assert len(filtered_candidates) <= len(all_candidates)

    def test_candidate_structure(self, detector, similar_items):
        """Test that candidates have correct structure."""
        candidates = detector.detect_candidates(
            items=similar_items,
            min_cluster_size=3,
            min_score=0.0,
        )

        if candidates:
            candidate = candidates[0]

            assert candidate.cluster_id is not None
            assert candidate.suggested_name is not None
            assert candidate.suggested_description is not None
            assert candidate.item_count >= 3
            assert len(candidate.sample_item_ids) <= 5

    def test_candidate_to_dict(self, detector, similar_items):
        """Test candidate serialization to dict."""
        candidates = detector.detect_candidates(
            items=similar_items,
            min_cluster_size=3,
            min_score=0.0,
        )

        if candidates:
            candidate_dict = candidates[0].to_dict()

            assert "cluster_id" in candidate_dict
            assert "suggested_name" in candidate_dict
            assert "suggested_description" in candidate_dict
            assert "representative_content" in candidate_dict
            assert "item_count" in candidate_dict
            assert "scores" in candidate_dict
            assert "date_range" in candidate_dict

            # Check scores structure
            scores = candidate_dict["scores"]
            assert "recommendation_score" in scores
            assert "frequency_score" in scores
            assert "consistency_score" in scores
            assert "recency_score" in scores
            assert "agent_workflow_score" in scores

    def test_cluster_id_stability(self, detector, similar_items):
        """Test that cluster IDs are stable across multiple detection calls.

        This is critical for the promote/dismiss flow - users need to be able
        to list candidates and then promote them with the same cluster_id.
        """
        # First detection
        candidates_1 = detector.detect_candidates(
            items=similar_items,
            min_cluster_size=3,
            min_score=0.0,
        )

        # Second detection with same items
        candidates_2 = detector.detect_candidates(
            items=similar_items,
            min_cluster_size=3,
            min_score=0.0,
        )

        assert len(candidates_1) == len(candidates_2)

        # Cluster IDs should be identical
        ids_1 = sorted([c.cluster_id for c in candidates_1])
        ids_2 = sorted([c.cluster_id for c in candidates_2])

        assert ids_1 == ids_2, "Cluster IDs should be stable across calls"

        # Also verify the IDs are deterministic hashes (32 hex chars)
        for candidate in candidates_1:
            assert len(candidate.cluster_id) == 32
            assert all(c in "0123456789abcdef" for c in candidate.cluster_id)


class TestCandidateScores:
    """Tests for CandidateScores dataclass."""

    def test_default_scores(self):
        """Test default score values."""
        scores = CandidateScores()

        assert scores.recommendation_score == 0.0
        assert scores.frequency_score == 0.0
        assert scores.consistency_score == 0.0
        assert scores.recency_score == 0.0
        assert scores.agent_workflow_score == 0.0

    def test_custom_scores(self):
        """Test custom score values."""
        scores = CandidateScores(
            recommendation_score=0.85,
            frequency_score=0.8,
            consistency_score=0.9,
            recency_score=0.75,
            agent_workflow_score=0.7,
        )

        assert scores.recommendation_score == 0.85
        assert scores.frequency_score == 0.8


class TestProcedureCandidate:
    """Tests for ProcedureCandidate dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now(timezone.utc)
        candidate = ProcedureCandidate(
            cluster_id="test-id",
            suggested_name="Test Pattern",
            suggested_description="A test pattern",
            representative_content="Test content",
            item_count=5,
            scores=CandidateScores(recommendation_score=0.85),
            common_skills=["skill1"],
            common_tools=["tool1"],
            sample_item_ids=["id1", "id2"],
            earliest_date=now - timedelta(days=7),
            latest_date=now,
        )

        result = candidate.to_dict()

        assert result["cluster_id"] == "test-id"
        assert result["suggested_name"] == "Test Pattern"
        assert result["item_count"] == 5
        assert result["scores"]["recommendation_score"] == 0.85
        assert "date_range" in result
