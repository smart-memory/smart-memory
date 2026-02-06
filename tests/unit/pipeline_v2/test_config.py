"""Unit tests for PipelineConfig and nested config dataclasses."""

import pytest

from smartmemory.pipeline.config import (
    ClassifyConfig,
    CoreferenceConfig,
    EnrichConfig,
    EvolveConfig,
    ExtractionConfig,
    LLMExtractConfig,
    LinkConfig,
    PipelineConfig,
    RetryConfig,
    SimplifyConfig,
    StoreConfig,
)


class TestRetryConfig:
    """Tests for RetryConfig leaf config."""

    def test_default_construction(self):
        """Default RetryConfig has sensible defaults."""
        cfg = RetryConfig()
        assert cfg.max_retries == 2
        assert cfg.backoff_seconds == 1.0
        assert cfg.on_failure == "abort"

    def test_valid_on_failure_abort(self):
        """on_failure='abort' is valid."""
        cfg = RetryConfig(on_failure="abort")
        assert cfg.on_failure == "abort"

    def test_valid_on_failure_skip(self):
        """on_failure='skip' is valid."""
        cfg = RetryConfig(on_failure="skip")
        assert cfg.on_failure == "skip"

    def test_invalid_on_failure_raises_value_error(self):
        """Invalid on_failure value raises ValueError."""
        with pytest.raises(ValueError, match="on_failure must be 'abort' or 'skip'"):
            RetryConfig(on_failure="ignore")


class TestPipelineConfigDefaults:
    """Tests for PipelineConfig default construction."""

    def test_default_construction(self):
        """PipelineConfig with no args has expected defaults."""
        cfg = PipelineConfig()
        assert cfg.workspace_id is None
        assert cfg.mode == "sync"
        assert isinstance(cfg.retry, RetryConfig)
        assert isinstance(cfg.classify, ClassifyConfig)
        assert isinstance(cfg.coreference, CoreferenceConfig)
        assert isinstance(cfg.simplify, SimplifyConfig)
        assert isinstance(cfg.extraction, ExtractionConfig)
        assert isinstance(cfg.store, StoreConfig)
        assert isinstance(cfg.link, LinkConfig)
        assert isinstance(cfg.enrich, EnrichConfig)
        assert isinstance(cfg.evolve, EvolveConfig)

    def test_default_nested_configs(self):
        """Nested configs have their own defaults."""
        cfg = PipelineConfig()
        assert cfg.classify.content_analysis_enabled is False
        assert cfg.classify.default_confidence == 0.9
        assert cfg.coreference.enabled is True
        assert cfg.coreference.resolver == "fastcoref"
        assert cfg.extraction.llm_extract.enabled is True
        assert cfg.extraction.llm_extract.max_entities == 10
        assert cfg.link.similarity_threshold == 0.8
        assert cfg.evolve.run_evolution is True
        assert cfg.evolve.run_clustering is True


class TestPipelineConfigMode:
    """Tests for mode validation."""

    def test_valid_mode_sync(self):
        """mode='sync' is valid."""
        cfg = PipelineConfig(mode="sync")
        assert cfg.mode == "sync"

    def test_valid_mode_async(self):
        """mode='async' is valid."""
        cfg = PipelineConfig(mode="async")
        assert cfg.mode == "async"

    def test_valid_mode_preview(self):
        """mode='preview' is valid."""
        cfg = PipelineConfig(mode="preview")
        assert cfg.mode == "preview"

    def test_invalid_mode_raises_value_error(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be one of"):
            PipelineConfig(mode="batch")


class TestPipelineConfigFactories:
    """Tests for named config factory methods."""

    def test_default_factory(self):
        """PipelineConfig.default() returns a standard production config."""
        cfg = PipelineConfig.default()
        assert cfg.mode == "sync"
        assert cfg.evolve.run_evolution is True
        assert cfg.evolve.run_clustering is True

    def test_default_factory_with_workspace(self):
        """PipelineConfig.default(workspace_id=...) sets workspace."""
        cfg = PipelineConfig.default(workspace_id="ws-42")
        assert cfg.workspace_id == "ws-42"

    def test_preview_factory(self):
        """PipelineConfig.preview() returns preview mode config."""
        cfg = PipelineConfig.preview()
        assert cfg.mode == "preview"

    def test_preview_disables_evolution(self):
        """Preview mode sets evolve.run_evolution=False."""
        cfg = PipelineConfig.preview()
        assert cfg.evolve.run_evolution is False
        assert cfg.evolve.run_clustering is False

    def test_preview_factory_with_workspace(self):
        """PipelineConfig.preview(workspace_id=...) sets workspace."""
        cfg = PipelineConfig.preview(workspace_id="ws-99")
        assert cfg.workspace_id == "ws-99"
        assert cfg.mode == "preview"


class TestPipelineConfigNestedOverride:
    """Tests for overriding nested config values."""

    def test_override_llm_extract_model(self):
        """Nested config override via constructor."""
        cfg = PipelineConfig(extraction=ExtractionConfig(llm_extract=LLMExtractConfig(model="gpt-4o-mini")))
        assert cfg.extraction.llm_extract.model == "gpt-4o-mini"

    def test_override_preserves_sibling_defaults(self):
        """Overriding one nested field preserves siblings' defaults."""
        cfg = PipelineConfig(extraction=ExtractionConfig(llm_extract=LLMExtractConfig(model="gpt-4o-mini")))
        # Entity ruler should still have its default
        assert cfg.extraction.entity_ruler.enabled is True
        # Other top-level configs should be untouched
        assert cfg.classify.default_confidence == 0.9

    def test_override_retry_config(self):
        """Override retry configuration."""
        cfg = PipelineConfig(retry=RetryConfig(max_retries=5, on_failure="skip"))
        assert cfg.retry.max_retries == 5
        assert cfg.retry.on_failure == "skip"

    def test_override_evolve_config(self):
        """Override evolve configuration."""
        cfg = PipelineConfig(evolve=EvolveConfig(run_evolution=False))
        assert cfg.evolve.run_evolution is False
        assert cfg.evolve.run_clustering is True  # Default preserved


class TestPipelineConfigSerialization:
    """Tests for to_dict() / from_dict() round-trip (via MemoryBaseModel)."""

    def test_to_dict_produces_dict(self):
        """to_dict() returns a plain dict."""
        cfg = PipelineConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["mode"] == "sync"

    def test_to_dict_nested_configs_are_dicts(self):
        """Nested config objects are serialized as dicts."""
        cfg = PipelineConfig()
        d = cfg.to_dict()
        assert isinstance(d["retry"], dict)
        assert isinstance(d["extraction"], dict)
        assert isinstance(d["extraction"]["llm_extract"], dict)

    def test_round_trip_default_config(self):
        """Default config survives to_dict() -> from_dict()."""
        original = PipelineConfig()
        d = original.to_dict()
        restored = PipelineConfig.from_dict(d)
        assert restored.mode == original.mode
        assert restored.workspace_id == original.workspace_id

    def test_round_trip_with_overrides(self):
        """Config with overrides survives to_dict() -> from_dict()."""
        original = PipelineConfig(
            workspace_id="ws-test",
            mode="preview",
            retry=RetryConfig(max_retries=5, on_failure="skip"),
        )
        d = original.to_dict()
        restored = PipelineConfig.from_dict(d)
        assert restored.workspace_id == "ws-test"
        assert restored.mode == "preview"

    def test_leaf_config_round_trip(self):
        """A leaf config (RetryConfig) survives to_dict() -> from_dict()."""
        original = RetryConfig(max_retries=3, backoff_seconds=2.5, on_failure="skip")
        d = original.to_dict()
        restored = RetryConfig.from_dict(d)
        assert restored.max_retries == 3
        assert restored.backoff_seconds == 2.5
        assert restored.on_failure == "skip"
