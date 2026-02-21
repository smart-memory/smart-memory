"""Additional coverage tests for smartmemory_lite.factory.

Focuses on:
- lite_context() resets VectorStore backend on normal exit
- lite_context() resets VectorStore backend even when body raises
- _apply_lite_pipeline_profile patches both _build_pipeline_config and _create_pipeline_runner
- create_lite_memory creates the data directory if it doesn't exist
- create_lite_memory respects custom data_dir
- The patched _build_pipeline_config sets coreference.enabled=False, llm_extract.enabled=False,
  and enricher_names=["basic_enricher"]
- The patched _create_pipeline_runner sets runner._metrics=None
"""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# Ensure the factory module can be imported (requires smartmemory_lite installed)
try:
    from smartmemory_lite.factory import create_lite_memory, lite_context, _apply_lite_pipeline_profile
    _LITE_AVAILABLE = True
except ImportError:
    _LITE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _LITE_AVAILABLE, reason="smartmemory-lite not installed")


# ── lite_context cleanup contract ────────────────────────────────────────────
# NOTE: These tests verify the contract via spy rather than checking the module
# global directly. The unit autouse fixture patches VectorStore, so the real
# module global (_DEFAULT_BACKEND) is not written. We test via the mock call
# count that set_default_backend(None) is always called in the finally block.

def test_lite_context_calls_set_default_backend_none_on_normal_exit(tmp_path):
    """lite_context() calls VectorStore.set_default_backend(None) on normal exit."""
    from unittest.mock import call
    from smartmemory.stores.vector.vector_store import VectorStore

    with lite_context(str(tmp_path)) as memory:
        assert memory is not None

    # The finally block must have called set_default_backend(None)
    # (mock is installed by autouse unit_isolation_patches fixture)
    set_calls = [c for c in VectorStore.set_default_backend.call_args_list if c == call(None)]
    assert len(set_calls) >= 1, (
        "lite_context must call VectorStore.set_default_backend(None) on exit"
    )


def test_lite_context_resets_backend_even_on_exception(tmp_path):
    """lite_context() calls set_default_backend(None) even when the body raises."""
    from unittest.mock import call
    from smartmemory.stores.vector.vector_store import VectorStore

    with pytest.raises(RuntimeError, match="intentional"):
        with lite_context(str(tmp_path)):
            raise RuntimeError("intentional test error")

    set_calls = [c for c in VectorStore.set_default_backend.call_args_list if c == call(None)]
    assert len(set_calls) >= 1, (
        "lite_context must call VectorStore.set_default_backend(None) even on exception"
    )


# ── create_lite_memory creates data directory ─────────────────────────────────

def test_create_lite_memory_creates_data_dir(tmp_path):
    """create_lite_memory() creates the data directory if it doesn't exist."""
    from smartmemory.stores.vector.vector_store import VectorStore
    nested_dir = tmp_path / "nested" / "subdir"
    assert not nested_dir.exists()
    try:
        memory = create_lite_memory(str(nested_dir))
        assert nested_dir.exists(), "data directory should be created"
    finally:
        VectorStore.set_default_backend(None)


def test_create_lite_memory_creates_db_file(tmp_path):
    """create_lite_memory() creates memory.db inside data_dir."""
    from smartmemory.stores.vector.vector_store import VectorStore
    try:
        memory = create_lite_memory(str(tmp_path))
        db_path = tmp_path / "memory.db"
        assert db_path.exists(), f"Expected memory.db at {db_path}"
    finally:
        VectorStore.set_default_backend(None)


# ── _apply_lite_pipeline_profile correctness ──────────────────────────────────

def test_apply_lite_pipeline_profile_disables_coreference(tmp_path):
    """The patched _build_pipeline_config sets coreference.enabled=False."""
    from smartmemory.stores.vector.vector_store import VectorStore
    try:
        memory = create_lite_memory(str(tmp_path))
        config = memory._build_pipeline_config()
        assert config.coreference.enabled is False, (
            "coreference must be disabled in lite pipeline profile"
        )
    finally:
        VectorStore.set_default_backend(None)


def test_apply_lite_pipeline_profile_disables_llm_extract(tmp_path):
    """The patched _build_pipeline_config sets extraction.llm_extract.enabled=False."""
    from smartmemory.stores.vector.vector_store import VectorStore
    try:
        memory = create_lite_memory(str(tmp_path))
        config = memory._build_pipeline_config()
        assert config.extraction.llm_extract.enabled is False, (
            "llm_extract must be disabled in lite pipeline profile"
        )
    finally:
        VectorStore.set_default_backend(None)


def test_apply_lite_pipeline_profile_limits_enrichers(tmp_path):
    """The patched _build_pipeline_config limits enrichers to basic_enricher only."""
    from smartmemory.stores.vector.vector_store import VectorStore
    try:
        memory = create_lite_memory(str(tmp_path))
        config = memory._build_pipeline_config()
        assert config.enrich.enricher_names == ["basic_enricher"], (
            "only basic_enricher should run in lite mode — no HTTP enrichers"
        )
    finally:
        VectorStore.set_default_backend(None)


def test_apply_lite_pipeline_profile_disables_wikidata(tmp_path):
    """The patched _build_pipeline_config disables wikidata grounding."""
    from smartmemory.stores.vector.vector_store import VectorStore
    try:
        memory = create_lite_memory(str(tmp_path))
        config = memory._build_pipeline_config()
        assert config.enrich.wikidata.enabled is False, (
            "wikidata grounding must be disabled in lite mode"
        )
    finally:
        VectorStore.set_default_backend(None)


def test_apply_lite_pipeline_profile_nulls_metrics(tmp_path):
    """The patched _create_pipeline_runner returns a runner with _metrics=None."""
    from smartmemory.stores.vector.vector_store import VectorStore
    try:
        memory = create_lite_memory(str(tmp_path))
        runner = memory._create_pipeline_runner()
        assert runner._metrics is None, (
            "pipeline runner._metrics must be None in lite mode to suppress Redis Streams calls"
        )
    finally:
        VectorStore.set_default_backend(None)


# ── env vars set by create_lite_memory ───────────────────────────────────────

def test_create_lite_memory_sets_cache_disabled_env(tmp_path):
    """create_lite_memory() sets SMARTMEMORY_CACHE_DISABLED=true in the environment."""
    from smartmemory.stores.vector.vector_store import VectorStore
    # Clear any existing value
    os.environ.pop("SMARTMEMORY_CACHE_DISABLED", None)
    try:
        memory = create_lite_memory(str(tmp_path))
        assert os.environ.get("SMARTMEMORY_CACHE_DISABLED") == "true"
    finally:
        VectorStore.set_default_backend(None)
