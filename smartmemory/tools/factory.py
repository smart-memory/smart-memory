"""smartmemory.tools.factory — zero-infra SmartMemory factory for Lite."""
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


def _default_data_dir() -> Path:
    return Path.home() / ".smartmemory"


def create_lite_memory(data_dir: Optional[str] = None, entity_ruler_patterns=None):
    """Create a SmartMemory instance backed by SQLite + usearch. No Docker required.

    Passes vector_backend, cache, observability, and pipeline_profile directly to the
    SmartMemory constructor — no monkey-patching required.
    """
    from smartmemory.graph.backends.sqlite import SQLiteBackend
    from smartmemory.graph.smartgraph import SmartGraph
    from smartmemory.pipeline.config import PipelineConfig
    from smartmemory.smart_memory import SmartMemory
    from smartmemory.stores.vector.backends.usearch import UsearchVectorBackend
    from smartmemory.utils.cache import NoOpCache

    data_path = Path(data_dir).expanduser() if data_dir else _default_data_dir()
    data_path.mkdir(parents=True, exist_ok=True)

    sqlite_backend = SQLiteBackend(db_path=str(data_path / "memory.db"))
    usearch_backend = UsearchVectorBackend(
        collection_name="memory",
        persist_directory=str(data_path),
    )
    graph = SmartGraph(backend=sqlite_backend)

    return SmartMemory(
        graph=graph,
        enable_ontology=False,
        vector_backend=usearch_backend,
        cache=NoOpCache(),
        observability=False,
        pipeline_profile=PipelineConfig.lite(),
        entity_ruler_patterns=entity_ruler_patterns,
    )


@contextmanager
def lite_context(data_dir: Optional[str] = None):
    """Context manager that creates a Lite SmartMemory and resets all globals on exit.

    Restores observability env, vector backend, cache override, and closes the SQLite
    connection deterministically. Always use this in tests and scripts:
        with lite_context() as memory:
            memory.ingest("hello")
    """
    from smartmemory.stores.vector.vector_store import VectorStore
    from smartmemory.utils.cache import set_cache_override

    # Capture observability env BEFORE any global mutation so we can restore it
    # unconditionally — even if create_lite_memory() raises partway through.
    prev_obs = os.environ.get("SMARTMEMORY_OBSERVABILITY")

    memory = None
    try:
        memory = create_lite_memory(data_dir)
        yield memory
    finally:
        # Restore globals regardless of whether construction, yield, or body raised.
        VectorStore.set_default_backend(None)
        set_cache_override(None)

        # Restore observability env to its pre-context value.
        if prev_obs is None:
            os.environ.pop("SMARTMEMORY_OBSERVABILITY", None)
        else:
            os.environ["SMARTMEMORY_OBSERVABILITY"] = prev_obs

        # Close the SQLite backend explicitly — don't rely on GC for WAL flush.
        if memory is not None:
            try:
                memory._graph.backend.close()
            except Exception:
                pass
