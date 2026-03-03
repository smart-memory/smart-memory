"""Integration test: JSONLPatternStore + PatternManager wire correctly into the extraction pipeline.

Verifies:
1. PatternManager(store=JSONLPatternStore(...)).get_patterns() returns the shape EntityRulerStage expects (dict[str, str])
2. create_lite_memory() accepts entity_ruler_patterns without error
3. Patterns with frequency < 2 are excluded; those with frequency >= 2 are included
"""

import json
import pytest


@pytest.mark.integration
def test_get_patterns_returns_correct_shape(tmp_path):
    """get_patterns() returns dict[str, str] as EntityRulerStage expects."""
    from smartmemory.ontology.pattern_manager import PatternManager
    from smartmemory_app.patterns import JSONLPatternStore

    pm = PatternManager(store=JSONLPatternStore(tmp_path))
    patterns = pm.get_patterns()

    assert isinstance(patterns, dict), "get_patterns() must return a dict"
    for k, v in patterns.items():
        assert isinstance(k, str), f"pattern key must be str, got {type(k)}"
        assert isinstance(v, str), f"pattern label must be str, got {type(v)}"


@pytest.mark.integration
def test_create_lite_memory_accepts_pattern_manager(tmp_path):
    """create_lite_memory() accepts entity_ruler_patterns without raising."""
    from smartmemory.ontology.pattern_manager import PatternManager
    from smartmemory.tools.factory import create_lite_memory
    from smartmemory_app.patterns import JSONLPatternStore

    pm = PatternManager(store=JSONLPatternStore(tmp_path))
    # Must not raise — this is the critical seam between plugin and core
    mem = create_lite_memory(data_dir=str(tmp_path), entity_ruler_patterns=pm)
    assert mem is not None


@pytest.mark.integration
def test_frequency_gate_controls_active_patterns(tmp_path):
    """Only patterns with frequency >= 2 are returned by get_patterns()."""
    from smartmemory.ontology.pattern_manager import PatternManager
    from smartmemory_app.patterns import JSONLPatternStore

    pattern_file = tmp_path / "entity_patterns.jsonl"
    entries = [
        {"name": "ActiveTool", "label": "TOOL", "confidence": 0.9, "frequency": 2},
        {"name": "InactiveTool", "label": "TOOL", "confidence": 0.9, "frequency": 1},
        {
            "name": "HighFreqTool",
            "label": "FRAMEWORK",
            "confidence": 0.95,
            "frequency": 5,
        },
    ]
    pattern_file.write_text("\n".join(json.dumps(e) for e in entries))

    pm = PatternManager(store=JSONLPatternStore(tmp_path))
    patterns = pm.get_patterns()

    assert "activetool" in patterns
    assert "highfreqtool" in patterns
    assert "inactivetool" not in patterns, "frequency=1 must be excluded"


@pytest.mark.integration
def test_add_patterns_persists_across_reload(tmp_path):
    """add_patterns(initial_count=2) persists to JSONL so a fresh PatternManager sees the change."""
    from smartmemory.ontology.pattern_manager import PatternManager
    from smartmemory_app.patterns import JSONLPatternStore

    pm1 = PatternManager(store=JSONLPatternStore(tmp_path))
    pm1.add_patterns({"FastMCP": "LIBRARY"}, source="test", initial_count=2)

    # Fresh instance reads from disk
    pm2 = PatternManager(store=JSONLPatternStore(tmp_path))
    patterns = pm2.get_patterns()
    assert "fastmcp" in patterns, "persisted pattern must survive reload"


@pytest.mark.integration
def test_seed_patterns_present_after_fresh_init(tmp_path):
    """A fresh data directory gets seed patterns from bundled seed_patterns.jsonl."""
    from smartmemory.ontology.pattern_manager import PatternManager
    from smartmemory_app.patterns import JSONLPatternStore, SEED_PATTERNS_FILE

    assert SEED_PATTERNS_FILE.exists(), (
        "bundled seed_patterns.jsonl must exist in package"
    )

    pm = PatternManager(store=JSONLPatternStore(tmp_path))
    # The store seeds entity_patterns.jsonl on __init__; PatternManager loads it.
    # Even if no seeds pass the frequency >= 2 gate, the file must exist.
    assert (tmp_path / "entity_patterns.jsonl").exists(), (
        "seed must be written to data dir"
    )
