"""Integration test: LitePatternManager wires correctly into the extraction pipeline.

Verifies:
1. LitePatternManager.get_patterns() returns the shape EntityRulerStage expects (dict[str, str])
2. create_lite_memory() accepts entity_ruler_patterns without error
3. Patterns with frequency < 2 are excluded; those with frequency >= 2 are included
"""

import json
import pytest


@pytest.mark.integration
def test_get_patterns_returns_correct_shape(tmp_path):
    """get_patterns() returns dict[str, str] as EntityRulerStage expects."""
    from smartmemory_pkg.patterns import LitePatternManager

    pm = LitePatternManager(tmp_path)
    patterns = pm.get_patterns()

    assert isinstance(patterns, dict), "get_patterns() must return a dict"
    for k, v in patterns.items():
        assert isinstance(k, str), f"pattern key must be str, got {type(k)}"
        assert isinstance(v, str), f"pattern label must be str, got {type(v)}"


@pytest.mark.integration
def test_create_lite_memory_accepts_pattern_manager(tmp_path):
    """create_lite_memory() accepts entity_ruler_patterns without raising."""
    from smartmemory.tools.factory import create_lite_memory
    from smartmemory_pkg.patterns import LitePatternManager

    pm = LitePatternManager(tmp_path)
    # Must not raise — this is the critical seam between plugin and core
    mem = create_lite_memory(data_dir=str(tmp_path), entity_ruler_patterns=pm)
    assert mem is not None


@pytest.mark.integration
def test_frequency_gate_controls_active_patterns(tmp_path):
    """Only patterns with frequency >= 2 are returned by get_patterns()."""
    from smartmemory_pkg.patterns import LitePatternManager

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

    pm = LitePatternManager(tmp_path)
    patterns = pm.get_patterns()

    assert "activetool" in patterns
    assert "highfreqtool" in patterns
    assert "inactivetool" not in patterns, "frequency=1 must be excluded"


@pytest.mark.integration
def test_add_pattern_persists_across_reload(tmp_path):
    """add_pattern() persists to JSONL so a fresh LitePatternManager sees the change."""
    from smartmemory_pkg.patterns import LitePatternManager

    pm1 = LitePatternManager(tmp_path)
    # Add pattern twice to hit the frequency >= 2 gate
    pm1.add_pattern("FastMCP", "LIBRARY", confidence=0.9)
    pm1.add_pattern("FastMCP", "LIBRARY", confidence=0.9)

    # Fresh instance reads from disk
    pm2 = LitePatternManager(tmp_path)
    patterns = pm2.get_patterns()
    assert "fastmcp" in patterns, "persisted pattern must survive reload"


@pytest.mark.integration
def test_seed_patterns_present_after_fresh_init(tmp_path):
    """A fresh data directory gets seed patterns from bundled seed_patterns.jsonl."""
    from smartmemory_pkg.patterns import LitePatternManager, SEED_PATTERNS_FILE

    assert SEED_PATTERNS_FILE.exists(), (
        "bundled seed_patterns.jsonl must exist in package"
    )

    pm = LitePatternManager(tmp_path)
    # The patterns dict should be populated from seeds (even if none pass the freq gate yet,
    # the internal _patterns dict should not be empty)
    assert len(pm._patterns) > 0, (
        "seed patterns must be loaded into _patterns on fresh init"
    )
    assert (tmp_path / "entity_patterns.jsonl").exists(), (
        "seed must be written to data dir"
    )
