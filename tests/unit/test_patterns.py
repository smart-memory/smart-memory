"""Tests for LitePatternManager."""

import json
import pytest


def test_seeds_on_first_run(tmp_path):
    """LitePatternManager seeds entity_patterns.jsonl on first run."""
    from smartmemory_cc.patterns import LitePatternManager

    pm = LitePatternManager(tmp_path)
    patterns = pm.get_patterns()
    assert len(patterns) > 0, "seed patterns must be present after first run"
    assert (tmp_path / "entity_patterns.jsonl").exists()


def test_load_existing(tmp_path):
    """LitePatternManager loads patterns from an existing JSONL file."""
    from smartmemory_cc.patterns import LitePatternManager

    # Write a custom JSONL file
    pattern_file = tmp_path / "entity_patterns.jsonl"
    entry = {"name": "pytest", "label": "TOOL", "confidence": 0.99, "frequency": 2}
    pattern_file.write_text(json.dumps(entry) + "\n")

    pm = LitePatternManager(tmp_path)
    patterns = pm.get_patterns()
    assert "pytest" in patterns
    assert patterns["pytest"] == "TOOL"


def test_get_patterns_quality_gate(tmp_path):
    """Patterns with frequency=1 are excluded; frequency=2 are included."""
    from smartmemory_cc.patterns import LitePatternManager

    pattern_file = tmp_path / "entity_patterns.jsonl"
    entries = [
        {"name": "HighFreq", "label": "TOOL", "confidence": 0.99, "frequency": 2},
        {"name": "LowFreq", "label": "TOOL", "confidence": 0.99, "frequency": 1},
    ]
    pattern_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    pm = LitePatternManager(tmp_path)
    patterns = pm.get_patterns()
    assert "highfreq" in patterns, "frequency=2 must pass quality gate"
    assert "lowfreq" not in patterns, "frequency=1 must be excluded by quality gate"


def test_add_pattern_increments_frequency(tmp_path):
    """add_pattern() increments frequency for existing patterns."""
    from smartmemory_cc.patterns import LitePatternManager

    pattern_file = tmp_path / "entity_patterns.jsonl"
    entry = {"name": "pytest", "label": "TOOL", "confidence": 0.85, "frequency": 1}
    pattern_file.write_text(json.dumps(entry) + "\n")

    pm = LitePatternManager(tmp_path)
    # frequency=1 initially, not in get_patterns()
    assert "pytest" not in pm.get_patterns()
    pm.add_pattern("pytest", "TOOL", confidence=0.90)
    # Now frequency=2, should appear
    assert "pytest" in pm.get_patterns()


def test_add_pattern_quality_gate_confidence(tmp_path):
    """add_pattern() raises ValueError for confidence <= 0.8."""
    from smartmemory_cc.patterns import LitePatternManager

    pm = LitePatternManager(tmp_path)
    with pytest.raises(ValueError, match="confidence"):
        pm.add_pattern("pytest", "TOOL", confidence=0.8)
    with pytest.raises(ValueError, match="confidence"):
        pm.add_pattern("pytest", "TOOL", confidence=0.5)


def test_add_pattern_quality_gate_name_len(tmp_path):
    """add_pattern() raises ValueError for name length <= 3."""
    from smartmemory_cc.patterns import LitePatternManager

    pm = LitePatternManager(tmp_path)
    with pytest.raises(ValueError, match="length"):
        pm.add_pattern("abc", "TOOL", confidence=0.9)
    with pytest.raises(ValueError, match="length"):
        pm.add_pattern("ab", "TOOL", confidence=0.9)


def test_seed_survives_plugin_update(tmp_path):
    """Existing entity_patterns.jsonl is not overwritten on second instantiation."""
    from smartmemory_cc.patterns import LitePatternManager

    # First run: seeds file
    LitePatternManager(tmp_path)
    # Add a custom entry manually
    custom_entry = {
        "name": "MyCustomTool",
        "label": "CUSTOM",
        "confidence": 0.99,
        "frequency": 2,
    }
    with open(tmp_path / "entity_patterns.jsonl", "a") as f:
        f.write(json.dumps(custom_entry) + "\n")

    # Second run: must load existing file, not overwrite with seeds
    pm2 = LitePatternManager(tmp_path)
    patterns = pm2.get_patterns()
    assert "mycustomtool" in patterns, "existing custom entry must be preserved"
