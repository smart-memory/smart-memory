"""Tests for JSONLPatternStore."""

import json
import pytest


def _suppress_seed_sync(tmp_path):
    """Write a seeds_applied.json to prevent seed sync during store init.

    Use in tests that need an empty or custom-only pattern file.
    """
    from smartmemory_app.patterns import SEEDS_MANIFEST
    if SEEDS_MANIFEST.exists():
        applied = json.loads(SEEDS_MANIFEST.read_text())
        (tmp_path / "seeds_applied.json").write_text(json.dumps(applied))
    else:
        (tmp_path / "seeds_applied.json").write_text('{"version":"1.0.0","files":{}}')


def test_seeds_on_first_run(tmp_path):
    """JSONLPatternStore seeds entity_patterns.jsonl on first run."""
    from smartmemory_app.patterns import JSONLPatternStore

    store = JSONLPatternStore(tmp_path)
    patterns = store.load("default")
    assert len(patterns) > 0, "seed patterns must be present after first run"
    assert (tmp_path / "entity_patterns.jsonl").exists()


def test_load_existing(tmp_path):
    """JSONLPatternStore loads patterns from an existing JSONL file."""
    from smartmemory_app.patterns import JSONLPatternStore

    # Write a custom JSONL file
    pattern_file = tmp_path / "entity_patterns.jsonl"
    entry = {"name": "pytest", "label": "TOOL", "confidence": 0.99, "frequency": 2}
    pattern_file.write_text(json.dumps(entry) + "\n")

    store = JSONLPatternStore(tmp_path)
    patterns = store.load("default")
    assert ("pytest", "TOOL") in patterns


def test_load_quality_gate(tmp_path):
    """Patterns with frequency=1 are excluded; frequency=2 are included."""
    from smartmemory_app.patterns import JSONLPatternStore

    pattern_file = tmp_path / "entity_patterns.jsonl"
    entries = [
        {"name": "HighFreq", "label": "TOOL", "confidence": 0.99, "frequency": 2},
        {"name": "LowFreq", "label": "TOOL", "confidence": 0.99, "frequency": 1},
    ]
    pattern_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    store = JSONLPatternStore(tmp_path)
    patterns = store.load("default")
    names = [name.lower() for name, _ in patterns]
    assert "highfreq" in names, "frequency=2 must pass quality gate"
    assert "lowfreq" not in names, "frequency=1 must be excluded by quality gate"


def test_save_increments_frequency(tmp_path):
    """save() increments frequency for existing patterns."""
    from smartmemory_app.patterns import JSONLPatternStore

    pattern_file = tmp_path / "entity_patterns.jsonl"
    entry = {"name": "pytest", "label": "TOOL", "confidence": 0.85, "frequency": 1}
    pattern_file.write_text(json.dumps(entry) + "\n")

    store = JSONLPatternStore(tmp_path)
    # frequency=1 initially, not returned by load()
    assert ("pytest", "TOOL") not in store.load("default")
    store.save("pytest", "TOOL", 0.90, 1, "default", "test")
    # Now frequency=2, should appear
    assert ("pytest", "TOOL") in store.load("default")


def test_save_count_2_visible_immediately(tmp_path):
    """Pattern saved with count=2 passes the quality gate immediately."""
    from smartmemory_app.patterns import JSONLPatternStore

    pattern_file = tmp_path / "entity_patterns.jsonl"
    pattern_file.write_text("")  # empty file, no seeds
    _suppress_seed_sync(tmp_path)

    store = JSONLPatternStore(tmp_path)
    store.save("FastAPI", "Technology", 0.9, 2, "default", "test")
    patterns = store.load("default")
    names = [name.lower() for name, _ in patterns]
    assert "fastapi" in names


def test_save_preserves_original_source(tmp_path):
    """source is stored on CREATE only — not overwritten on update."""
    from smartmemory_app.patterns import JSONLPatternStore

    pattern_file = tmp_path / "entity_patterns.jsonl"
    pattern_file.write_text("")
    _suppress_seed_sync(tmp_path)

    store = JSONLPatternStore(tmp_path)
    store.save("FastAPI", "Technology", 0.9, 2, "default", "original_source")
    store.save("FastAPI", "Technology", 0.9, 1, "default", "updated_source")

    # Read raw JSONL to check source field directly
    raw = json.loads((tmp_path / "entity_patterns.jsonl").read_text().strip())
    assert raw["source"] == "original_source", "source must not be overwritten on update"


def test_save_count_1_needs_second_call(tmp_path):
    """Default count=1 means pattern is invisible until a second save."""
    from smartmemory_app.patterns import JSONLPatternStore

    pattern_file = tmp_path / "entity_patterns.jsonl"
    pattern_file.write_text("")
    _suppress_seed_sync(tmp_path)

    store = JSONLPatternStore(tmp_path)
    store.save("Flask", "Technology", 0.9, 1, "default")
    names = [n.lower() for n, _ in store.load("default")]
    assert "flask" not in names  # frequency=1

    store.save("Flask", "Technology", 0.9, 1, "default")
    names = [n.lower() for n, _ in store.load("default")]
    assert "flask" in names  # frequency=2


def test_save_persists_to_file(tmp_path):
    """count=2 value is written to JSONL and survives a fresh store load."""
    from smartmemory_app.patterns import JSONLPatternStore

    pattern_file = tmp_path / "entity_patterns.jsonl"
    pattern_file.write_text("")
    _suppress_seed_sync(tmp_path)

    store = JSONLPatternStore(tmp_path)
    store.save("FastAPI", "Technology", 0.9, 2, "default")

    # Fresh instance reads from disk
    store2 = JSONLPatternStore(tmp_path)
    names = [n.lower() for n, _ in store2.load("default")]
    assert "fastapi" in names


def test_seed_survives_plugin_update(tmp_path):
    """Existing entity_patterns.jsonl is not overwritten on second instantiation."""
    from smartmemory_app.patterns import JSONLPatternStore

    # First run: seeds file
    JSONLPatternStore(tmp_path)
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
    store2 = JSONLPatternStore(tmp_path)
    patterns = store2.load("default")
    names = [n.lower() for n, _ in patterns]
    assert "mycustomtool" in names, "existing custom entry must be preserved"


# ------------------------------------------------------------------ #
# Quality gate — enforced by PatternManager, not JSONLPatternStore
# ------------------------------------------------------------------ #


def test_pattern_manager_blocklist_rejection(tmp_path):
    """PatternManager.add_patterns() rejects blocklist words regardless of count."""
    from smartmemory.ontology.pattern_manager import PatternManager
    from smartmemory_app.patterns import JSONLPatternStore

    pattern_file = tmp_path / "entity_patterns.jsonl"
    pattern_file.write_text("")
    _suppress_seed_sync(tmp_path)

    pm = PatternManager(store=JSONLPatternStore(tmp_path))
    accepted = pm.add_patterns({"the": "Concept", "python": "Technology"}, initial_count=2)

    assert accepted == 1
    assert "the" not in pm.get_patterns()
    assert "python" in pm.get_patterns()


def test_pattern_manager_short_name_rejection(tmp_path):
    """PatternManager.add_patterns() rejects names shorter than 2 characters."""
    from smartmemory.ontology.pattern_manager import PatternManager
    from smartmemory_app.patterns import JSONLPatternStore

    pattern_file = tmp_path / "entity_patterns.jsonl"
    pattern_file.write_text("")
    _suppress_seed_sync(tmp_path)

    pm = PatternManager(store=JSONLPatternStore(tmp_path))
    accepted = pm.add_patterns({"a": "Concept", "qt": "Technology"}, initial_count=2)

    assert accepted == 1  # "a" rejected (len < 2), "qt" accepted (len == 2)
    assert "a" not in pm.get_patterns()
    assert "qt" in pm.get_patterns()


def test_pattern_manager_add_patterns_count_2_survives_reload(tmp_path):
    """Patterns added with initial_count=2 persist through a store reload."""
    from smartmemory.ontology.pattern_manager import PatternManager
    from smartmemory_app.patterns import JSONLPatternStore

    pattern_file = tmp_path / "entity_patterns.jsonl"
    pattern_file.write_text("")
    _suppress_seed_sync(tmp_path)

    pm = PatternManager(store=JSONLPatternStore(tmp_path))
    pm.add_patterns({"FastAPI": "Technology"}, initial_count=2)

    assert "fastapi" in pm.get_patterns()

    pm.reload()
    assert "fastapi" in pm.get_patterns()  # survives reload


# ------------------------------------------------------------------ #
# delete()
# ------------------------------------------------------------------ #


def test_delete_removes_pattern(tmp_path):
    """delete() removes a pattern from the JSONL store."""
    from smartmemory_app.patterns import JSONLPatternStore

    pattern_file = tmp_path / "entity_patterns.jsonl"
    entry = {"name": "pytest", "label": "TOOL", "confidence": 0.99, "frequency": 2}
    pattern_file.write_text(json.dumps(entry) + "\n")

    store = JSONLPatternStore(tmp_path)
    assert ("pytest", "TOOL") in store.load("default")

    store.delete("pytest", "TOOL", "default")
    assert ("pytest", "TOOL") not in store.load("default")


# ------------------------------------------------------------------ #
# PatternStore protocol
# ------------------------------------------------------------------ #


def test_jsonl_store_satisfies_protocol(tmp_path):
    from smartmemory_app.patterns import JSONLPatternStore
    from smartmemory.ontology.pattern_manager import PatternStore

    assert isinstance(JSONLPatternStore(tmp_path), PatternStore)


# ------------------------------------------------------------------ #
# SEED-1d: Versioned seed sync
# ------------------------------------------------------------------ #


def test_seed_sync_creates_applied_manifest(tmp_path):
    """First run writes seeds_applied.json tracking what was synced."""
    from smartmemory_app.patterns import JSONLPatternStore

    JSONLPatternStore(tmp_path)
    applied = tmp_path / "seeds_applied.json"
    assert applied.exists(), "seeds_applied.json must be created after first sync"
    data = json.loads(applied.read_text())
    assert "version" in data
    assert "files" in data


def test_seed_sync_additive_on_upgrade(tmp_path):
    """New seed patterns are merged additively; existing patterns preserved."""
    from smartmemory_app.patterns import JSONLPatternStore, SEEDS_DIR, SEEDS_MANIFEST

    # First run: initial seed
    store = JSONLPatternStore(tmp_path)
    initial_count = len(store.load("default"))
    assert initial_count > 0

    # Simulate user adding a custom pattern
    store.save("MyCustomLib", "FRAMEWORK", 0.99, 2, "default", "user")
    assert ("MyCustomLib", "FRAMEWORK") in store.load("default")

    # Simulate package upgrade: change a seed file checksum
    applied = json.loads((tmp_path / "seeds_applied.json").read_text())
    # Invalidate one file's checksum to trigger re-sync
    first_file = list(applied["files"].keys())[0]
    applied["files"][first_file]["checksum"] = "stale"
    (tmp_path / "seeds_applied.json").write_text(json.dumps(applied))

    # Re-init: should sync new patterns additively
    store2 = JSONLPatternStore(tmp_path)
    patterns = store2.load("default")
    names = [n.lower() for n, _ in patterns]
    assert "mycustomlib" in names, "user-added pattern must survive seed sync"
    assert len(patterns) >= initial_count, "seed sync must not remove patterns"


def test_seed_sync_preserves_user_customizations(tmp_path):
    """User-modified patterns (e.g. changed label) are never overwritten by seeds."""
    from smartmemory_app.patterns import JSONLPatternStore

    # First run
    store = JSONLPatternStore(tmp_path)

    # User overrides a seed pattern's label
    entries = store._read_all()
    if entries:
        first_key = next(iter(entries))
        entries[first_key]["label"] = "USER_CUSTOM"
        entries[first_key]["source"] = "user_override"
        store._write_all(entries)

        # Clear applied manifest to force re-sync
        (tmp_path / "seeds_applied.json").unlink()

        # Re-init: should NOT overwrite the user's customization
        store2 = JSONLPatternStore(tmp_path)
        entries2 = store2._read_all()
        assert entries2[first_key]["label"] == "USER_CUSTOM"
        assert entries2[first_key]["source"] == "user_override"


def test_seed_sync_no_applied_means_full_sync(tmp_path):
    """Missing seeds_applied.json triggers full sync."""
    from smartmemory_app.patterns import JSONLPatternStore

    # Create an existing patterns file with just one entry
    (tmp_path / "entity_patterns.jsonl").write_text(
        json.dumps({"name": "Solo", "label": "TOOL", "confidence": 0.99, "frequency": 2}) + "\n"
    )
    # No seeds_applied.json — should trigger full seed sync

    store = JSONLPatternStore(tmp_path)
    patterns = store.load("default")
    names = [n.lower() for n, _ in patterns]
    assert "solo" in names, "existing pattern must survive"
    assert len(patterns) > 1, "seeds must have been merged in"


def test_seed_sync_idempotent(tmp_path):
    """Running sync twice produces the same result."""
    from smartmemory_app.patterns import JSONLPatternStore

    store1 = JSONLPatternStore(tmp_path)
    count1 = len(store1.load("default"))

    store2 = JSONLPatternStore(tmp_path)
    count2 = len(store2.load("default"))

    assert count1 == count2, "second sync must not add duplicates"
