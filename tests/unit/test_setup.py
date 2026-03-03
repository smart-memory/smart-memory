"""Tests for the idempotent setup installer."""

import json
from unittest.mock import patch


def test_register_hooks_idempotent(tmp_path):
    """Running _register_hooks() twice does not duplicate hook entries."""
    settings_file = tmp_path / "settings.json"
    hooks_dest = tmp_path / "hooks"

    with (
        patch("smartmemory_app.setup.SETTINGS", settings_file),
        patch("smartmemory_app.setup.HOOKS_DEST", hooks_dest),
    ):
        from smartmemory_app.setup import _register_hooks, _get_hook_registrations

        _register_hooks()
        _register_hooks()  # second run

        # Build expected entries inside patch context so they use hooks_dest paths.
        expected = _get_hook_registrations()

    cfg = json.loads(settings_file.read_text())
    for event, entry in expected.items():
        event_hooks = cfg["hooks"].get(event, [])
        count = sum(1 for h in event_hooks if h == entry)
        assert count == 1, (
            f"Hook entry for {event} must appear exactly once, got {count}"
        )


def test_copy_skills_no_overwrite(tmp_path):
    """_copy_skills() does not overwrite existing skill files."""
    skills_dest = tmp_path / "skills"
    skills_dest.mkdir()
    skills_src = tmp_path / "skills_src"
    skills_src.mkdir()

    # Write an existing custom skill
    existing_skill = skills_dest / "remember.md"
    existing_skill.write_text("# Custom content that must not be overwritten")

    # Write a new skill in source (different content)
    (skills_src / "remember.md").write_text("# New content from plugin")
    (skills_src / "search.md").write_text("# Search skill")

    with (
        patch("smartmemory_app.setup.SKILLS_SRC", skills_src),
        patch("smartmemory_app.setup.CLAUDE_DIR", tmp_path),
    ):
        from smartmemory_app.setup import _copy_skills

        _copy_skills()

    # Custom content must be preserved
    assert existing_skill.read_text() == "# Custom content that must not be overwritten"
    # New skill must be copied
    assert (skills_dest / "search.md").exists()


def test_seed_data_dir_idempotent(tmp_path):
    """_seed_data_dir() can be called twice without error or data corruption."""
    with patch("smartmemory_app.setup.DATA_DIR", tmp_path):
        from smartmemory_app.setup import _seed_data_dir

        _seed_data_dir()
        first_content = (tmp_path / "entity_patterns.jsonl").read_text()
        _seed_data_dir()
        second_content = (tmp_path / "entity_patterns.jsonl").read_text()

    assert first_content == second_content, "seed_data_dir must be idempotent"
    assert len(first_content) > 0, "seed file must not be empty"


def test_deregister_hooks_removes_entries(tmp_path):
    """_deregister_hooks() removes registered entries from settings.json."""
    settings_file = tmp_path / "settings.json"
    hooks_dest = tmp_path / "hooks"

    # First register
    with (
        patch("smartmemory_app.setup.SETTINGS", settings_file),
        patch("smartmemory_app.setup.HOOKS_DEST", hooks_dest),
    ):
        from smartmemory_app.setup import _register_hooks, _deregister_hooks

        _register_hooks()
        _deregister_hooks()

    cfg = json.loads(settings_file.read_text())
    for event in ["SessionStart", "Stop", "PostToolUseFailure"]:
        assert cfg["hooks"].get(event, []) == [], (
            f"All SmartMemory entries for {event} must be removed after deregister"
        )


def test_deregister_hooks_preserves_other_hooks(tmp_path):
    """_deregister_hooks() leaves non-SmartMemory hooks in settings.json untouched."""
    settings_file = tmp_path / "settings.json"
    hooks_dest = tmp_path / "hooks"
    other_hook = {"command": "bash", "args": ["/other/hook.sh"]}
    settings_file.write_text(json.dumps({"hooks": {"SessionStart": [other_hook]}}))

    with (
        patch("smartmemory_app.setup.SETTINGS", settings_file),
        patch("smartmemory_app.setup.HOOKS_DEST", hooks_dest),
    ):
        from smartmemory_app.setup import _register_hooks, _deregister_hooks

        _register_hooks()
        _deregister_hooks()

    cfg = json.loads(settings_file.read_text())
    assert other_hook in cfg["hooks"]["SessionStart"], (
        "Non-SmartMemory hooks must survive deregistration"
    )


def test_remove_data_dir_honours_env_var(tmp_path, monkeypatch):
    """_remove_data_dir() removes the SMARTMEMORY_DATA_DIR path, not the default."""
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "memory.db").touch()
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(custom_dir))

    with patch("smartmemory_app.setup.DATA_DIR", tmp_path / "default"):
        from smartmemory_app.setup import _remove_data_dir

        _remove_data_dir()

    assert not custom_dir.exists(), "SMARTMEMORY_DATA_DIR must be removed"


def test_seed_data_dir_honours_env_var(tmp_path, monkeypatch):
    """_seed_data_dir() uses SMARTMEMORY_DATA_DIR when set, not the default DATA_DIR."""
    custom_dir = tmp_path / "custom"
    monkeypatch.setenv("SMARTMEMORY_DATA_DIR", str(custom_dir))

    with patch("smartmemory_app.setup.DATA_DIR", tmp_path / "default"):
        from smartmemory_app.setup import _seed_data_dir

        _seed_data_dir()

    assert (custom_dir / "entity_patterns.jsonl").exists(), (
        "_seed_data_dir must seed into SMARTMEMORY_DATA_DIR, not the default path"
    )
    assert not (tmp_path / "default" / "entity_patterns.jsonl").exists(), (
        "default DATA_DIR must not be seeded when SMARTMEMORY_DATA_DIR is set"
    )
