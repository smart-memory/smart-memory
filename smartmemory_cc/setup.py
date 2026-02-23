import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import click

HOOKS_SRC = Path(__file__).parent / "hooks"
SKILLS_SRC = Path(__file__).parent / "skills"
CLAUDE_DIR = Path.home() / ".claude"
HOOKS_DEST = CLAUDE_DIR / "hooks"  # where hooks are copied to
SETTINGS = CLAUDE_DIR / "settings.json"
DATA_DIR = Path.home() / ".smartmemory"

HOOK_NAMES = ["session-start.sh", "session-end.sh", "post-tool-failure.sh"]
SKILL_NAMES = ["remember.md", "search.md", "ingest.md", "orient.md"]


def _get_hook_registrations() -> dict:
    """Build hook registration entries from current HOOKS_DEST.

    Constructed lazily (not at import time) so that tests can monkeypatch
    HOOKS_DEST and have the change reflected in both writing and reading.
    Registered paths point to the COPIED destination (~/.claude/hooks/), not
    the package source, so registrations are stable across package upgrades.
    """
    return {
        "SessionStart": {
            "command": "bash",
            "args": [str(HOOKS_DEST / "session-start.sh")],
        },
        "Stop": {"command": "bash", "args": [str(HOOKS_DEST / "session-end.sh")]},
        "PostToolUseFailure": {
            "command": "bash",
            "args": [str(HOOKS_DEST / "post-tool-failure.sh")],
        },
    }


# Module-level alias for external callers that need the structure at import time.
# NOTE: frozen at import — prefer _get_hook_registrations() inside functions.
HOOK_REGISTRATIONS = _get_hook_registrations()


@click.command()
def setup() -> None:
    """Wire SmartMemory hooks and skills into ~/.claude/."""
    _ensure_spacy()
    _copy_hooks()
    _copy_skills()
    _register_hooks()
    _seed_data_dir()
    click.echo("SmartMemory wired. Restart Claude Code to activate hooks.")


@click.command()
@click.option(
    "--keep-data",
    is_flag=True,
    default=False,
    help="Keep ~/.smartmemory/ data directory (memories, patterns). Only removes hooks and skills.",
)
def uninstall(keep_data: bool) -> None:
    """Remove SmartMemory hooks, skills, and optionally data from ~/.smartmemory/."""
    _deregister_hooks()
    _remove_hooks()
    _remove_skills()
    if not keep_data:
        _remove_data_dir()
    click.echo("SmartMemory removed. Restart Claude Code to deactivate hooks.")


# ── Install helpers ──────────────────────────────────────────────────────────


def _ensure_spacy() -> None:
    try:
        import spacy

        spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        click.echo("Downloading en_core_web_sm (~15MB)...")
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            capture_output=True,
        )
        if result.returncode != 0:
            click.echo(
                "WARNING: spaCy model download failed. Run manually:\n"
                "  python -m spacy download en_core_web_sm",
                err=True,
            )


def _copy_hooks() -> None:
    dest = CLAUDE_DIR / "hooks"
    dest.mkdir(parents=True, exist_ok=True)
    for src in HOOKS_SRC.glob("*.sh"):
        shutil.copy2(src, dest / src.name)


def _copy_skills() -> None:
    dest = CLAUDE_DIR / "skills"
    dest.mkdir(parents=True, exist_ok=True)
    for src in SKILLS_SRC.glob("*.md"):
        target = dest / src.name
        if not target.exists():
            shutil.copy2(src, target)


def _register_hooks() -> None:
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}
    hooks = cfg.setdefault("hooks", {})
    changed = False
    for event, entry in _get_hook_registrations().items():
        existing = hooks.get(event, [])
        if not isinstance(existing, list):
            existing = [existing]
        if entry not in existing:
            existing.append(entry)
            hooks[event] = existing
            changed = True
    if changed:
        SETTINGS.write_text(json.dumps(cfg, indent=2))


def _seed_data_dir() -> None:
    # Honour SMARTMEMORY_DATA_DIR so setup seeds the same directory that the
    # runtime (storage._resolve_data_dir) will use. Falling back to DATA_DIR
    # only when the env var is unset mirrors storage.py's resolution exactly.
    raw = os.environ.get("SMARTMEMORY_DATA_DIR")
    data_dir = Path(raw) if raw else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    from smartmemory_cc.patterns import LitePatternManager

    LitePatternManager(data_dir)  # seeds entity_patterns.jsonl if absent


# ── Uninstall helpers ────────────────────────────────────────────────────────


def _deregister_hooks() -> None:
    """Remove hook entries from settings.json. Leaves other hooks intact."""
    if not SETTINGS.exists():
        return
    cfg = json.loads(SETTINGS.read_text())
    hooks = cfg.get("hooks", {})
    changed = False
    for event, entry in _get_hook_registrations().items():
        existing = hooks.get(event, [])
        if not isinstance(existing, list):
            existing = [existing]
        filtered = [h for h in existing if h != entry]
        if len(filtered) != len(existing):
            hooks[event] = filtered
            changed = True
    if changed:
        SETTINGS.write_text(json.dumps(cfg, indent=2))


def _remove_hooks() -> None:
    for name in HOOK_NAMES:
        path = HOOKS_DEST / name
        if path.exists():
            path.unlink()


def _remove_skills() -> None:
    skills_dest = CLAUDE_DIR / "skills"
    for name in SKILL_NAMES:
        path = skills_dest / name
        if path.exists():
            path.unlink()


def _remove_data_dir() -> None:
    raw = os.environ.get("SMARTMEMORY_DATA_DIR")
    data_dir = Path(raw) if raw else DATA_DIR
    if data_dir.exists():
        shutil.rmtree(data_dir)
        click.echo(f"Removed data directory: {data_dir}")


if __name__ == "__main__":
    setup()
