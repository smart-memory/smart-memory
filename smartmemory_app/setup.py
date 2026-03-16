"""DIST-LITE-5: Setup — first-run questionnaire + Claude Code hook wiring.

`smartmemory setup` is the single entry point for both:
  - First-run mode/pipeline config (local vs remote, LLM provider, etc.)
  - Claude Code hook installation (~/.claude/hooks/ + settings.json)

Local mode: asks pipeline questions, writes config, wires Claude Code hooks.
  All local deps (smartmemory-core, spaCy, usearch, filelock) are included
  in the base `pip install smartmemory` — no extras needed.
Remote mode: validates API key against /auth/me, stores in OS keychain,
  writes config. No hook wiring (remote mode has no hook-invoked pipeline).
"""
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
HOOKS_DEST = CLAUDE_DIR / "hooks"
SETTINGS = CLAUDE_DIR / "settings.json"
DATA_DIR = Path.home() / ".smartmemory"

# Maps source filename (in package) → namespaced dest filename (in ~/.claude/hooks/)
_HOOK_FILE_MAP = {
    "session-start.sh": "smartmemory-session-start.sh",
    "session-end.sh": "smartmemory-session-end.sh",
    "post-tool-failure.sh": "smartmemory-post-tool-failure.sh",
}
HOOK_NAMES = list(_HOOK_FILE_MAP.values())
SKILL_NAMES = ["remember.md", "search.md", "ingest.md", "orient.md"]

# Legacy entries to clean up during install (pre-namespacing)
_LEGACY_HOOK_REGISTRATIONS = {
    "SessionStart": {"command": "bash", "args": [str(HOOKS_DEST / "session-start.sh")]},
    "Stop": {"command": "bash", "args": [str(HOOKS_DEST / "session-end.sh")]},
    "PostToolUseFailure": {"command": "bash", "args": [str(HOOKS_DEST / "post-tool-failure.sh")]},
}


def _get_hook_registrations() -> dict:
    """Build hook registration entries using current Claude Code hooks format.

    Each event maps to a registration dict with matcher + hooks array.
    Paths point to the COPIED destination (~/.claude/hooks/), not the package
    source, so registrations are stable across package upgrades.
    """
    return {
        "SessionStart": {
            "matcher": "",
            "hooks": [{"type": "command", "command": f"bash {HOOKS_DEST / 'smartmemory-session-start.sh'}"}],
        },
        "Stop": {
            "matcher": "",
            "hooks": [{"type": "command", "command": f"bash {HOOKS_DEST / 'smartmemory-session-end.sh'}"}],
        },
        "PostToolUseFailure": {
            "matcher": "",
            "hooks": [{"type": "command", "command": f"bash {HOOKS_DEST / 'smartmemory-post-tool-failure.sh'}"}],
        },
    }


# Module-level alias for external callers that need the structure at import time.
# NOTE: frozen at import — prefer _get_hook_registrations() inside functions.
HOOK_REGISTRATIONS = _get_hook_registrations()


# ── Setup command ─────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["local", "remote"]),
    default=None,
    help="Skip questionnaire: set mode directly.",
)
@click.option(
    "--api-key",
    "api_key",
    default=None,
    help="Remote API key (stored in OS keychain). Required for --mode remote.",
)
@click.option(
    "--for",
    "for_tool",
    default=None,
    help="Tool config (e.g. 'cursor'). Works in both modes.",
)
def setup(mode: str | None, api_key: str | None, for_tool: str | None) -> None:
    """First-run questionnaire: configure local or remote mode."""
    if mode is None:
        click.echo("Welcome to SmartMemory.\n")
        click.echo("Where do you want to store memories?")
        click.echo("  1. Local  — on this machine, no account needed")
        click.echo("  2. Remote — SmartMemory hosted service (api.smartmemory.ai)")
        choice = click.prompt(">", type=click.Choice(["1", "2"]), default="1")
        mode = "local" if choice == "1" else "remote"

    if mode == "remote":
        _setup_remote(api_key)
    else:
        _setup_local()

    if for_tool:
        _setup_tool_config(for_tool)

    click.echo("\nSetup complete. Run: smartmemory server")


# ── Mode setup helpers ────────────────────────────────────────────────────────


def _setup_local() -> None:
    """Ask pipeline questions, write config, wire Claude Code hooks.

    All local deps (smartmemory-core, spaCy, usearch, filelock) are already
    installed as part of `pip install smartmemory` — no extra install step needed.
    """
    from smartmemory_app.config import SmartMemoryConfig, save_config

    coref = click.confirm(
        "\nEnable coreference resolution? "
        "(downloads ~500MB fastcoref model on first run)",
        default=False,
    )
    click.echo("\nSmartMemory needs an LLM to extract entities and relationships.")
    click.echo("  groq         — free tier, fast inference (recommended)")
    click.echo("  claude-agent — Claude Agent SDK, OAuth (no API key)")
    click.echo("  anthropic    — Anthropic API (ANTHROPIC_API_KEY)")
    click.echo("  openai       — OpenAI API (OPENAI_API_KEY)")
    click.echo("  ollama       — free, local (llama3.1, mistral)")
    click.echo("  lmstudio     — local, OpenAI-compatible endpoint")
    click.echo("  none         — EntityRuler only (very limited)")
    llm = click.prompt(
        "LLM provider",
        default="groq",
    )
    embedding = click.prompt(
        "Embedding provider? (local / openai / ollama)",
        default="local",
    )
    data_dir = click.prompt("Default data directory", default="~/.smartmemory")

    cfg = SmartMemoryConfig(
        mode="local",
        coreference=coref,
        llm_provider=llm,
        embedding_provider=embedding,
        data_dir=data_dir,
    )
    save_config(cfg)

    # Wire Claude Code hooks (preserved from original setup.py)
    _ensure_spacy()
    _copy_hooks()
    _copy_skills()
    _register_hooks()
    _seed_data_dir()


def _setup_remote(api_key: str | None) -> None:
    """Validate API key, store in OS keychain, write remote config."""
    import httpx
    from smartmemory_app.config import SmartMemoryConfig, save_config, set_api_key

    if not api_key:
        api_key = click.prompt(
            "SmartMemory API key (stored in OS keychain, not config file)",
            hide_input=True,
        )

    click.echo("Validating API key...")
    try:
        r = httpx.get(
            "https://api.smartmemory.ai/auth/me",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        r.raise_for_status()
        user = r.json()
        team_id = user.get("default_team_id") or ""
        click.echo(f"Authenticated as {user.get('email')}. Team: {team_id}")
    except Exception as e:
        click.echo(f"ERROR: API key validation failed: {e}", err=True)
        raise SystemExit(1)

    set_api_key(api_key)  # persist to OS keychain (warns if unavailable, never raises)
    cfg = SmartMemoryConfig(mode="remote", api_key_set=True, team_id=team_id)
    save_config(cfg)


def _setup_tool_config(tool: str) -> None:
    """Emit tool-specific MCP config snippet."""
    click.echo(f"\nTool config for {tool}:")
    click.echo('  Add to your MCP config: {"command": "smartmemory", "args": ["server"]}')


# ── Uninstall command ─────────────────────────────────────────────────────────


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


# ── Install helpers ───────────────────────────────────────────────────────────


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
    for src_name, dest_name in _HOOK_FILE_MAP.items():
        src = HOOKS_SRC / src_name
        if src.exists():
            shutil.copy2(src, dest / dest_name)


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

    # Remove legacy registrations (old format + old filenames)
    for event, legacy_entry in _LEGACY_HOOK_REGISTRATIONS.items():
        existing = hooks.get(event, [])
        if not isinstance(existing, list):
            existing = [existing]
        filtered = [h for h in existing if h != legacy_entry]
        # Also remove old-format entries pointing to legacy filenames
        filtered = [
            h for h in filtered
            if not (isinstance(h, dict) and "command" in h and "args" in h and any("session-start.sh" in str(a) or "session-end.sh" in str(a) or "post-tool-failure.sh" in str(a) for a in h.get("args", [])))
        ]
        # Also remove new-format entries pointing to legacy (non-namespaced) filenames
        filtered = [
            h for h in filtered
            if not (isinstance(h, dict) and "hooks" in h and any(
                "hooks/session-start.sh" in hh.get("command", "") or
                "hooks/session-end.sh" in hh.get("command", "") or
                "hooks/post-tool-failure.sh" in hh.get("command", "")
                for hh in h.get("hooks", []) if isinstance(hh, dict)
            ))
        ]
        if len(filtered) != len(existing):
            hooks[event] = filtered
            changed = True

    # Add current registrations (namespaced, new format)
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
    # runtime (storage._resolve_data_dir) will use.
    raw = os.environ.get("SMARTMEMORY_DATA_DIR")
    data_dir = Path(raw) if raw else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    from smartmemory_app.patterns import JSONLPatternStore
    JSONLPatternStore(data_dir)  # seeds entity_patterns.jsonl if absent (side effect of __init__)


# ── Uninstall helpers ─────────────────────────────────────────────────────────


def _deregister_hooks() -> None:
    """Remove all SmartMemory hook entries from settings.json (current + legacy)."""
    if not SETTINGS.exists():
        return
    cfg = json.loads(SETTINGS.read_text())
    hooks = cfg.get("hooks", {})
    changed = False

    all_entries = {**_get_hook_registrations(), **_LEGACY_HOOK_REGISTRATIONS}
    for event, entry in all_entries.items():
        existing = hooks.get(event, [])
        if not isinstance(existing, list):
            existing = [existing]
        filtered = [h for h in existing if h != entry]
        # Also catch any entry referencing smartmemory hook files
        filtered = [
            h for h in filtered
            if not (isinstance(h, dict) and any(
                "smartmemory-" in str(v) for v in (h.get("args", []) + [hh.get("command", "") for hh in h.get("hooks", []) if isinstance(hh, dict)])
            ))
        ]
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
    # Clean up legacy (non-namespaced) files only if they're SmartMemory-only
    for src_name in _HOOK_FILE_MAP:
        path = HOOKS_DEST / src_name
        if path.exists():
            content = path.read_text()
            if "smartmemory_app" in content and content.count("\n") < 15:
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
