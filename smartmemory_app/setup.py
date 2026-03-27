"""DIST-LITE-5 + DIST-SETUP-TUI-1: Setup — first-run questionnaire + Claude Code hook wiring.

`smartmemory setup` is the single entry point for both:
  - First-run mode/pipeline config (local vs remote, LLM provider, etc.)
  - Claude Code hook installation (~/.claude/hooks/ + settings.json)

Local mode: asks pipeline questions, writes config, wires Claude Code hooks.
  All local deps (smartmemory-core, spaCy, usearch, filelock) are included
  in the base `pip install smartmemory` — no extras needed.
Remote mode: validates API key against /auth/me, stores in OS keychain,
  writes config. No hook wiring (remote mode has no hook-invoked pipeline).

DIST-SETUP-TUI-1: When running interactively with textual installed,
  launches a Textual TUI with arrow-key selection. Falls back to click
  prompts when non-interactive, flags provided, or textual unavailable.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import click


# ── SetupResult — contract between TUI and business logic ─────────────────


@dataclass
class SetupResult:
    """Answers collected from the TUI (or click prompts). Matches SmartMemoryConfig fields."""

    mode: str = "local"
    llm_provider: str = "groq"
    llm_model: str = ""
    embedding_provider: str = "local"
    spacy_model: str = "en_core_web_sm"
    coreference: bool = False
    data_dir: str = "~/.smartmemory"

HOOKS_SRC = Path(__file__).parent / "hooks"
SKILLS_SRC = Path(__file__).parent / "skills"
PLIST_TEMPLATE = Path(__file__).parent / "data" / "ai.smartmemory.daemon.plist"
WORKER_PLIST_TEMPLATE = Path(__file__).parent / "data" / "ai.smartmemory.worker.plist"
CLAUDE_DIR = Path.home() / ".claude"
HOOKS_DEST = CLAUDE_DIR / "hooks"
SETTINGS = CLAUDE_DIR / "settings.json"
DATA_DIR = Path.home() / ".smartmemory"
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
PLIST_NAME = "ai.smartmemory.daemon.plist"
WORKER_PLIST_NAME = "ai.smartmemory.worker.plist"

# Maps source filename (in package) → namespaced dest filename (in ~/.claude/hooks/)
_HOOK_FILE_MAP = {
    # DIST-AGENT-HOOKS-1: 6-phase lifecycle hooks
    "orient.sh": "smartmemory-orient.sh",
    "recall.sh": "smartmemory-recall.sh",
    "observe.sh": "smartmemory-observe.sh",
    "distill.sh": "smartmemory-distill.sh",
    "learn.sh": "smartmemory-learn.sh",
    "persist.sh": "smartmemory-persist.sh",
}
HOOK_NAMES = list(_HOOK_FILE_MAP.values())
SKILL_NAMES = ["remember.md", "search.md", "ingest.md", "orient.md"]

# Legacy entries to clean up during install (pre-namespacing)
_LEGACY_HOOK_REGISTRATIONS = {
    # Pre-DIST-AGENT-HOOKS-1 registrations to clean up
    "SessionStart": {"command": "bash", "args": [str(HOOKS_DEST / "session-start.sh")]},
    "Stop": {"command": "bash", "args": [str(HOOKS_DEST / "session-end.sh")]},
    "PostToolUseFailure": {"command": "bash", "args": [str(HOOKS_DEST / "post-tool-failure.sh")]},
    # Also clean old namespaced variants
    "_legacy_SessionStart": {"command": "bash", "args": [str(HOOKS_DEST / "smartmemory-session-start.sh")]},
    "_legacy_Stop": {"command": "bash", "args": [str(HOOKS_DEST / "smartmemory-session-end.sh")]},
    "_legacy_PostToolUseFailure": {"command": "bash", "args": [str(HOOKS_DEST / "smartmemory-post-tool-failure.sh")]},
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
            "hooks": [{"type": "command", "command": f"bash {HOOKS_DEST / 'smartmemory-orient.sh'}"}],
        },
        "UserPromptSubmit": {
            "matcher": "",
            "hooks": [{"type": "command", "command": f"bash {HOOKS_DEST / 'smartmemory-recall.sh'}"}],
        },
        "PostToolUse": {
            "matcher": "",
            "hooks": [{"type": "command", "command": f"bash {HOOKS_DEST / 'smartmemory-observe.sh'}"}],
        },
        "Stop": {
            "matcher": "",
            "hooks": [{"type": "command", "command": f"bash {HOOKS_DEST / 'smartmemory-distill.sh'}"}],
        },
        "PostToolUseFailure": {
            "matcher": "",
            "hooks": [{"type": "command", "command": f"bash {HOOKS_DEST / 'smartmemory-learn.sh'}"}],
        },
        "SessionEnd": {
            "matcher": "",
            "hooks": [{"type": "command", "command": f"bash {HOOKS_DEST / 'smartmemory-persist.sh'}"}],
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
    """First-run questionnaire: configure local or remote mode.

    Three self-contained branches — each handles config, post-config, AND daemon
    start. No shared post-branch code to avoid double-start bugs.
    """
    # Branch 1: Flags provided — use click flow directly
    # _setup_click() already handles daemon start for local mode internally
    if mode is not None:
        _setup_click(mode, api_key)
        if for_tool:
            _setup_tool_config(for_tool)
        return

    # Branch 2: Interactive TUI (if available)
    if _can_run_tui():
        try:
            from smartmemory_app.setup_tui import run_setup_tui
            result = run_setup_tui()
            if result is None:
                click.echo("Setup cancelled.")
                return
            if result.mode == "remote":
                # TUI only selected mode — hand off to click-based remote setup
                _setup_remote(api_key)
                click.echo("\nSetup complete.")
            else:
                # TUI ProgressScreen already ran config + hooks + daemon
                pass
            if for_tool:
                _setup_tool_config(for_tool)
            return
        except Exception as e:
            click.echo(f"TUI unavailable ({e}), using text prompts.\n")

    # Branch 3: Click fallback
    _setup_click(None, api_key)
    if for_tool:
        _setup_tool_config(for_tool)


def _setup_click(mode: str | None, api_key: str | None) -> None:
    """Click-based setup flow (non-TUI). Self-contained: config + post-config."""
    if mode is None:
        click.echo("Welcome to SmartMemory.\n")
        click.echo("Where do you want to store memories?")
        click.echo("  1. Local  — on this machine, no account needed")
        click.echo("  2. Remote — SmartMemory hosted service (api.smartmemory.ai)")
        choice = click.prompt(">", type=click.Choice(["1", "2"]), default="1")
        mode = "local" if choice == "1" else "remote"

    if mode == "remote":
        _setup_remote(api_key)
        click.echo("\nSetup complete.")
    else:
        _setup_local()
        _start_daemon_local()


def _can_run_tui() -> bool:
    """Check if we can launch a Textual TUI."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if os.environ.get("CI"):
        return False
    try:
        import textual  # noqa: F401
        return True
    except ImportError:
        return False


def _start_daemon_local() -> None:
    """Start daemon automatically in local mode.

    On macOS: let launchd own the process (RunAtLoad starts it immediately).
    On other platforms: start manually via start_daemon().
    Never do both — that causes a duplicate-bind crash loop.
    """
    import time

    launchd_ok = _install_launchd_plist()
    if launchd_ok:
        click.echo("\nWaiting for launchd to start daemon...")
        try:
            from smartmemory_app.daemon import is_running
            for _ in range(60):
                if is_running():
                    break
                time.sleep(1)
            if is_running():
                click.echo("SmartMemory is running (managed by launchd).")
            else:
                click.echo("Warning: daemon not yet healthy. Check: smartmemory status")
        except Exception as e:
            click.echo(f"Warning: health check failed ({e}). Check: smartmemory status")
    else:
        click.echo("\nStarting SmartMemory daemon...")
        try:
            from smartmemory_app.daemon import start_daemon
            start_daemon()
            click.echo("SmartMemory is running.")
        except Exception as e:
            click.echo(f"Warning: daemon start failed ({e}). Start manually: smartmemory start")


def _apply_setup_result(result: SetupResult, on_step: Callable[[str], None] | None = None) -> None:
    """Apply a SetupResult from TUI. Saves config + runs post-config steps.

    Args:
        result: Collected answers from the TUI.
        on_step: Optional callback called with step name after each step completes.
                 Used by ProgressScreen for per-step UI updates.
    """
    from smartmemory_app.config import SmartMemoryConfig, save_config

    data_dir = str(Path(result.data_dir).expanduser())

    cfg = SmartMemoryConfig(
        mode=result.mode,
        coreference=result.coreference,
        llm_provider=result.llm_provider,
        llm_model=result.llm_model,
        embedding_provider=result.embedding_provider,
        spacy_model=result.spacy_model,
        data_dir=data_dir,
    )
    save_config(cfg)
    if on_step:
        on_step("Config written")

    _ensure_spacy(result.spacy_model)
    if on_step:
        on_step("spaCy model ready")

    _copy_hooks()
    if on_step:
        on_step("Hooks installed")

    _copy_skills()
    if on_step:
        on_step("Skills installed")

    _register_hooks()
    if on_step:
        on_step("Hooks registered")

    _seed_data_dir(data_dir)
    if on_step:
        on_step("Patterns seeded")


# ── Mode setup helpers ────────────────────────────────────────────────────────


def _read_env_from_profile(name: str) -> str:
    """Try to read an export value from the user's shell profile."""
    shell = os.environ.get("SHELL", "/bin/zsh")
    if "zsh" in shell:
        profile = Path.home() / ".zshrc"
    elif "bash" in shell:
        profile = Path.home() / ".bashrc"
    else:
        profile = Path.home() / ".profile"
    try:
        for line in profile.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(f"export {name}="):
                # Extract value — handles export KEY="value" and export KEY=value
                val = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                return val
    except Exception:
        pass
    return ""


def _persist_env_var(name: str, value: str) -> None:
    """Write an export line to the user's shell profile so the key persists.

    If the key already exists in the profile, replaces the line.
    Also sets os.environ so the current process picks it up immediately.
    """
    shell = os.environ.get("SHELL", "/bin/zsh")
    if "zsh" in shell:
        profile = Path.home() / ".zshrc"
    elif "bash" in shell:
        profile = Path.home() / ".bashrc"
    else:
        profile = Path.home() / ".profile"

    export_line = f'export {name}="{value}"'

    try:
        existing = profile.read_text() if profile.exists() else ""
        lines = existing.splitlines()

        # Replace existing line or append
        replaced = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"export {name}="):
                lines[i] = export_line
                replaced = True
                break

        if replaced:
            profile.write_text("\n".join(lines) + "\n")
        else:
            with open(profile, "a") as f:
                f.write(f"\n{export_line}\n")

        click.echo(f"  Written to {profile}")
    except Exception as e:
        click.echo(f"  Warning: could not write to {profile}: {e}")
        click.echo(f"  Add this manually: {export_line}")

    # Set in current process so daemon picks it up
    os.environ[name] = value


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

    # Prompt for API key if the provider needs one
    _LLM_KEY_ENVVAR = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    key_envvar = _LLM_KEY_ENVVAR.get(llm)
    if key_envvar:
        # Check env, then keychain, then shell profile
        existing = os.environ.get(key_envvar, "").strip()
        if not existing:
            try:
                import keyring
                existing = (keyring.get_password("smartmemory", key_envvar) or "").strip()
                if existing:
                    os.environ[key_envvar] = existing  # load into current process
            except Exception:
                pass
        if not existing:
            # Try reading from shell profile as last resort
            existing = _read_env_from_profile(key_envvar)
            if existing:
                os.environ[key_envvar] = existing
        if existing:
            masked = existing[:8] + "..." + existing[-4:] if len(existing) > 12 else "***"
            api_key = click.prompt(
                f"\n{key_envvar} ({masked}). Press Enter to keep, or paste new key",
                default="",
            )
        else:
            api_key = click.prompt(
                f"\n{key_envvar} not set. Enter your API key (Enter to skip)",
                default="",
            )
        if api_key.strip():
            _persist_env_var(key_envvar, api_key.strip())
            # Also store in OS keychain so daemon can load it without sourcing shell
            try:
                import keyring
                keyring.set_password("smartmemory", key_envvar, api_key.strip())
                click.echo(f"  Also stored in OS keychain.")
            except Exception:
                pass  # keyring optional — shell profile is the primary store

    embedding = click.prompt(
        "Embedding provider? (local / openai / ollama)",
        default="local",
    )

    click.echo("\nspaCy model (used for entity extraction):")
    click.echo("  sm  — 15MB, fast, good for most use cases")
    click.echo("  md  — 45MB, better NER accuracy")
    click.echo("  lg  — 590MB, best accuracy, slower to download")
    spacy_size = click.prompt(
        "Which size?",
        type=click.Choice(["sm", "md", "lg"], case_sensitive=False),
        default="sm",
    )
    spacy_model = f"en_core_web_{spacy_size}"

    data_dir = click.prompt("Default data directory", default="~/.smartmemory")

    cfg = SmartMemoryConfig(
        mode="local",
        coreference=coref,
        llm_provider=llm,
        embedding_provider=embedding,
        spacy_model=spacy_model,
        data_dir=data_dir,
    )
    save_config(cfg)

    # Wire Claude Code hooks (preserved from original setup.py)
    _ensure_spacy(spacy_model)
    _copy_hooks()
    _copy_skills()
    _register_hooks()
    _seed_data_dir()

    # Start (or restart) daemon so it picks up the new config + keys
    from smartmemory_app.daemon import is_running, stop_daemon, start_daemon
    if is_running(require_healthy=False):
        click.echo("\nRestarting daemon with new config...")
        stop_daemon()
        import time
        time.sleep(2)
    click.echo("Starting daemon...")
    start_daemon()
    click.echo("Done. SmartMemory is ready.")


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

    # DIST-AGENT-HOOKS-1: Wire hooks for remote mode too.
    # Hooks call CLI which falls back to remote API via storage dispatch.
    _copy_hooks()
    _copy_skills()
    _register_hooks()
    click.echo("Lifecycle hooks installed. Auto-recall and observation active.")


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
    """Remove SmartMemory hooks, skills, launchd plist, and optionally data from ~/.smartmemory/."""
    # Stop daemon before removing anything
    try:
        from smartmemory_app.daemon import stop_daemon
        stop_daemon()
    except Exception:
        pass
    _uninstall_launchd_plist()
    _deregister_hooks()
    _remove_hooks()
    _remove_skills()
    if not keep_data:
        _remove_data_dir()
    click.echo("SmartMemory removed. Restart Claude Code to deactivate hooks.")


# ── Install helpers ───────────────────────────────────────────────────────────


_SPACY_MODEL_SIZES = {
    "en_core_web_sm": "~15MB",
    "en_core_web_md": "~45MB",
    "en_core_web_lg": "~590MB",
}


def _ensure_spacy(model: str = "en_core_web_sm") -> None:
    try:
        import spacy
        spacy.load(model)
    except (ImportError, OSError):
        size = _SPACY_MODEL_SIZES.get(model, "")
        click.echo(f"Downloading {model} ({size})...")
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model],
            capture_output=True,
        )
        if result.returncode != 0:
            click.echo(
                f"WARNING: spaCy model download failed. Run manually:\n"
                f"  python -m spacy download {model}",
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


def _seed_data_dir(data_dir: str | None = None) -> None:
    """Seed entity patterns in the data directory.

    Args:
        data_dir: Explicit data directory path. When provided, expanduser() is applied.
                  When None, falls back to SMARTMEMORY_DATA_DIR env var or ~/.smartmemory.
    """
    if data_dir is not None:
        resolved = Path(data_dir).expanduser()
    else:
        raw = os.environ.get("SMARTMEMORY_DATA_DIR")
        resolved = Path(raw) if raw else DATA_DIR
    resolved.mkdir(parents=True, exist_ok=True)
    from smartmemory_app.patterns import JSONLPatternStore
    JSONLPatternStore(resolved)  # seeds entity_patterns.jsonl if absent (side effect of __init__)


def _install_launchd_plist() -> bool:
    """Install launchd plist for auto-start + crash recovery (macOS only).

    Substitutes {PYTHON_PATH}, {DAEMON_PORT}, {DATA_DIR}, {BIN_DIR} into the
    template plist and writes to ~/Library/LaunchAgents/. Then loads the plist
    with launchctl so launchd starts managing the daemon immediately.

    Returns True if the plist was loaded successfully (caller should NOT also
    call start_daemon — launchd owns the process via RunAtLoad).
    Returns False on non-macOS, missing template, or launchctl failure.
    """
    import platform
    if platform.system() != "Darwin":
        return False  # launchd is macOS-only

    if not PLIST_TEMPLATE.exists():
        click.echo("Warning: launchd plist template not found — skipping auto-start setup.")
        return False

    from smartmemory_app.config import load_config
    cfg = load_config()

    python_path = sys.executable
    bin_dir = str(Path(python_path).parent)
    # Use config as source of truth for data_dir — this is what the user chose
    # during setup (or the default). Env var SMARTMEMORY_DATA_DIR is a runtime
    # override that shouldn't be baked into a persistent plist.
    data_dir = str(Path(cfg.data_dir).expanduser())

    # Resolve GROQ_API_KEY from env or keyring
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        try:
            import keyring
            groq_key = keyring.get_password("smartmemory", "groq_api_key") or ""
        except Exception:
            pass

    replacements = {
        "{PYTHON_PATH}": python_path,
        "{DAEMON_PORT}": str(cfg.daemon_port),
        "{DATA_DIR}": data_dir,
        "{BIN_DIR}": bin_dir,
        "{GROQ_API_KEY}": groq_key,
    }

    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    # Install both daemon and worker plists
    templates = [
        (PLIST_TEMPLATE, PLIST_NAME, "daemon"),
    ]
    if WORKER_PLIST_TEMPLATE.exists():
        templates.append((WORKER_PLIST_TEMPLATE, WORKER_PLIST_NAME, "worker"))

    for template_path, plist_name, label in templates:
        if not template_path.exists():
            click.echo(f"Warning: {label} plist template not found — skipping.")
            continue

        content = template_path.read_text()
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)

        plist_dest = LAUNCH_AGENTS_DIR / plist_name

        # Unload existing plist before overwriting (launchctl requires this)
        if plist_dest.exists():
            subprocess.run(
                ["launchctl", "unload", str(plist_dest)],
                capture_output=True,
            )

        plist_dest.write_text(content)

        result = subprocess.run(
            ["launchctl", "load", str(plist_dest)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            click.echo(f"Installed launchd plist: {plist_dest}")
        else:
            click.echo(f"Warning: launchctl load failed for {label}: {result.stderr.strip()}")
            click.echo(f"Load manually: launchctl load {plist_dest}")
            all_ok = False

    if all_ok:
        click.echo("SmartMemory will auto-start on login and restart on crash.")
        if groq_key:
            click.echo("Enrichment worker enabled (GROQ_API_KEY found).")
        else:
            click.echo("Warning: GROQ_API_KEY not found — enrichment worker won't extract relations.")
    return all_ok


def _uninstall_launchd_plist() -> None:
    """Unload and remove daemon + worker launchd plists (macOS only)."""
    import platform
    if platform.system() != "Darwin":
        return

    for plist_name, label in [(PLIST_NAME, "daemon"), (WORKER_PLIST_NAME, "worker")]:
        plist_dest = LAUNCH_AGENTS_DIR / plist_name
        if not plist_dest.exists():
            continue
        subprocess.run(
            ["launchctl", "unload", str(plist_dest)],
            capture_output=True,
        )
        plist_dest.unlink(missing_ok=True)
        click.echo(f"Removed {label} launchd plist.")

    click.echo("SmartMemory will no longer auto-start.")


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
