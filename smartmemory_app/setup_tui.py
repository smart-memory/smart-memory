"""DIST-SETUP-TUI-1: Textual TUI for `smartmemory setup`.

Arrow-key selection for mode, LLM provider, embedding provider. Live model
discovery for ollama/lmstudio. Summary screen with edit/toggle. Progress
screen with per-step checklist + daemon spinner.

Entry point: run_setup_tui() -> SetupResult | None
"""
import os
from dataclasses import dataclass, field

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    OptionList,
    Static,
    Switch,
)
from textual.widgets.option_list import Option

from smartmemory_app.setup import SetupResult


# ── Provider definitions ──────────────────────────────────────────────────

LLM_PROVIDERS = [
    ("groq", "Free tier, fast inference (recommended)"),
    ("claude-agent", "Claude Agent SDK, OAuth (no API key)"),
    ("openai", "OpenAI API"),
    ("anthropic", "Anthropic API"),
    ("ollama", "Local, free (llama3.1, mistral)"),
    ("lmstudio", "Local, OpenAI-compatible endpoint"),
    ("none", "EntityRuler only (very limited)"),
]

EMBEDDING_PROVIDERS = [
    ("local", "Built-in (no API key needed)"),
    ("openai", "OpenAI embeddings (1536-dim)"),
    ("ollama", "Local Ollama embeddings"),
]

_KEY_VARS = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

LOCAL_MODEL_PROVIDERS = {"ollama", "lmstudio"}


def _detect_keys() -> str:
    """Build a status line showing detected API keys."""
    parts = []
    for provider, var in _KEY_VARS.items():
        found = bool(os.environ.get(var))
        icon = "[green]✓[/]" if found else "[dim]✗[/]"
        parts.append(f"{var} {icon}")
    return "  ".join(parts)


# ── Screens ───────────────────────────────────────────────────────────────


class WelcomeScreen(Screen):
    BINDINGS = [Binding("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Vertical(id="welcome-box"):
                yield Static("\n[bold]Welcome to SmartMemory[/bold]\n", id="title")
                yield Static("Where do you want to store memories?\n")
                yield OptionList(
                    Option("Local — on this machine, no account needed", id="local"),
                    Option("Remote — SmartMemory hosted service", id="remote"),
                    id="mode-list",
                )
        yield Footer()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id == "remote":
            self.app.exit(SetupResult(mode="remote"))
        else:
            self.app.push_screen(LLMScreen())

    def action_quit(self) -> None:
        self.app.exit(None)


class LLMScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Vertical(id="llm-box"):
                yield Static(
                    "\n[bold]LLM Provider[/bold]\n\n"
                    "SmartMemory uses an LLM to extract entities\n"
                    "and relationships from your text.\n"
                )
                options = [
                    Option(f"{pid:<14s} {desc}", id=pid)
                    for pid, desc in LLM_PROVIDERS
                ]
                yield OptionList(*options, id="llm-list")
                yield Static(f"\nKeys: {_detect_keys()}\n", id="key-badges")
        yield Footer()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        provider = str(event.option.id)
        self.app._result.llm_provider = provider
        if provider in LOCAL_MODEL_PROVIDERS:
            self.app.push_screen(ModelScreen(provider))
        else:
            self.app.push_screen(EmbeddingScreen())

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_quit(self) -> None:
        self.app.exit(None)


class ModelScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, provider: str) -> None:
        super().__init__()
        self._provider = provider

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Vertical(id="model-box"):
                yield Static(f"\n[bold]Model Selection[/bold]\n")
                yield Static(f"Discovering models on {self._provider}...\n", id="status")
                yield LoadingIndicator(id="loader")
                yield OptionList(id="model-list")
                yield Button("Skip", id="skip-btn", variant="default")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#model-list", OptionList).display = False
        self.query_one("#skip-btn", Button).display = False
        self._discover()

    @work(exclusive=True, group="discover")
    async def _discover(self) -> None:
        models = await _discover_models(self._provider)
        loader = self.query_one("#loader", LoadingIndicator)
        status = self.query_one("#status", Static)
        model_list = self.query_one("#model-list", OptionList)
        skip_btn = self.query_one("#skip-btn", Button)

        loader.display = False
        if models:
            status.update(f"Found {len(models)} models:\n")
            model_list.clear_options()
            for m in models:
                name = m["name"]
                size = m.get("size", "")
                label = f"{name:<30s} {size}" if size else name
                model_list.add_option(Option(label, id=name))
            model_list.display = True
        else:
            status.update(f"[yellow]Could not reach {self._provider}[/yellow]\n")
            skip_btn.display = True

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.app._result.llm_model = str(event.option.id)
        self.app.push_screen(EmbeddingScreen())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "skip-btn":
            self.app.push_screen(EmbeddingScreen())

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_quit(self) -> None:
        self.app.exit(None)


class EmbeddingScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Vertical(id="embed-box"):
                yield Static("\n[bold]Embedding Provider[/bold]\n")
                has_openai = bool(os.environ.get("OPENAI_API_KEY"))
                if has_openai:
                    yield Static("[green]OPENAI_API_KEY detected[/green] — defaulting to openai\n")
                options = [
                    Option(f"{pid:<10s} {desc}", id=pid)
                    for pid, desc in EMBEDDING_PROVIDERS
                ]
                ol = OptionList(*options, id="embed-list")
                yield ol
        yield Footer()

    def on_mount(self) -> None:
        # Highlight openai if key detected
        if os.environ.get("OPENAI_API_KEY"):
            ol = self.query_one("#embed-list", OptionList)
            ol.highlighted = 1  # openai is index 1

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.app._result.embedding_provider = str(event.option.id)
        self.app.push_screen(SummaryScreen())

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_quit(self) -> None:
        self.app.exit(None)


class SummaryScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        r = self.app._result
        with Center():
            with Vertical(id="summary-box"):
                yield Static("\n[bold]Configuration Summary[/bold]\n")
                yield Static(f"  Mode:        {r.mode}")
                yield Static(f"  LLM:         {r.llm_provider}" + (f" ({r.llm_model})" if r.llm_model else ""))
                yield Static(f"  Embeddings:  {r.embedding_provider}")
                yield Static("")
                yield Static("  Data directory:")
                yield Input(value=r.data_dir, id="data-dir-input")
                yield Static("")
                with Vertical(id="coref-row"):
                    yield Static("  Coreference resolution (~500MB model):", id="coref-label")
                    yield Switch(value=r.coreference, id="coref-switch")
                yield Static("")
                with Center():
                    yield Button("Confirm", id="confirm-btn", variant="success")
                    yield Button("Back", id="back-btn", variant="default")
                    yield Button("Cancel", id="cancel-btn", variant="error")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-btn":
            r = self.app._result
            r.data_dir = self.query_one("#data-dir-input", Input).value
            r.coreference = self.query_one("#coref-switch", Switch).value
            self.app.push_screen(ProgressScreen())
        elif event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "cancel-btn":
            self.app.exit(None)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_quit(self) -> None:
        self.app.exit(None)


class ProgressScreen(Screen):

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Vertical(id="progress-box"):
                yield Static("\n[bold]Setting up SmartMemory...[/bold]\n")
                yield Static("○ Config written", id="step-config")
                yield Static("○ spaCy model ready", id="step-spacy")
                yield Static("○ Hooks installed", id="step-hooks")
                yield Static("○ Skills installed", id="step-skills")
                yield Static("○ Hooks registered", id="step-registered")
                yield Static("○ Patterns seeded", id="step-patterns")
                yield Static("○ Daemon started", id="step-daemon")
                yield Static("", id="final-status")
        yield Footer()

    def on_mount(self) -> None:
        self._run_setup()

    @work(thread=True)
    def _run_setup(self) -> None:
        from smartmemory_app.setup import _apply_setup_result, _start_daemon_local

        step_map = {
            "Config written": "step-config",
            "spaCy model ready": "step-spacy",
            "Hooks installed": "step-hooks",
            "Skills installed": "step-skills",
            "Hooks registered": "step-registered",
            "Patterns seeded": "step-patterns",
        }

        def on_step(name: str) -> None:
            widget_id = step_map.get(name)
            if widget_id:
                self.app.call_from_thread(
                    self.query_one(f"#{widget_id}", Static).update,
                    f"[green]✓[/green] {name}",
                )

        try:
            _apply_setup_result(self.app._result, on_step=on_step)

            self.app.call_from_thread(
                self.query_one("#step-daemon", Static).update,
                "● Starting daemon...",
            )
            _start_daemon_local()
            self.app.call_from_thread(
                self.query_one("#step-daemon", Static).update,
                "[green]✓[/green] Daemon started",
            )

            # Show final status
            status_text = "\n[bold green]SmartMemory is ready![/bold green]\n\nTry: [cyan]smartmemory persist \"hello world\"[/cyan]\n\nPress any key to exit."
            self.app.call_from_thread(
                self.query_one("#final-status", Static).update,
                status_text,
            )
        except Exception as e:
            self.app.call_from_thread(
                self.query_one("#final-status", Static).update,
                f"\n[bold red]Setup failed: {e}[/bold red]\n\nPress any key to exit.",
            )

    def on_key(self) -> None:
        final = self.query_one("#final-status", Static)
        if str(final.renderable).strip():
            self.app.exit(self.app._result)


# ── Model discovery ───────────────────────────────────────────────────────


async def _discover_models(provider: str) -> list[dict]:
    """Fetch available models from local/API provider."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            if provider == "ollama":
                r = await client.get("http://localhost:11434/api/tags")
                r.raise_for_status()
                models = r.json().get("models", [])
                return [
                    {"name": m["name"], "size": _format_size(m.get("size", 0))}
                    for m in models
                ]
            elif provider == "lmstudio":
                r = await client.get("http://localhost:1234/v1/models")
                r.raise_for_status()
                return [{"name": m["id"]} for m in r.json().get("data", [])]
    except Exception:
        pass
    return []


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable (e.g., 4.7 GB)."""
    if size_bytes <= 0:
        return ""
    gb = size_bytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = size_bytes / (1024 ** 2)
    return f"{mb:.0f} MB"


# ── App ───────────────────────────────────────────────────────────────────


class SetupApp(App):
    TITLE = "SmartMemory Setup"
    CSS = """
    #welcome-box, #llm-box, #model-box, #embed-box, #summary-box, #progress-box {
        width: 60;
        max-width: 80;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }
    #title {
        text-align: center;
    }
    OptionList {
        height: auto;
        max-height: 12;
    }
    #data-dir-input {
        width: 40;
    }
    #coref-row {
        height: 3;
        layout: horizontal;
    }
    #coref-label {
        width: 1fr;
    }
    Button {
        margin: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._result = SetupResult()

    def on_mount(self) -> None:
        self.push_screen(WelcomeScreen())


# ── Entry point ───────────────────────────────────────────────────────────


def run_setup_tui() -> SetupResult | None:
    """Launch the Textual TUI. Returns SetupResult on success, None on cancel."""
    app = SetupApp()
    return app.run()
