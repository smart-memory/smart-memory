"""SmartMemory CLI — daemon management + memory operations.

DIST-DAEMON-1: All memory commands (add, ingest, search, recall) try the
daemon HTTP API first (<200ms). Falls back to direct storage calls if daemon
is not running (~22s cold start).
"""

import logging
import json

import click

log = logging.getLogger(__name__)


def _parse_extra_props(args: list[str]) -> dict[str, str]:
    """Parse Click extra args (--key value pairs) into a property dict."""
    props = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            props[args[i][2:]] = args[i + 1]
            i += 2
        else:
            i += 1
    return props


def _daemon_url() -> str:
    from smartmemory_app.config import load_config

    return f"http://127.0.0.1:{load_config().daemon_port}"


def _daemon_request(method: str, path: str, timeout: int = 120, **kwargs):
    """Try daemon HTTP API. Returns parsed JSON or None if daemon unreachable.

    Retries once on connection drop — handles the case where the daemon
    auto-restarts after a pip upgrade (version guard middleware exits the
    process, launchd restarts it within ~5s).
    """
    import httpx
    import time

    for attempt in range(2):
        try:
            r = httpx.request(method, f"{_daemon_url()}{path}", timeout=timeout, **kwargs)
            r.raise_for_status()
            return r.json() if r.status_code != 204 else {}
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError):
            if attempt == 0:
                time.sleep(2)  # wait for launchd to restart daemon (~1.2s startup)
                continue
            return None  # still down after retry — fall back to direct
        except httpx.HTTPStatusError as e:
            # Surface server errors (e.g. 501 for unsupported filters) to caller
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            raise click.ClickException(detail)
        except httpx.ReadTimeout:
            return None


@click.group()
def cli() -> None:
    """SmartMemory — persistent AI memory system."""


# Register setup/uninstall commands from setup module
from smartmemory_app.setup import setup as _setup_cmd, uninstall as _uninstall_cmd  # noqa: E402

cli.add_command(_setup_cmd, name="setup")
cli.add_command(_uninstall_cmd, name="uninstall")


# ── Daemon lifecycle ────────────────────────────────────────────────────────


@cli.command("start")
def start_cmd() -> None:
    """Start the SmartMemory daemon."""
    from smartmemory_app.daemon import start_daemon, is_running

    if is_running():
        click.echo("SmartMemory daemon is already running.")
        return
    click.echo("Starting SmartMemory daemon (loading models)...")
    try:
        start_daemon()
        click.echo("Daemon ready.")
    except Exception as e:
        click.echo(f"Failed to start daemon: {e}", err=True)
        raise SystemExit(1)


@cli.command("stop")
def stop_cmd() -> None:
    """Stop the SmartMemory daemon."""
    from smartmemory_app.daemon import stop_daemon, is_running

    if not is_running(require_healthy=False):
        click.echo("Daemon is not running.")
        return
    stop_daemon()
    click.echo("Daemon stopped.")


@cli.command("restart")
def restart_cmd() -> None:
    """Restart the SmartMemory daemon."""
    from smartmemory_app.daemon import stop_daemon, start_daemon, is_running

    if is_running(require_healthy=False):
        click.echo("Stopping daemon...")
        stop_daemon()
    click.echo("Starting daemon...")
    start_daemon()
    click.echo("Daemon ready.")


@cli.command("status")
def status_cmd() -> None:
    """Show SmartMemory daemon status."""
    from smartmemory_app.daemon import get_status

    info = get_status()
    if info is None:
        click.echo("SmartMemory daemon is not running.")
        click.echo("Start with: smartmemory start")
        return
    click.echo(f"SmartMemory daemon: {info.get('status', '?')}")
    click.echo(f"  Memories:   {info.get('memories', '?')}")
    click.echo(f"  LLM:        {info.get('llm_provider', '?')}")
    click.echo(f"  Embeddings: {info.get('embedding_provider', '?')}")
    click.echo(f"  PID:        {info.get('pid', '?')}")
    async_info = info.get("async_enrichment", {})
    if async_info.get("enabled"):
        pending = async_info.get("pending", 0)
        done = async_info.get("done", 0)
        failed = async_info.get("failed", 0)
        click.echo(f"  Queue:      pending={pending}, done={done}, failed={failed}")
    else:
        click.echo("  Queue:      (no table)")


@cli.command("viewer")
@click.option("--port", default=None, type=int, help="Port override.")
def viewer_cmd(port: int | None) -> None:
    """Open the knowledge graph viewer in the browser."""
    import webbrowser
    from smartmemory_app.daemon import start_daemon, is_running, _port

    if not is_running():
        click.echo("Starting daemon...")
        start_daemon()
    p = port or _port()
    webbrowser.open(f"http://localhost:{p}")


@cli.command("worker")
@click.option("--loop", is_flag=True, help="Poll continuously instead of drain-and-exit.")
def worker_cmd(loop: bool) -> None:
    """Run the enrichment worker (Tier 2 LLM extraction).

    Drains the SQLite enrichment queue. Use --loop for continuous polling.
    """
    from smartmemory_app.enrichment_worker import drain_queue, run_loop

    if loop:
        run_loop()
    else:
        n = drain_queue()
        click.echo(f"Processed {n} jobs")


# ── Memory operations ───────────────────────────────────────────────────────

_VALID_MEMORY_TYPES = {
    "working", "semantic", "episodic", "procedural", "zettel",
    "reasoning", "opinion", "observation", "decision",
}


def _validate_memory_type(ctx, param, value: str) -> str:
    if value not in _VALID_MEMORY_TYPES:
        sorted_types = sorted(_VALID_MEMORY_TYPES)
        raise click.BadParameter(
            f"Invalid type '{value}'. Valid types: {', '.join(sorted_types)}"
        )
    return value


@cli.command("add", context_settings=dict(
    ignore_unknown_options=True, allow_extra_args=True,
))
@click.argument("text", default="-")
@click.option("--type", "memory_type", default="episodic", show_default=True, callback=_validate_memory_type)
@click.option("--all", "as_whole", is_flag=True, help="Add stdin as one memory instead of line-by-line.")
@click.pass_context
def add_cmd(ctx, text: str, memory_type: str, as_whole: bool) -> None:
    """Add text as a memory. Use - or pipe stdin to read from a file.

    When reading from stdin, each non-empty line becomes a separate memory.
    Use --all to add the entire input as a single memory instead.

    Examples:
        smartmemory add "Alice leads Project Atlas"
        smartmemory add - < notes.txt              # line-by-line
        smartmemory add --all - < document.txt     # whole file
        echo "some text" | smartmemory add -

    Supports arbitrary property flags: --project atlas --domain legal
    """
    import sys

    if text == "-":
        if sys.stdin.isatty():
            raise click.ClickException("No input. Pipe text or use: smartmemory add \"text\"")
        raw = sys.stdin.read()
        if not raw.strip():
            raise click.ClickException("Content cannot be empty.")
        chunks = [raw.strip()] if as_whole else [l.strip() for l in raw.splitlines() if l.strip()]
        if not chunks:
            raise click.ClickException("Content cannot be empty.")
        props = _parse_extra_props(ctx.args)
        ids = []
        for chunk in chunks:
            body: dict = {"content": chunk, "memory_type": memory_type}
            if props:
                body["properties"] = props
            result = _daemon_request("POST", "/memory/ingest", json=body)
            if result:
                ids.append(result.get("item_id", "?"))
            else:
                from smartmemory_app.storage import ingest

                ids.append(ingest(chunk, memory_type, properties=props))
        click.echo(f"Added {len(ids)} memories")
        for item_id in ids:
            click.echo(item_id)
        return
    if not text.strip():
        raise click.ClickException("Content cannot be empty.")
    props = _parse_extra_props(ctx.args)
    body: dict = {"content": text, "memory_type": memory_type}
    if props:
        body["properties"] = props
    result = _daemon_request("POST", "/memory/ingest", json=body)
    if result:
        click.echo(result.get("item_id", "?"))
    else:
        from smartmemory_app.storage import ingest

        click.echo(ingest(text, memory_type, properties=props))


@cli.command("recall")
@click.option("--cwd", default=None, help="Current working directory for context.")
@click.option("--top-k", default=10, show_default=True)
def recall_cmd(cwd: str, top_k: int) -> None:
    """Recall memories (SessionStart hook)."""
    result = _daemon_request(
        "GET", "/memory/recall", params={"cwd": cwd or "", "top_k": top_k}
    )
    if result:
        context = result.get("context", "")
    else:
        from smartmemory_app.storage import recall

        context = recall(cwd, top_k)
    if context:
        click.echo(context)


@cli.command("search", context_settings=dict(
    ignore_unknown_options=True, allow_extra_args=True,
))
@click.argument("query")
@click.option("--top-k", default=5, show_default=True)
@click.pass_context
def search_cmd(ctx, query: str, top_k: int) -> None:
    """Search memories by semantic similarity. Use '*' to list all.

    Supports property filters: --project atlas --domain legal
    """
    props = _parse_extra_props(ctx.args)
    body: dict = {"query": query, "top_k": top_k}
    if props:
        body["filters"] = props
    results = _daemon_request("POST", "/memory/search", json=body)
    if results is None:
        from smartmemory_app.storage import search

        try:
            results = search(query, top_k, filters=props)
        except NotImplementedError as e:
            raise click.ClickException(str(e))
    if not results:
        click.echo("No results.")
        return
    for r in results:
        content = r.get("content", "")[:200]
        mem_type = r.get("memory_type", "?")
        item_id = r.get("item_id", "?")
        # CORE-PROPS-1: Tilde marker for low-confidence memories
        conf = r.get("confidence", 1.0)
        conf_marker = "~" if isinstance(conf, (int, float)) and conf < 0.5 else ""
        # CORE-PROPS-1 Phase 2: stale marker
        stale_marker = "⚠" if r.get("stale") else ""
        click.echo(f"{stale_marker}{conf_marker}[{mem_type}] {item_id[:8]}  {content}")


@cli.command("get")
@click.argument("item_id")
def get_cmd(item_id: str) -> None:
    """Fetch a single memory by item ID."""
    try:
        result = _daemon_request("GET", f"/memory/{item_id}")
    except click.ClickException:
        raise  # surface daemon HTTP errors cleanly
    except Exception:
        result = None

    if result is None:
        from smartmemory_app.storage import get

        result = get(item_id)

    if not result:
        click.echo("Memory not found.", err=True)
        raise SystemExit(1)

    click.echo(json.dumps(result, indent=2, sort_keys=True))


# ── Discovery + config ──────────────────────────────────────────────────────


@cli.command("models")
@click.option(
    "--provider",
    default=None,
    help="Filter by provider (ollama, lmstudio, groq, openai)",
)
def models_cmd(provider: str | None) -> None:
    """List available models from local and cloud providers."""
    import httpx

    providers_to_check = (
        [provider] if provider else ["ollama", "lmstudio", "groq", "openai"]
    )

    for p in providers_to_check:
        if p == "ollama":
            try:
                r = httpx.get("http://localhost:11434/api/tags", timeout=3)
                r.raise_for_status()
                models = r.json().get("models", [])
                if models:
                    click.echo(f"\nOllama ({len(models)} models):")
                    for m in models:
                        name = m.get("name", "?")
                        size = m.get("size", 0)
                        size_gb = f"{size / 1e9:.1f}GB" if size else "?"
                        click.echo(f"  ollama/{name}  ({size_gb})")
                else:
                    click.echo("\nOllama: running but no models pulled")
            except Exception:
                click.echo("\nOllama: not running (start with: ollama serve)")

        elif p == "lmstudio":
            try:
                r = httpx.get("http://localhost:1234/v1/models", timeout=3)
                r.raise_for_status()
                models = r.json().get("data", [])
                if models:
                    click.echo(f"\nLM Studio ({len(models)} models):")
                    for m in models:
                        click.echo(f"  lmstudio/{m.get('id', '?')}")
                else:
                    click.echo("\nLM Studio: running but no models loaded")
            except Exception:
                click.echo(
                    "\nLM Studio: not running (start LM Studio and load a model)"
                )

        elif p == "groq":
            import os

            key = os.environ.get("GROQ_API_KEY")
            if not key:
                click.echo("\nGroq: GROQ_API_KEY not set (get one at console.groq.com)")
                continue
            try:
                r = httpx.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=5,
                )
                r.raise_for_status()
                models = sorted(r.json().get("data", []), key=lambda m: m.get("id", ""))
                if models:
                    click.echo(f"\nGroq ({len(models)} models):")
                    for m in models:
                        click.echo(f"  groq/{m.get('id', '?')}")
            except Exception as e:
                click.echo(f"\nGroq: API error ({e})")

        elif p == "openai":
            import os

            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                click.echo("\nOpenAI: OPENAI_API_KEY not set")
                continue
            try:
                r = httpx.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=5,
                )
                r.raise_for_status()
                models = sorted(r.json().get("data", []), key=lambda m: m.get("id", ""))
                chat_models = [
                    m
                    for m in models
                    if any(
                        m.get("id", "").startswith(prefix)
                        for prefix in ("gpt-4", "gpt-3.5", "o1", "o3", "o4")
                    )
                ]
                if chat_models:
                    click.echo(f"\nOpenAI ({len(chat_models)} chat models):")
                    for m in chat_models:
                        click.echo(f"  openai/{m.get('id', '?')}")
            except Exception as e:
                click.echo(f"\nOpenAI: API error ({e})")

        elif p == "claude-agent":
            click.echo(
                "\nClaude Agent SDK: uses Claude Code OAuth (no model selection needed)"
            )

        elif p == "anthropic":
            click.echo(
                "\nAnthropic: claude-3-5-haiku-latest, claude-sonnet-4-5-20250514, claude-opus-4-6-20250603"
            )


@cli.command("config")
@click.argument("key", required=False)
@click.argument("value", required=False)
def config_cmd(key: str | None, value: str | None) -> None:
    """View or change SmartMemory configuration.

    \b
    smartmemory config                  Show all settings
    smartmemory config llm_provider     Show one setting
    smartmemory config llm_provider groq  Change a setting
    """
    from smartmemory_app.config import load_config, save_config, config_path

    cfg = load_config()

    if key is None:
        from smartmemory_app import __version__ as wrapper_version
        try:
            from importlib.metadata import version as _pkg_version
            core_version = _pkg_version("smartmemory-core")
        except Exception:
            core_version = "?"
        click.echo(f"SmartMemory v{wrapper_version} (core {core_version})")
        click.echo(f"Config: {config_path()}\n")
        click.echo(f"  mode              = {cfg.mode}")
        click.echo(f"  llm_provider      = {cfg.llm_provider}")
        click.echo(f"  llm_model         = {cfg.llm_model or '(auto)'}")
        click.echo(f"  embedding_provider = {cfg.embedding_provider}")
        click.echo(f"  daemon_port       = {cfg.daemon_port}")
        click.echo(f"  coreference       = {cfg.coreference}")
        click.echo(f"  data_dir          = {cfg.data_dir}")
        click.echo(f"  api_url           = {cfg.api_url}")
        click.echo(f"  api_key_set       = {cfg.api_key_set}")
        click.echo(f"  team_id           = {cfg.team_id or '(none)'}")
        return

    settable = {
        "llm_provider": {
            "values": [
                "groq",
                "claude-agent",
                "anthropic",
                "openai",
                "ollama",
                "lmstudio",
                "none",
            ]
        },
        "llm_model": {"values": None},
        "embedding_provider": {"values": ["local", "openai", "ollama"]},
        "daemon_port": {"values": None},
        "coreference": {"values": ["true", "false"]},
        "data_dir": {"values": None},
        "mode": {"values": ["local", "remote"]},
    }

    if key not in settable:
        current = getattr(cfg, key, None)
        if current is not None:
            click.echo(f"{key} = {current}")
        else:
            click.echo(f"Unknown key: {key}")
            click.echo(f"Settable keys: {', '.join(settable)}")
        return

    if value is None:
        click.echo(f"{key} = {getattr(cfg, key)}")
        allowed = settable[key]["values"]
        if allowed:
            click.echo(f"Allowed: {', '.join(allowed)}")
        return

    allowed = settable[key]["values"]
    if allowed and value not in allowed:
        click.echo(f"Invalid value '{value}'. Allowed: {', '.join(allowed)}")
        return

    coerced: object = value
    if key == "coreference":
        coerced = value.lower() == "true"
    elif key == "daemon_port":
        coerced = int(value)

    setattr(cfg, key, coerced)
    save_config(cfg)
    click.echo(f"{key} = {value}")


# ── Admin subgroup ─────────────────────────────────────────────────────────


@cli.group("admin")
def admin_group() -> None:
    """Administrative commands (import, export, reindex, etc.)."""


@admin_group.command("import")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--mode",
    "import_mode",
    default="full",
    type=click.Choice(["full", "direct"]),
    show_default=True,
    help="full=ingest pipeline, direct=add() skip extraction",
)
@click.option(
    "--batch-size", default=100, show_default=True, help="Checkpoint interval"
)
@click.option("--resume", is_flag=True, help="Resume from last checkpoint")
@click.option("--dry-run", is_flag=True, help="Validate only, don't import")
@click.option("--domain-filter", default=None, help="Filter by metadata.domain")
def import_cmd(
    path: str,
    import_mode: str,
    batch_size: int,
    resume: bool,
    dry_run: bool,
    domain_filter: str | None,
) -> None:
    """Import a corpus JSONL file into SmartMemory."""
    from smartmemory.corpus.reader import CorpusReader
    from smartmemory.corpus.importer import CorpusImporter

    reader = CorpusReader(path)
    header = reader.read_header()
    click.echo(f"Corpus: source={header.source}, domain={header.domain or '(none)'}")

    if dry_run:
        click.echo("Dry run — validating records...")

    total = header.item_count or reader.count_records()

    try:
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TimeElapsedColumn,
            MofNCompleteColumn,
        )

        use_rich = True
    except ImportError:
        use_rich = False

    if not dry_run:
        from smartmemory_app.storage import get_memory

        sm = get_memory()
    else:
        sm = None

    importer = CorpusImporter(
        smart_memory=sm,
        mode=import_mode,
        batch_size=batch_size,
        domain_filter=domain_filter,
    )

    if use_rich and total > 0:
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Importing", total=total)

            def on_progress(stats, record):
                progress.update(task, completed=stats.total)

            stats = importer.run(
                path, resume=resume, dry_run=dry_run, progress_callback=on_progress
            )
    else:
        stats = importer.run(path, resume=resume, dry_run=dry_run)

    action = "Validated" if dry_run else "Imported"
    click.echo(
        f"{action} {stats.imported} records ({stats.errors} errors, {stats.rate:.1f} items/sec)"
    )


@admin_group.command("export")
@click.argument("path", type=click.Path())
@click.option("--memory-type", default=None, help="Filter by memory type")
@click.option(
    "--include-entities", is_flag=True, help="Include extracted entities/relations"
)
@click.option("--limit", default=0, help="Max records to export (0=all)")
@click.option(
    "--source", default="smartmemory-export", help="Source tag in corpus header"
)
@click.option("--domain", default="", help="Domain tag in corpus header")
def export_cmd(
    path: str,
    memory_type: str | None,
    include_entities: bool,
    limit: int,
    source: str,
    domain: str,
) -> None:
    """Export memories to a corpus JSONL file."""
    from smartmemory.corpus.exporter import CorpusExporter
    from smartmemory_app.storage import get_memory

    sm = get_memory()
    exporter = CorpusExporter(
        smart_memory=sm,
        memory_type=memory_type,
        include_entities=include_entities,
        limit=limit,
    )
    count = exporter.run(path, source=source, domain=domain)
    click.echo(f"Exported {count} records to {path}")


# ── Wikidata mining ────────────────────────────────────────────────────────


@admin_group.command("mine")
@click.option(
    "--domain", "domain_qid", default=None, help="Single P31 QID (e.g. Q9143)"
)
@click.option(
    "--domain-file",
    default=None,
    type=click.Path(exists=True),
    help="JSON config file with multiple domains",
)
@click.option("--all-defaults", is_flag=True, help="Mine expanded default domains")
@click.option("--incremental", default=None, help="Only entities newer than ISO date")
@click.option(
    "--limit", default=5000, show_default=True, help="Per-domain entity limit"
)
@click.option("--output", "-o", default="./mining-output/", help="Output directory")
@click.option(
    "--format",
    "output_format",
    default="both",
    type=click.Choice(["corpus", "snapshot", "both"]),
    show_default=True,
)
@click.option(
    "--quota-limit", default=0, help="Max SPARQL queries per run (0=unlimited)"
)
def mine_cmd(
    domain_qid: str | None,
    domain_file: str | None,
    all_defaults: bool,
    incremental: str | None,
    limit: int,
    output: str,
    output_format: str,
    quota_limit: int,
) -> None:
    """Mine Wikidata for entities via SPARQL."""
    from smartmemory.grounding.miner import (
        EXPANDED_DOMAINS,
        WikidataMiner,
        load_domain_config,
    )

    if domain_file:
        domains = load_domain_config(domain_file)
    elif domain_qid:
        domains = {domain_qid: domain_qid}
    elif all_defaults:
        domains = EXPANDED_DOMAINS
    else:
        click.echo("Specify --domain, --domain-file, or --all-defaults")
        raise SystemExit(1)

    miner = WikidataMiner(quota_limit=quota_limit, limit_per_domain=limit)
    click.echo(f"Mining {len(domains)} domain(s)...")
    result = miner.mine_domains(
        domains=domains,
        output_dir=output,
        output_format=output_format,
        incremental_since=incremental,
    )
    click.echo(
        f"Mined {len(result.entities)} unique entities ({result.total_queries} queries)"
    )
    if result.quota_exhausted:
        click.echo("Quota exhausted — resume with same command to continue")
    for domain_name, count in result.domain_counts.items():
        click.echo(f"  {domain_name}: {count}")


# ── REBEL conversion ───────────────────────────────────────────────────────


@admin_group.command("convert-rebel")
@click.option(
    "--output", "-o", required=True, type=click.Path(), help="Output corpus JSONL path"
)
@click.option("--limit", default=0, help="Max samples to convert (0=all)")
@click.option("--domain", default=None, help="Domain keyword filter (tech, science)")
@click.option(
    "--split", default="train", show_default=True, help="HuggingFace dataset split"
)
def convert_rebel_cmd(output: str, limit: int, domain: str | None, split: str) -> None:
    """Convert REBEL dataset (HuggingFace) to corpus JSONL."""
    from smartmemory.corpus.rebel import REBELConverter

    converter = REBELConverter(domain=domain, limit=limit, split=split)
    click.echo(f"Converting REBEL ({domain or 'all'}, limit={limit or 'unlimited'})...")
    count = converter.convert_to_file(output)
    click.echo(f"Wrote {count} records to {output}")


# ── Seed packs ─────────────────────────────────────────────────────────────


@admin_group.command("list-packs")
def list_packs_cmd() -> None:
    """List available seed packs from the registry."""
    from smartmemory.corpus.registry import PackRegistry

    registry = PackRegistry()
    click.echo("Fetching pack registry...")
    packs = registry.fetch()
    if not packs:
        click.echo("No seed packs available yet. Pack registry coming soon.")
        click.echo("Track progress: https://docs.smartmemory.ai/smartmemory/seed-packs")
        return
    click.echo(f"\n{'Name':<25} {'Version':<10} {'Size':<8} {'Domain':<15} Description")
    click.echo("-" * 80)
    for p in packs:
        click.echo(
            f"{p.name:<25} {p.version:<10} {p.size_mb:<8.1f} {p.domain:<15} {p.description}"
        )


@admin_group.command("install-pack")
@click.argument("name")
@click.option(
    "--source",
    default=None,
    type=click.Path(exists=True),
    help="Local pack directory (skip registry download)",
)
@click.option(
    "--mode",
    "install_mode",
    default="direct",
    type=click.Choice(["full", "direct"]),
    show_default=True,
)
@click.option("--skip-patterns", is_flag=True, help="Skip EntityRuler pattern import")
@click.option("--skip-entities", is_flag=True, help="Skip grounding entity import")
def install_pack_cmd(
    name: str,
    source: str | None,
    install_mode: str,
    skip_patterns: bool,
    skip_entities: bool,
) -> None:
    """Install a seed pack into SmartMemory."""
    from pathlib import Path
    from smartmemory.corpus.pack import InstalledPacks, SeedPack
    from smartmemory_app.storage import get_memory, _resolve_data_dir

    if source:
        pack_dir = source
    else:
        # Check bundled packs first
        bundled = (
            Path(__file__).parent.parent
            / "smart-memory-core"
            / "smartmemory"
            / "data"
            / "seed_packs"
            / name
        )
        if bundled.exists():
            pack_dir = str(bundled)
        else:
            from smartmemory.corpus.registry import PackRegistry

            registry = PackRegistry()
            registry.fetch()
            pack_info = registry.get(name)
            if not pack_info:
                click.echo(f"Pack '{name}' not found in registry.")
                raise SystemExit(1)
            pack_dir = registry.download(pack_info, str(_resolve_data_dir() / "packs"))

    pack = SeedPack(pack_dir)
    errors = pack.validate()
    if errors:
        click.echo(f"Invalid pack: {'; '.join(errors)}")
        raise SystemExit(1)

    sm = get_memory()
    installed = InstalledPacks(_resolve_data_dir())
    click.echo(f"Installing pack: {pack.manifest.name} v{pack.manifest.version}...")
    counts = pack.install(
        smart_memory=sm,
        installed_packs=installed,
        mode=install_mode,
        skip_patterns=skip_patterns,
        skip_entities=skip_entities,
    )
    click.echo(
        f"Installed: {counts['corpus']} memories, {counts['patterns']} patterns, {counts['entities']} entities"
    )


# ── Data management ─────────────────────────────────────────────────────────


@cli.command("clear")
@click.confirmation_option(prompt="This will delete all local memories. Are you sure?")
def clear_cmd() -> None:
    """Delete all local memories and reset the vector index."""
    result = _daemon_request("POST", "/memory/clear")
    if result is not None:
        click.echo(f"Cleared {result.get('cleared', '?')} files via daemon.")
        return

    # Daemon not running — clear files directly
    from smartmemory_app.storage import _resolve_data_dir, _shutdown

    _shutdown()

    data_path = _resolve_data_dir()
    if not data_path.exists():
        click.echo("No data directory found. Nothing to clear.")
        return

    removed = 0
    for pattern in [
        "*.db",
        "*.db-shm",
        "*.db-wal",
        "*.db-journal",
        "*.usearch",
        "*.json",
        "*.jsonl",
        "*.log",
        ".write.lock",
    ]:
        for f in data_path.glob(pattern):
            try:
                f.unlink()
                removed += 1
            except OSError as e:
                click.echo(f"Warning: could not remove {f.name}: {e}")

    click.echo(f"Cleared {removed} files from {data_path}")

    from smartmemory_app.setup import _seed_data_dir

    _seed_data_dir()
    click.echo("Re-seeded entity patterns.")


# ── Admin: reindex ─────────────────────────────────────────────────────────


@admin_group.command("reindex")
def reindex_cmd() -> None:
    """Re-embed all memories with the current embedding model."""
    from smartmemory_app.config import load_config

    cfg = load_config()
    if cfg.mode == "remote":
        raise click.ClickException("Reindex is only available in local mode.")
    result = _daemon_request("POST", "/memory/reindex")
    if result is not None:
        click.echo(
            f"Reindexed {result.get('reindexed', '?')} memories "
            f"({result.get('dims', '?')}d, {result.get('provider', '?')}, "
            f"{result.get('elapsed_s', '?')}s)"
        )
    else:
        raise click.ClickException(
            "Daemon is not running. Start it first: smartmemory start"
        )


@admin_group.command("reextract")
def reextract_cmd() -> None:
    """Re-run entity extraction on all memories to populate the knowledge graph.

    Use after upgrading to rebuild entity nodes and edges for memories that
    were stored before entity extraction was available on lite mode.
    """
    from smartmemory_app.config import load_config

    cfg = load_config()
    if cfg.mode == "remote":
        raise click.ClickException("Reextract is only available in local mode.")
    click.echo("Re-extracting entities from all memories...")
    result = _daemon_request("POST", "/memory/reextract", timeout=300)
    if result is not None:
        click.echo(
            f"Done: {result.get('extracted', 0)} memories processed, "
            f"{result.get('entities_created', 0)} new entity nodes, "
            f"{result.get('skipped', 0)} skipped, "
            f"{result.get('elapsed_s', '?')}s"
        )
    else:
        raise click.ClickException(
            "Daemon is not running. Start it first: smartmemory start"
        )


# ── Hidden/internal ─────────────────────────────────────────────────────────


@cli.command("server", hidden=True)
def server_cmd() -> None:
    """Start the SmartMemory MCP server (called by MCP clients, not users)."""
    from smartmemory_app.server import main

    main()


@cli.command("events-server", hidden=True)
@click.option("--port", default=9015, show_default=True, help="WebSocket port")
def events_server_cmd(port: int) -> None:
    """Run the lite WebSocket events server standalone (debugging only)."""
    from smartmemory_app.events_server import main

    main(port=port)


if __name__ == "__main__":
    cli()
