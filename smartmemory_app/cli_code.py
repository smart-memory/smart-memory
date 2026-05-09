"""`sm code` subgroup — code repository indexing.

Wraps :func:`smartmemory.SmartMemory.ingest_code`, which writes ``code:index``
origin-tagged nodes into the local knowledge graph and emits structured
progress to stdout.

Progress streaming
------------------
``ingest_code`` is a synchronous bulk operation; we stream progress around it
rather than inside it (the indexer logs its own debug lines but doesn't
publish per-file events). The CLI prints structured JSON-ish lines marking
phase boundaries so callers can pipe-parse:

    [code:index] phase=start repo=<repo> path=<path> langs=<...>
    [code:index] phase=done repo=<repo> files=<n> entities=<n> edges=<n> embeddings=<n> elapsed_s=<s>

Errors are printed prefixed with ``[code:index] error:``.

Origin
------
Every entity created by ``CodeIndexer.to_properties()`` already sets
``origin = "code:index"`` (Tier 1 user content per
``smartmemory/origin_policy.py``); this command does not need to set it
explicitly.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import click

log = logging.getLogger(__name__)


# Default exclusions for "real" repos. Kept tight on purpose — tree-sitter
# can choke on minified vendor bundles, and node_modules / venvs blow up the
# parser pass.
_DEFAULT_EXCLUDES = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "tmp",
}


def _detect_repo_name(path: Path) -> str:
    """Best-effort repo identifier — folder name, falling back to git remote."""
    name = path.resolve().name
    if name and name not in {"", ".", "/"}:
        return name
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            return url.rsplit("/", 1)[-1].removesuffix(".git") or "repo"
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass
    return "repo"


@click.group("code")
def code_group() -> None:
    """Code repository indexing commands."""


@code_group.command("index")
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--repo", default=None, help="Repo identifier (defaults to folder name).")
@click.option(
    "--language",
    "languages",
    multiple=True,
    type=click.Choice(["python", "typescript"]),
    help="Languages to index. Repeatable. Defaults to python.",
)
@click.option(
    "--exclude",
    "extra_excludes",
    multiple=True,
    help="Additional directory names to exclude (repeatable).",
)
@click.option(
    "--commit-hash",
    default=None,
    help="Override git commit SHA stamped on every entity. "
    "Defaults to auto-detect via `git rev-parse HEAD`. "
    "Pass empty string to suppress auto-detection.",
)
def code_index_cmd(
    path: str,
    repo: str | None,
    languages: tuple[str, ...],
    extra_excludes: tuple[str, ...],
    commit_hash: str | None,
) -> None:
    """Index a code repository into the knowledge graph.

    Parses Python (and optionally TypeScript/JavaScript) files, writes code
    entities + relationships as graph nodes/edges, and generates vector
    embeddings. Every node is tagged with origin=code:index (Tier 1).

    \b
    Examples:
        sm code index .
        sm code index ~/repos/myapp --repo myapp --language python --language typescript
    """
    repo_path = Path(path).resolve()
    repo_id = repo or _detect_repo_name(repo_path)
    lang_list = list(languages) if languages else ["python"]
    excludes = sorted(_DEFAULT_EXCLUDES | set(extra_excludes))

    click.echo(
        f"[code:index] phase=start repo={repo_id} path={repo_path} "
        f"langs={','.join(lang_list)} excludes={len(excludes)}"
    )

    # We index against the local lite memory directly. Going through the daemon
    # HTTP API would require a new route; the indexer is heavy and runs in the
    # caller's process anyway, so direct is the simpler path.
    try:
        from smartmemory_app.storage import get_memory
    except Exception as exc:  # pragma: no cover — import failure is environmental
        raise click.ClickException(f"Could not import storage layer: {exc}")

    try:
        memory = get_memory()
    except Exception as exc:
        raise click.ClickException(
            f"SmartMemory is not configured: {exc}. Run: sm init"
        )

    if not hasattr(memory, "ingest_code"):
        raise click.ClickException(
            "Active backend does not support code indexing "
            "(remote mode is not yet supported). Switch to local mode: sm config mode local"
        )

    started = time.time()
    try:
        result = memory.ingest_code(
            directory=str(repo_path),
            repo=repo_id,
            commit_hash=commit_hash,
            exclude_dirs=excludes,
            languages=lang_list,
        )
    except Exception as exc:
        click.echo(f"[code:index] error: {exc}", err=True)
        log.exception("code index failed")
        raise SystemExit(1)
    elapsed = time.time() - started

    click.echo(
        f"[code:index] phase=done repo={repo_id} "
        f"files={result.files_parsed} skipped={result.files_skipped} "
        f"entities={result.entities_created} edges={result.edges_created} "
        f"embeddings={result.embeddings_generated} "
        f"elapsed_s={result.elapsed_seconds or round(elapsed, 2)}"
    )

    if result.errors:
        # Surface (but do not fail on) parse errors — IndexResult treats them
        # as soft warnings. Cap the dump so a busted vendor dir does not flood
        # stdout. NEVER swallow silently — the rule of no_silent_fallbacks.
        cap = 20
        click.echo(
            f"[code:index] warning: {len(result.errors)} parse error(s); showing first {min(cap, len(result.errors))}",
            err=True,
        )
        for line in result.errors[:cap]:
            click.echo(f"  - {line}", err=True)
