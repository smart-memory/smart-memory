"""Automatic memory lifecycle engine for DIST-AGENT-HOOKS-1.

Manages the 6-phase lifecycle: Orient, Recall, Observe, Distill, Learn, Persist.
Each CLI invocation creates a short-lived instance that loads/saves session-scoped
state from $SMARTMEMORY_DATA_DIR/sessions/<session_id>.json.

All storage operations delegate to smartmemory_app.storage.* functions.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from smartmemory_app.lifecycle_config import LifecycleConfig, RecallStrategy

log = logging.getLogger(__name__)

# Trivial prompts that never trigger recall
_SKIP_TOKENS = {"yes", "no", "ok", "sure", "y", "n", "k", "yep", "nope", "yeah"}
_MAX_SKIP_TOKENS = 3

# Approximate token counting: 1 token ≈ 4 chars (conservative)
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


class MemoryLifecycle:
    """Automatic memory lifecycle for agent sessions.

    Not a singleton — each CLI invocation creates an instance, loads session
    state, executes one phase, saves state.
    """

    def __init__(self, session_id: str, config: LifecycleConfig | None = None):
        self.session_id = session_id
        self._config = config or LifecycleConfig()

        # Session state (persisted to sessions/<session_id>.json)
        self._current_user_turn: str | None = None
        self._last_assistant_message: str | None = None
        self._last_injection_embedding: list[float] | None = None
        self._last_recalled_prompt: str | None = None
        self._turn_count: int = 0
        self._observation_count: int = 0
        self._config_overrides: dict = {}

        self._load_state()

        # Apply session overrides if any
        if self._config_overrides:
            self._config = self._config.apply_overrides(self._config_overrides)

    # ── Phase methods ──────────────────────────────────────────────────

    def orient(self, cwd: str | None = None) -> str:
        """Phase 1: Session start — recall previous context with progressive disclosure.

        Returns formatted context block within orient_budget tokens.
        Clears stale session state for fresh start.
        """
        if not self._config.enabled:
            return ""

        # Clear state for new session
        self._current_user_turn = None
        self._last_assistant_message = None
        self._last_injection_embedding = None
        self._last_recalled_prompt = None
        self._turn_count = 0
        self._observation_count = 0

        from smartmemory_app.storage import recall, search

        # Get recent + relevant memories
        context = recall(cwd, top_k=10)

        # Also search for patterns and decisions if cwd provided
        patterns: list[dict] = []
        if cwd:
            try:
                patterns = search(
                    f"patterns conventions decisions for {os.path.basename(cwd)}",
                    top_k=5,
                )
            except Exception:
                pass  # non-critical

        output = self._format_orient_block(context, patterns)
        self._save_state()
        return output

    def recall(self, prompt: str) -> str:
        """Phase 2: Per-prompt recall — always captures prompt, optionally injects context.

        Always stores prompt as _current_user_turn for distill pairing.
        Returns formatted context if recall gate passes, empty string otherwise.
        """
        if not self._config.enabled:
            return ""

        # Always capture prompt for distill pairing (unconditional)
        self._current_user_turn = prompt
        self._turn_count += 1

        # Check if recall should fire
        if not self._should_recall(prompt):
            self._save_state()
            return ""

        from smartmemory_app.storage import search

        try:
            results = search(prompt, top_k=5)
        except Exception as e:
            log.warning("Recall search failed: %s", e)
            self._save_state()
            return ""

        if not results:
            self._save_state()
            return ""

        output = self._format_recall_block(results)

        # Cache prompt for dedup and topic comparison
        self._last_recalled_prompt = prompt
        self._cache_embedding(prompt)

        self._save_state()
        return output

    def observe(self, tool_name: str, tool_input: dict, tool_result: str) -> None:
        """Phase 3: Capture tool call as observation. Async via storage.ingest()."""
        if not self._config.enabled or not self._config.observe_tool_calls:
            return

        input_summary = json.dumps(tool_input)[:200] if tool_input else ""
        result_summary = (tool_result or "")[:300]
        text = f"Tool `{tool_name}` called. Input: {input_summary}. Result: {result_summary}"

        from smartmemory_app.storage import ingest

        try:
            ingest(text, memory_type="episodic", properties={"origin": "hook:observe"})
            self._observation_count += 1
            self._save_state()
        except Exception as e:
            log.warning("Observe ingest failed: %s", e)

    def distill(self, response: str) -> None:
        """Phase 4: Pair assistant response with stored prompt, save turn pair.

        Called by Stop hook with last_assistant_message.
        """
        if not self._config.enabled or not self._config.distill_turns:
            return

        self._last_assistant_message = response

        user_turn = self._current_user_turn or ""
        if not user_turn and not response:
            self._save_state()
            return

        # Format as distill pair
        pair = f"User: {user_turn[:500]}\nAssistant: {response[:1000]}"

        from smartmemory_app.storage import ingest

        try:
            ingest(pair, memory_type="working", properties={"origin": "lifecycle:distill"})
        except Exception as e:
            log.warning("Distill ingest failed: %s", e)

        # Clear current turn (consumed)
        self._current_user_turn = None
        self._save_state()

    def learn(self, tool_name: str, error: str) -> None:
        """Phase 5: Capture tool failure as episodic memory."""
        if not self._config.enabled or not self._config.learn_from_errors:
            return

        text = f"Error in `{tool_name}`: {error[:500]}"

        from smartmemory_app.storage import ingest

        try:
            ingest(text, memory_type="episodic", properties={"origin": "hook:learn"})
        except Exception as e:
            log.warning("Learn ingest failed: %s", e)

    def persist(self) -> None:
        """Phase 6: Session end — save session summary, clean up state file."""
        if not self._config.enabled:
            self._delete_state()
            return

        summary = self._last_assistant_message
        if not summary:
            self._delete_state()
            return

        text = f"Session summary (turns={self._turn_count}, observations={self._observation_count}): {summary[:1000]}"

        from smartmemory_app.storage import ingest

        try:
            ingest(text, memory_type="episodic", properties={"origin": "hook:persist"})
        except Exception as e:
            log.warning("Persist ingest failed: %s", e)

        self._delete_state()

    # ── Recall gate ────────────────────────────────────────────────────

    def _should_recall(self, prompt: str) -> bool:
        """Decide whether to fire recall for this prompt."""
        strategy = self._config.recall_strategy

        # Session-only: never recall per-prompt
        if strategy == RecallStrategy.SESSION_ONLY:
            return False

        # Trivial-skip rules (apply to topic_change and every_prompt)
        stripped = prompt.strip()
        if not stripped:
            return False
        if len(stripped.split()) <= _MAX_SKIP_TOKENS and stripped.lower() in _SKIP_TOKENS:
            return False
        if stripped.startswith("/"):
            return False  # slash commands
        if stripped == self._last_recalled_prompt:
            return False  # dedup

        # Every-prompt: always recall after trivial gate
        if strategy == RecallStrategy.EVERY_PROMPT:
            return True

        # Topic-change: compare embedding similarity
        if strategy == RecallStrategy.TOPIC_CHANGE:
            if self._last_injection_embedding is None:
                return True  # first recall in session
            return self._topic_changed(prompt)

        return False

    def _topic_changed(self, prompt: str) -> bool:
        """Check if prompt topic diverges from last injection."""
        if self._last_injection_embedding is None:
            return True
        try:
            from smartmemory_app.storage import get_memory
            mem = get_memory()
            embedding = mem.embed(prompt)
            if embedding is None:
                return True
            similarity = self._cosine_similarity(embedding, self._last_injection_embedding)
            return similarity < self._config.topic_threshold
        except Exception:
            return True  # fail open — recall when unsure

    def _cache_embedding(self, prompt: str) -> None:
        """Cache the prompt embedding for topic comparison."""
        try:
            from smartmemory_app.storage import get_memory
            mem = get_memory()
            self._last_injection_embedding = mem.embed(prompt)
        except Exception:
            pass

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── Formatting ─────────────────────────────────────────────────────

    def _format_orient_block(self, context: str, patterns: list[dict]) -> str:
        """Build layered Orient context block within budget."""
        budget = self._config.orient_budget
        lines: list[str] = []

        # Layer 1: Recall context (highest priority)
        if context:
            context_tokens = _estimate_tokens(context)
            if context_tokens <= budget:
                lines.append(context)
                budget -= context_tokens

        # Layer 2: Patterns/decisions (if budget remains)
        if patterns and budget > 100:
            pattern_lines = []
            for p in patterns:
                content = p.get("content", "") if isinstance(p, dict) else str(p)
                mtype = p.get("memory_type", "?") if isinstance(p, dict) else "?"
                line = f"- [{mtype}] {content[:150]}"
                line_tokens = _estimate_tokens(line)
                if budget - line_tokens < 0:
                    break
                pattern_lines.append(line)
                budget -= line_tokens
            if pattern_lines:
                lines.append("\n## Patterns")
                lines.extend(pattern_lines)

        return "\n".join(lines) if lines else ""

    def _format_recall_block(self, results: list[dict]) -> str:
        """Build Recall context block within budget."""
        budget = self._config.recall_budget
        lines = ["[SmartMemory Recall]"]
        used = _estimate_tokens(lines[0])

        for r in results:
            content = r.get("content", "") if isinstance(r, dict) else str(r)
            mtype = r.get("memory_type", "?") if isinstance(r, dict) else "?"
            line = f"- [{mtype}] {content[:200]}"
            line_tokens = _estimate_tokens(line)
            if used + line_tokens > budget:
                break
            lines.append(line)
            used += line_tokens

        return "\n".join(lines) if len(lines) > 1 else ""

    # ── Session state persistence ──────────────────────────────────────

    def _state_dir(self) -> Path:
        data_dir = os.environ.get("SMARTMEMORY_DATA_DIR", str(Path.home() / ".smartmemory"))
        d = Path(data_dir) / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _state_path(self) -> Path:
        # Sanitize session_id to prevent path traversal
        safe_id = "".join(c for c in self.session_id if c.isalnum() or c in "-_")
        if not safe_id:
            safe_id = "unknown"
        return self._state_dir() / f"{safe_id}.json"

    def _load_state(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self._current_user_turn = data.get("current_user_turn")
            self._last_assistant_message = data.get("last_assistant_message")
            self._last_injection_embedding = data.get("last_injection_embedding")
            self._last_recalled_prompt = data.get("last_recalled_prompt")
            self._turn_count = data.get("turn_count", 0)
            self._observation_count = data.get("observation_count", 0)
            self._config_overrides = data.get("config_overrides", {})
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load session state: %s", e)

    def _save_state(self) -> None:
        data = {
            "session_id": self.session_id,
            "current_user_turn": self._current_user_turn,
            "last_assistant_message": self._last_assistant_message,
            "last_injection_embedding": self._last_injection_embedding,
            "last_recalled_prompt": self._last_recalled_prompt,
            "turn_count": self._turn_count,
            "observation_count": self._observation_count,
            "config_overrides": self._config_overrides,
            "updated_at": time.time(),
        }
        path = self._state_path()
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data))
            tmp.rename(path)
        except OSError as e:
            log.warning("Failed to save session state: %s", e)

    def _delete_state(self) -> None:
        try:
            self._state_path().unlink(missing_ok=True)
        except OSError:
            pass

    @classmethod
    def cleanup_stale_sessions(cls, max_age_hours: int = 24) -> int:
        """Delete session state files older than max_age_hours. Returns count deleted."""
        data_dir = os.environ.get("SMARTMEMORY_DATA_DIR", str(Path.home() / ".smartmemory"))
        sessions_dir = Path(data_dir) / "sessions"
        if not sessions_dir.exists():
            return 0
        cutoff = time.time() - (max_age_hours * 3600)
        deleted = 0
        for f in sessions_dir.glob("*.json"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    deleted += 1
            except OSError:
                pass
        return deleted
