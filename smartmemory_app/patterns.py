from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


SEED_PATTERNS_FILE = Path(__file__).parent / "data" / "seed_patterns.jsonl"
QUALITY_MIN_CONFIDENCE = 0.8
QUALITY_MIN_NAME_LEN = 3
QUALITY_MIN_FREQUENCY = 2  # pattern activates at frequency >= 2


class JSONLPatternStore:
    """JSONL-backed PatternStore for zero-infra Lite mode.

    Implements PatternStore protocol for use with unified PatternManager.
    Uses filelock for concurrent-write safety — PatternManager persists
    outside its threading.Lock, so the store must own its own I/O lock.

    JSONL record format (source is optional on read for backward-compat):
        {"name": str, "label": str, "confidence": float, "frequency": int, "source": str}
    """

    _LOCK_SUFFIX = ".write.lock"

    def __init__(self, data_dir: str | Path) -> None:
        self._path = Path(data_dir) / "entity_patterns.jsonl"
        self._lock_path = Path(str(self._path) + self._LOCK_SUFFIX)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seed_if_absent()

    def _seed_if_absent(self) -> None:
        """Write seed patterns from bundled data/seed_patterns.jsonl if file absent."""
        if self._path.exists():
            return
        if not SEED_PATTERNS_FILE.exists():
            self._path.touch()
            log.warning(
                "Seed patterns file missing at %s — entity ruler will start empty",
                SEED_PATTERNS_FILE,
            )
            return
        with open(SEED_PATTERNS_FILE) as src, open(self._path, "w") as dst:
            for line in src:
                line = line.strip()
                if line:
                    dst.write(line + "\n")

    def load(self, workspace_id: str) -> list[tuple[str, str]]:
        """Return (name, label) for patterns with frequency >= 2.

        workspace_id is accepted but unused — JSONL files are scoped by directory.
        source field is optional on read (backward-compat with pre-CORE-EXT-2 files).
        """
        if not self._path.exists():
            return []
        results = []
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("frequency", 1) >= QUALITY_MIN_FREQUENCY:
                        results.append((entry["name"], entry["label"]))
        except Exception as exc:
            log.warning("JSONLPatternStore.load failed: %s", exc)
        return results

    def save(
        self,
        name: str,
        label: str,
        confidence: float,
        count: int,
        workspace_id: str,
        source: str = "unknown",
    ) -> None:
        """Upsert a pattern. Acquires filelock for the full read-modify-write.

        count = initial frequency for new entries (use 2 to immediately pass quality gate).
        For existing entries, frequency is incremented by 1 (count param ignored on update).
        source is stored on CREATE only — not overwritten on update (preserves original
        discovery provenance, matching FalkorDB ON MATCH SET behavior).
        """
        with FileLock(str(self._lock_path)):
            entries = self._read_all()
            key = name.lower()
            if key in entries:
                entries[key]["frequency"] += 1
                entries[key]["confidence"] = max(confidence, entries[key]["confidence"])
                # Do NOT overwrite source — preserve original discovery provenance.
            else:
                entries[key] = {
                    "name": name,
                    "label": label,
                    "confidence": confidence,
                    "frequency": count,
                    "source": source,
                }
            self._write_all(entries)

    def delete(self, name: str, label: str, workspace_id: str) -> None:
        """Remove a pattern by name and label."""
        with FileLock(str(self._lock_path)):
            entries = self._read_all()
            key = name.lower()
            if key in entries and entries[key].get("label") == label:
                del entries[key]
                self._write_all(entries)

    def _read_all(self) -> dict[str, dict]:
        """Read all JSONL records as {name.lower(): entry_dict}."""
        result: dict[str, dict] = {}
        if not self._path.exists():
            return result
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    result[entry["name"].lower()] = entry
        return result

    def _write_all(self, entries: dict[str, dict]) -> None:
        """Atomically rewrite JSONL via .tmp → replace()."""
        tmp = self._path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for entry in entries.values():
                f.write(json.dumps(entry) + "\n")
        tmp.replace(self._path)
