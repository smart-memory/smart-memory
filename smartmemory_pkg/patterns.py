from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


SEED_PATTERNS_FILE = Path(__file__).parent / "data" / "seed_patterns.jsonl"
QUALITY_MIN_CONFIDENCE = 0.8
QUALITY_MIN_NAME_LEN = 3
QUALITY_MIN_FREQUENCY = 2  # pattern activates at frequency >= 2


class LitePatternManager:
    """JSONL-backed entity pattern store for zero-infra Lite mode.

    Duck-types PatternManager.get_patterns() for EntityRulerStage.
    Backed by {data_dir}/entity_patterns.jsonl instead of FalkorDB + Redis.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._path = Path(data_dir) / "entity_patterns.jsonl"
        self._patterns: dict[str, dict] = {}  # name.lower() -> entry dict
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._seed()
            return
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._patterns[entry["name"].lower()] = entry

    def _seed(self) -> None:
        """Write seed patterns from bundled data/seed_patterns.jsonl."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not SEED_PATTERNS_FILE.exists():
            # Bundled seed file absent (bad wheel build or corrupted install).
            # Create an empty patterns file so the plugin can still start.
            self._path.touch()
            log.warning(
                "Seed patterns file missing at %s — entity ruler will start empty",
                SEED_PATTERNS_FILE,
            )
            return
        with open(SEED_PATTERNS_FILE) as src, open(self._path, "w") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                dst.write(json.dumps(entry) + "\n")
                self._patterns[entry["name"].lower()] = entry

    def get_patterns(self) -> dict[str, str]:
        """Return name.lower() -> label for patterns meeting quality gate."""
        return {
            k: v["label"]
            for k, v in self._patterns.items()
            if v.get("frequency", 1) >= QUALITY_MIN_FREQUENCY
        }

    def add_pattern(
        self, name: str, label: str, confidence: float = 0.85, initial_frequency: int = 1
    ) -> None:
        """Add or increment a pattern. Persists to JSONL. Caller holds filelock.

        Args:
            initial_frequency: Starting frequency for new patterns. Use 2 for
                AST-validated code patterns to bypass the ``frequency >= 2``
                quality gate in ``get_patterns()``. Default 1 preserves existing
                threshold for LLM-discovered patterns.
        """
        if confidence <= QUALITY_MIN_CONFIDENCE:
            raise ValueError(f"confidence {confidence} <= {QUALITY_MIN_CONFIDENCE}")
        if len(name) <= QUALITY_MIN_NAME_LEN:
            raise ValueError(f"name '{name}' length <= {QUALITY_MIN_NAME_LEN}")
        key = name.lower()
        if key in self._patterns:
            self._patterns[key]["frequency"] += 1
            self._patterns[key]["confidence"] = max(
                confidence, self._patterns[key]["confidence"]
            )
        else:
            self._patterns[key] = {
                "name": name,
                "label": label,
                "confidence": confidence,
                "frequency": initial_frequency,
            }
        self._rewrite()

    def _rewrite(self) -> None:
        tmp = self._path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for entry in self._patterns.values():
                f.write(json.dumps(entry) + "\n")
        tmp.replace(self._path)
