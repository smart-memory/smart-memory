"""LAUNCH-METRICS-1 — CLI-side launch event emission.

Lightweight wrapper that POSTs to the daemon HTTP API at /launch/event. The
daemon proxies to the service when configured for remote mode; in local mode
it can write directly to a local Mongo. Failures are best-effort (logged at
WARNING) — observability must never break a CLI command.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

log = logging.getLogger(__name__)


VALID_EVENT_TYPES = frozenset({
    "install.start",
    "setup.complete",
    "mcp.install",
    "index.start",
    "index.complete",
    "recall.invoke",
    "recall.first",
    "recall.accepted",
    "decision.create",
})


def _daemon_url() -> Optional[str]:
    try:
        from smartmemory_app.config import load_config

        return f"http://127.0.0.1:{load_config().daemon_port}"
    except Exception:
        return None


def emit(event_type: str, props: Optional[Mapping[str, Any]] = None) -> bool:
    """Best-effort emit. Returns True on apparent success, False otherwise.

    Honors ``SMARTMEMORY_DISABLE_LAUNCH_METRICS=1`` for opt-out.
    """
    if os.environ.get("SMARTMEMORY_DISABLE_LAUNCH_METRICS") == "1":
        return False
    if event_type not in VALID_EVENT_TYPES:
        log.warning("launch_metrics: rejected unknown event_type=%s", event_type)
        return False

    base = _daemon_url()
    if not base:
        return False

    try:
        import httpx  # type: ignore
    except Exception:
        return False

    payload = {"event_type": event_type, "props": dict(props or {})}
    try:
        r = httpx.post(f"{base}/launch/event", json=payload, timeout=2.0)
        return 200 <= r.status_code < 300
    except Exception as e:
        log.warning("launch_metrics: daemon emit failed event=%s err=%s", event_type, e)
        return False
