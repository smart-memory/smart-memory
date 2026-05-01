"""DIST-LITE-5: Remote backend — httpx client wrapping the SmartMemory hosted API.

Extracted and adapted from smart-memory-mcp/smartmemory_mcp/server.py.
The module-level _session dict from that file becomes instance attrs here,
enabling multiple RemoteMemory instances (future: named profiles).

Interface matches local storage.py for MCP tool methods (ingest/search/get/recall)
plus graph methods matching local_api.py response shapes for the viewer.

Critical invariants:
  - _request() never raises — returns {"error": "..."} on all failure modes
  - ingest() uses POST /memory/ingest and reads result["item_id"]
    (NOT /memory/add which returns result["id"] — different field names)
  - recall() is fully client-side — no /memory/recall endpoint exists in the hosted API
  - get_node() maps to GET /memory/{id} — no dedicated entity-node endpoint
"""
from __future__ import annotations

import warnings
from typing import Optional

import httpx

from smartmemory_app.config import get_api_key, set_api_key


class RemoteMemory:
    """Thin httpx client with the same interface as local SmartMemory for MCP tools,
    plus graph methods matching local_api.py response shapes for the viewer."""

    def __init__(
        self,
        api_url: str = "https://api.smartmemory.ai",
        team_id: str = "",
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._team_id = team_id
        self._access_token = get_api_key()
        self._bootstrapped = False
        if self._access_token:
            self._bootstrap()

    # ── Session bootstrap ──────────────────────────────────────────────────

    def _bootstrap(self) -> None:
        """Discover team_id from /auth/me on first use. Best-effort; failures are silent."""
        if self._bootstrapped or not self._access_token:
            return
        self._bootstrapped = True
        try:
            r = httpx.get(
                f"{self._api_url}/auth/me",
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            if r.status_code == 200:
                user = r.json()
                if not self._team_id:
                    self._team_id = user.get("default_team_id") or ""
        except Exception:
            pass  # bootstrap is best-effort; real errors surface on tool calls

    def _headers(self, workspace_id: Optional[str] = None) -> dict:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "X-Workspace-Id": workspace_id or self._team_id,
        }

    def _request(
        self,
        method: str,
        path: str,
        workspace_id: Optional[str] = None,
        timeout: int = 30,
        **kwargs,
    ):
        """Execute an API request. Never raises — returns {"error": "..."} on any failure."""
        try:
            r = httpx.request(
                method,
                f"{self._api_url}{path}",
                headers=self._headers(workspace_id),
                timeout=timeout,
                **kwargs,
            )
            r.raise_for_status()
            return r.json() if r.status_code != 204 else None
        except httpx.ConnectError:
            return {"error": f"SmartMemory API unreachable at {self._api_url}. "
                             "Check SMARTMEMORY_API_URL."}
        except httpx.HTTPStatusError as e:
            return {"error": f"API error {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"error": f"Request failed: {e}"}

    # ── Auth ────────────────────────────────────────────────────────────────

    def login(self, api_key: str, team_id: str = "") -> str:
        """Set API key, persist to keychain, discover team from /auth/me."""
        self._access_token = api_key
        self._bootstrapped = False
        self._team_id = team_id
        try:
            r = httpx.get(
                f"{self._api_url}/auth/me",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=15,
            )
            r.raise_for_status()
            user = r.json()
            self._team_id = team_id or user.get("default_team_id") or self._team_id
            self._bootstrapped = True
        except httpx.HTTPStatusError as e:
            return f"API key validation failed ({e.response.status_code}): {e.response.text}"
        except Exception as e:
            return f"Login failed: {e}"
        set_api_key(api_key)  # persist to OS keychain (warns if unavailable, never raises)
        # Persist team_id to config so next startup uses the correct workspace
        from smartmemory_app.config import load_config, save_config
        cfg = load_config()
        cfg.team_id = self._team_id
        save_config(cfg)
        return f"Logged in. Team: {self._team_id}"

    def whoami(self) -> str:
        result = self._request("GET", "/auth/me")
        if err := (result or {}).get("error"):
            return err
        return f"User: {result.get('email', '?')}, Team: {self._team_id}, API: {self._api_url}"

    def switch_team(self, team_id: str) -> str:
        self._team_id = team_id
        return f"Switched to team: {team_id}"

    # ── MCP tool interface (same signatures as local storage.py) ────────────

    def ingest(self, content: str, memory_type: str = "semantic") -> str:
        """POST /memory/ingest (full pipeline). Returns item_id string.

        Uses /memory/ingest (not /memory/add) — ingest runs entity extraction,
        enrichment, linking, grounding. Returns {"item_id": ...} (not {"id": ...}).
        """
        body = {"content": content, "context": {"memory_type": memory_type}}
        result = self._request("POST", "/memory/ingest", timeout=120, json=body)
        if err := (result or {}).get("error"):
            return f"Error: {err}"
        return result.get("item_id", "unknown")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """POST /memory/search. Returns list[dict] (not MemoryItem objects)."""
        body = {"query": query, "top_k": top_k, "enable_hybrid": True}
        result = self._request("POST", "/memory/search", json=body)
        # _request() returns a list on success, dict on error, None on 204
        if isinstance(result, dict) and (err := result.get("error")):
            return [{"error": err}]
        return result if isinstance(result, list) else []

    def get(self, item_id: str) -> dict | None:
        """GET /memory/{item_id}. Returns flat dict or None if not found."""
        result = self._request("GET", f"/memory/{item_id}")
        if result is None or (result or {}).get("error"):
            return None
        return result

    def recall(
        self,
        cwd: str | None = None,
        top_k: int = 10,
        *,
        query: str | None = None,
        include_snapshot: bool = True,
        workspace_id: str | None = None,
        strict: bool | None = None,
    ) -> str:
        """HOOK-RECALL-RELEVANCE-1: workspace-scoped, ranked, deduped recall (remote).

        Mirrors local storage.recall(). Remote has no /memory/recall endpoint and
        no sort_by="recency" support, so candidate selection runs through the
        hosted /memory/search.
        """
        import os
        from smartmemory.origin_policy import get_default_tiers, get_tier
        from smartmemory_app.recall_format import (
            _trace, derive_workspace_id, format_recall_lines, time_ms,
        )

        t0 = time_ms()
        workspace_id = workspace_id or derive_workspace_id(cwd)
        if strict is None:
            strict = os.environ.get(
                "SMARTMEMORY_RECALL_STRICT", "").lower() in ("1", "true", "yes")
        recall_tiers = get_default_tiers("recall")

        # 1. Optional snapshot frame (graph-mirrored)
        frame = ""
        if include_snapshot:
            try:
                snaps = self._request(
                    "POST", "/memory/search",
                    workspace_id=workspace_id,
                    json={"query": "", "memory_type": "snapshot",
                          "sort_by": "recency", "top_k": 1},
                )
                rows = snaps if isinstance(snaps, list) else (snaps or {}).get("results", [])
                if rows:
                    snap = rows[0]
                    snap_meta = snap.get("metadata") or {}
                    ws_match = (
                        workspace_id is None
                        or snap_meta.get("workspace_id") in (None, workspace_id)
                    )
                    if ws_match:
                        frame = (snap.get("content") or "").strip()
            except Exception:
                pass  # snapshot is best-effort

        # 2. Candidates
        if query:
            results = self.search(query, top_k=top_k * 2) or []
        else:
            requested = max(1, top_k)
            recent_k = max(1, (requested + 1) // 2)
            semantic_k = max(0, requested - recent_k)
            recent = self.search("", top_k=recent_k) or []
            semantic = self.search(cwd or "", top_k=semantic_k) if cwd and semantic_k else []
            results = list(recent) + list(semantic)
        results = [r for r in results if r.get("memory_type") != "snapshot"]

        # 3. Origin tier filter (dict-aware; legacy "unknown" / missing pass through)
        def _tier_ok(r: dict) -> bool:
            origin = r.get("origin") or "unknown"
            if origin == "unknown":
                return True
            return get_tier(origin) in recall_tiers
        results = [r for r in results if _tier_ok(r)]

        # 4. Workspace metadata filter (strict drops legacy untagged items).
        if workspace_id:
            scoped = []
            for r in results:
                meta = r.get("metadata") or {}
                ws = meta.get("workspace_id") if isinstance(meta, dict) else None
                if ws == workspace_id:
                    scoped.append(r)
                elif ws is None and not strict:
                    scoped.append(r)
            results = scoped

        # 5. Confidence floor + reference exclusion (preserved)
        recall_floor = float(os.environ.get("SMARTMEMORY_RECALL_FLOOR", "0.3"))
        results = [
            r for r in results
            if (r.get("confidence") if r.get("confidence") is not None else 1.0) >= recall_floor
        ]
        results = [r for r in results if not r.get("reference", False)]

        # 6. Format (dedup + empty-suppress + top_k cap inside)
        body = format_recall_lines(results, top_k=top_k)
        emitted = body.count("\n- ") if body else 0

        # 7. Compose
        if body and frame:
            out = f"{body}\n\n{frame}"
        elif body:
            out = body
        elif frame:
            out = f"## SmartMemory Context\n\n{frame}"
        else:
            out = ""

        # 8. Trace
        _trace(
            phase="user_prompt" if query else "session_start",
            workspace_id=workspace_id,
            cwd=cwd,
            query=query,
            candidate_count=len(results),
            emitted=emitted,
            snapshot_used=bool(frame),
            latency_ms=time_ms() - t0,
        )
        return out

    # ── Graph methods — called by local_api.py for viewer ──────────────────

    def get_graph_full(self) -> dict:
        """GET /memory/graph/full — same response shape as local get_graph_full()."""
        result = self._request("GET", "/memory/graph/full")
        if err := (result or {}).get("error"):
            return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0, "error": err}
        return result

    def get_edges_bulk(self, node_ids: list[str]) -> dict:
        """POST /memory/graph/edges — same response shape as local get_edges_bulk()."""
        result = self._request("POST", "/memory/graph/edges", json={"node_ids": node_ids})
        if err := (result or {}).get("error"):
            return {"edges": [], "error": err}
        return result

    def get_node(self, item_id: str) -> dict | None:
        """GET /memory/{item_id} — memory nodes only.

        No dedicated entity-node endpoint exists in the hosted API.
        Entity-only nodes (pipeline artifacts without an item_id) have no
        remote retrieval path in v1.
        """
        return self.get(item_id)

    def get_neighbors(self, item_id: str) -> dict:
        """GET /memory/{item_id}/neighbors — response must have 'neighbors' and 'edges' keys.

        createFetchAdapter.getNeighbors reads: const neighbors = res?.neighbors || []
        Returning {"nodes": ...} produces an empty expansion in the viewer.

        The service (links.py) returns {neighbors, item_id} — no 'edges' key.
        We normalize defensively so the viewer always receives both keys.
        """
        result = self._request("GET", f"/memory/{item_id}/neighbors")
        if err := (result or {}).get("error"):
            return {"neighbors": [], "edges": [], "error": err}
        result.setdefault("edges", [])  # service omits 'edges'; viewer expects it
        return result
