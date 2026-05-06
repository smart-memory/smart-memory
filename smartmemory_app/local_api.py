"""DIST-LITE-4/5 + DIST-DAEMON-1: Memory API — local SQLite or remote hosted API.

Mounted at /memory by viewer_server.py — routes here are relative to that mount.
GET  /graph/full         → /memory/graph/full
POST /graph/edges        → /memory/graph/edges
GET  /list               → /memory/list
POST /ingest             → /memory/ingest           (DAEMON-1: full pipeline ingest)
POST /search             → /memory/search           (DAEMON-1: semantic search)
GET  /recall             → /memory/recall           (DAEMON-1: session context)
POST /clear              → /memory/clear
POST /reindex            → /memory/reindex          (re-embed all with current model)
GET  /{id}/neighbors     → /memory/{id}/neighbors  (BEFORE /{id} — declaration order matters; DIST-OBSIDIAN-LITE-1: each neighbor carries direction)
GET  /{id}               → /memory/{id}
PATCH /{id}              → /memory/{id}            (DIST-OBSIDIAN-LITE-1: CORE-CRUD-UPDATE-1 contract)
DELETE /{id}             → /memory/{id}            (DIST-OBSIDIAN-LITE-1: lifted from prior 405; cascades vector via mem.delete)
DELETE /graph/nodes/{id} → /memory/graph/nodes/{id} (405 — entity-node ops still read-only)

All endpoints acquire _rw_lock for thread safety under uvicorn's thread pool.
"""
import threading
from typing import Any, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Response
from pydantic import BaseModel

from smartmemory_app.config import UnconfiguredError
from smartmemory_app.storage import get_memory

# DIST-DAEMON-1: All endpoints serialize through this lock. Uvicorn runs sync
# endpoints in a thread pool — without locking, concurrent ingest+clear or
# ingest+read could race on the SmartMemory singleton's non-thread-safe state.
_rw_lock = threading.RLock()

api = FastAPI(title="SmartMemory Local API", docs_url=None, redoc_url=None)

# Fields that normalizeAPIResponse reads at top level (normalize.js:38,46).
# Also includes entity-detection fields (normalize.js:38): node_category, entity_type.
# Everything else from the properties blob lands in metadata.
_TOP_LEVEL_FIELDS = frozenset({
    "label", "content", "memory_type", "category",
    "confidence", "created_at",
    "node_category", "entity_type",
})


def _get_mem():
    """Return the active memory backend. Converts config errors → meaningful HTTP responses.

    FastAPI surfaces unhandled exceptions as 500. Two typed exceptions escape get_memory():
      UnconfiguredError — no config exists; HTTP 503 (run setup to fix)
      ValueError        — invalid mode value in env var; HTTP 400 (fix the env var)
    """
    try:
        return get_memory()
    except UnconfiguredError as e:
        raise HTTPException(
            status_code=503,
            detail=f"SmartMemory not configured. Run: smartmemory setup. ({e})",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"SmartMemory misconfigured: {e}",
        )


def _get_backend():
    """Return the SQLiteBackend directly — avoids going through SmartMemory pipeline.

    Routes through _get_mem() so UnconfiguredError converts to HTTP 503, not 500.
    Local mode only — callers must take the remote branch before calling this.
    """
    return _get_mem()._graph.backend


def _flatten_node(raw: dict) -> dict:
    """Reshape a serialize() node dict to the flat shape normalizeAPIResponse expects.

    serialize() (sqlite.py:329-335) returns:
        {item_id, memory_type, valid_from, valid_to, created_at, properties: {...}}

    normalizeAPIResponse (normalize.js:38,46) reads label/content/category/
    node_category/entity_type/confidence at the TOP LEVEL — not under properties.

    Note: get_node() / get_neighbors() use _row_to_node() (sqlite.py:119-127) which
    is already flat — do NOT call _flatten_node() on their output.
    """
    props = raw.get("properties", {})
    return {
        "item_id":       raw.get("item_id") or raw.get("id", ""),
        # memory_type is a column (stripped from blob at add_node:153) — read from raw
        "memory_type":   raw.get("memory_type", props.get("memory_type", "semantic")),
        "created_at":    raw.get("created_at", props.get("created_at", "")),
        # All other viewer fields live inside the properties blob
        "label":         props.get("label", ""),
        "content":       props.get("content", ""),
        "category":      props.get("category"),
        "node_category": props.get("node_category"),
        "entity_type":   props.get("entity_type"),
        "confidence":    props.get("confidence", 1.0),
        "metadata": {
            k: v for k, v in props.items() if k not in _TOP_LEVEL_FIELDS
        },
    }


@api.get("/graph/full")
def get_graph_full() -> dict:
    mem = _get_mem()
    from smartmemory_app.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        return mem.get_graph_full()
    with _rw_lock:
        backend = _get_backend()
        snapshot = backend.serialize()
    nodes = [_flatten_node(n) for n in snapshot.get("nodes", [])]
    # Filter out internal Version nodes — Cytoscape crashes if edges reference
    # nodes that aren't in the graph (HAS_VERSION edges → missing Version targets).
    node_ids = set()
    filtered_nodes = []
    for n in nodes:
        if n.get("memory_type") == "Version":
            continue
        filtered_nodes.append(n)
        node_ids.add(n["item_id"])
    # Only include edges where both endpoints exist in the filtered node set
    edges = [
        e for e in snapshot.get("edges", [])
        if e.get("source_id") in node_ids and e.get("target_id") in node_ids
    ]
    return {
        "nodes": filtered_nodes,
        "edges": edges,
        "node_count": len(filtered_nodes),
        "edge_count": len(edges),
    }


class EdgesBulkRequest(BaseModel):
    node_ids: list[str]


@api.post("/graph/edges")
def get_edges_bulk(body: EdgesBulkRequest) -> dict:
    """Matches createFetchAdapter.getEdgesBulk — POST with {node_ids} body (fetchAdapter.js:35)."""
    mem = _get_mem()
    from smartmemory_app.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        return mem.get_edges_bulk(body.node_ids)
    with _rw_lock:
        backend = _get_backend()
        seen: set[tuple] = set()
        edges: list[dict] = []
        for node_id in body.node_ids:
            for edge in backend.get_edges_for_node(node_id):
                key = (edge["source_id"], edge["target_id"], edge["edge_type"])
                if key not in seen:
                    seen.add(key)
                    edges.append(edge)
    return {"edges": edges}


# Option B delete endpoint — /graph/nodes/{node_id} via router mounted at /graph.
# Both delete endpoints must exist: fetchAdapter.js:48-49 calls two separate paths.
_graph_router = APIRouter()


@_graph_router.delete("/nodes/{node_id}")
def delete_entity_node_405(node_id: str) -> Response:
    """Local viewer is read-only. Entity node deletes return 405."""
    return Response(status_code=405, content="Local viewer is read-only.")


api.include_router(_graph_router, prefix="/graph")


@api.get("/list")
def list_memories(limit: int = 200, offset: int = 0) -> dict:
    mem = _get_mem()
    from smartmemory_app.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        full = mem.get_graph_full()
        nodes = full.get("nodes", [])
        paginated = nodes[offset: offset + limit]
        return {"items": paginated, "total": len(nodes), "limit": limit, "offset": offset}
    with _rw_lock:
        backend = _get_backend()
        snapshot = backend.serialize()
    nodes = [_flatten_node(n) for n in snapshot.get("nodes", [])]
    paginated = nodes[offset: offset + limit]
    return {"items": paginated, "total": len(nodes), "limit": limit, "offset": offset}


# IMPORTANT: /{memory_id}/neighbors MUST be declared BEFORE /{memory_id}.
# FastAPI matches in declaration order — the bare /{memory_id} would otherwise
# capture /neighbors as the memory_id value.

@api.get("/recall")
def recall_endpoint(
    cwd: str = None,
    top_k: int = 10,
    query: str = None,
    workspace_id: str = None,
    include_snapshot: bool = True,
    strict: bool = False,
) -> dict:
    """Recall recent + relevant memories for session context.

    HOOK-RECALL-RELEVANCE-1: workspace_id, query, include_snapshot, strict
    params added. Backward compatible — existing callers passing only
    cwd+top_k still work. strict=True drops legacy items with no
    workspace_id (eliminates Alice/Atlas-style cross-workspace leak).
    """
    with _rw_lock:
        from smartmemory_app.storage import recall
        context = recall(
            cwd, top_k,
            query=query,
            workspace_id=workspace_id or None,
            include_snapshot=include_snapshot,
            strict=strict,
        )
    return {"context": context}


@api.get("/{memory_id}/neighbors")
def get_neighbors(memory_id: str) -> dict:
    """Neighbors with direction-tagged link types.

    DIST-OBSIDIAN-LITE-1: each neighbor entry now includes a `direction` field
    ("outgoing" | "incoming") so clients can disambiguate asymmetric edges
    (e.g. SUPERSEDES, written one-way newer→older). Mirrors the service
    contract added in DIST-OBSIDIAN-1 (links.py:148-158).

    Response shape:
      {
        "neighbors": [{"item_id": str, "link_type": str, "direction": "outgoing"|"incoming"}, ...],
        "edges": [...]   # unchanged; raw edge rows
      }

    Adapter contract preserved: createFetchAdapter.getNeighbors reads
    res?.neighbors || [] — still a list, just with richer entries.
    """
    mem = _get_mem()
    from smartmemory_app.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        return mem.get_neighbors(memory_id)
    with _rw_lock:
        backend = _get_backend()
        edges = backend.get_edges_for_node(memory_id)

    # Walk edges to assign direction. The daemon's adjacency-list
    # get_edges_for_node returns both legs in one query, so this is a single
    # pass — simpler than the service's two-call get_neighbors approach.
    seen: set = set()
    formatted = []
    for e in edges:
        src = e.get("source_id")
        tgt = e.get("target_id")
        link_type = e.get("edge_type") or e.get("link_type")
        if not link_type:
            continue  # malformed edge — skip rather than emit link_type=None
        if src == memory_id and tgt and tgt != memory_id:
            direction, other = "outgoing", tgt
        elif tgt == memory_id and src and src != memory_id:
            direction, other = "incoming", src
        else:
            continue  # self-loop or malformed
        key = (other, str(link_type), direction)
        if key in seen:
            continue
        seen.add(key)
        formatted.append({"item_id": other, "link_type": link_type, "direction": direction})
    return {"neighbors": formatted, "edges": edges}


@api.get("/{memory_id}")
def get_memory_item(memory_id: str) -> dict[str, Any]:
    """get_node() uses _row_to_node() — output is already flat, no transformation needed."""
    mem = _get_mem()
    from smartmemory_app.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        node = mem.get_node(memory_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Memory not found")
        return node
    with _rw_lock:
        backend = _get_backend()
        node = backend.get_node(memory_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return node


@api.post("/clear")
def clear_all() -> dict:
    """Clear all memories, reset vector index, re-seed patterns.

    Resets the in-process singleton so subsequent API calls return fresh data.
    Publishes graph_cleared event so connected viewers refresh.
    """
    with _rw_lock:
        from smartmemory_app.storage import _resolve_data_dir, _shutdown
        _shutdown()

        data_path = _resolve_data_dir()
        removed = 0
        if data_path.exists():
            for pattern in [
                "*.db", "*.db-shm", "*.db-wal", "*.db-journal",
                "*.usearch", "*.json", "*.jsonl", "*.log", ".write.lock",
            ]:
                for f in data_path.glob(pattern):
                    try:
                        f.unlink()
                        removed += 1
                    except OSError:
                        pass

        # Re-seed patterns
        from smartmemory_app.setup import _seed_data_dir
        _seed_data_dir()

        # Flush pending enrichment jobs INSIDE lock — prevents a concurrent
        # ingest from enqueuing a job between clear and flush
        from smartmemory_app.async_enrichment import reset_queue
        reset_queue()

    # Publish graph_cleared event (outside lock — emit is thread-safe)
    try:
        from smartmemory_app.event_sink import get_event_sink
        sink = get_event_sink()
        sink.emit("span", {
            "component": "graph",
            "operation": "clear_all",
            "name": "graph.clear_all",
            "nuclear": True,
        })
    except Exception:
        pass

    return {"cleared": removed}


@api.post("/reindex")
def reindex() -> dict:
    """Re-embed all memories with the current embedding model.

    Use after changing embedding provider/model to rebuild the vector index
    without losing any data. Blocks writes during re-indexing.
    """
    import json
    import os
    import sqlite3
    import time

    with _rw_lock:
        from smartmemory.plugins.embedding import EmbeddingService
        from smartmemory_app.storage import _resolve_data_dir

        data_dir = str(_resolve_data_dir())
        svc = EmbeddingService()

        # Detect dimensions from a test embedding
        test_vec = svc.embed("dimension probe")
        dims = len(test_vec)

        # Read all memory nodes from SQLite
        db_path = os.path.join(data_dir, "memory.db")
        db = sqlite3.connect(db_path)
        rows = db.execute(
            "SELECT item_id, properties FROM nodes "
            "WHERE memory_type IS NOT NULL AND memory_type != 'Version'"
        ).fetchall()
        db.close()

        if not rows:
            return {"reindexed": 0, "dims": dims, "provider": svc.provider}

        # Delete old vector index — will be recreated at new dimension
        from smartmemory.stores.vector.backends.usearch import UsearchVectorBackend

        backend = UsearchVectorBackend(persist_directory=data_dir, collection_name="memory")

        t0 = time.time()
        embedded = 0
        skipped = 0
        for item_id, props_json in rows:
            props = json.loads(props_json) if props_json else {}
            content = props.get("content", props.get("label", ""))
            if not content:
                skipped += 1
                continue
            try:
                vec = svc.embed(content[:512])
                backend.upsert(item_id=item_id, embedding=vec.tolist(), metadata={"content": content[:200]})
                embedded += 1
            except Exception:
                skipped += 1

        backend._save()
        elapsed = time.time() - t0

    return {
        "reindexed": embedded,
        "skipped": skipped,
        "total": len(rows),
        "dims": dims,
        "provider": svc.provider,
        "elapsed_s": round(elapsed, 1),
    }


@api.post("/reextract")
def reextract_entities() -> dict:
    """Re-run entity extraction on all stored memories and create entity nodes.

    Use after upgrading from a version that didn't create entity nodes on SQLite.
    Reads each memory, runs EntityRuler (spaCy + seed patterns), creates entity
    nodes and CONTAINS_ENTITY/MENTIONED_IN edges via add_dual_node.
    """
    import json
    import os
    import sqlite3
    import time

    with _rw_lock:
        from smartmemory_app.storage import _resolve_data_dir, _get_memory

        data_dir = str(_resolve_data_dir())
        mem = _get_memory()

        # Read all user memory nodes (skip entity/relation/Version)
        db_path = os.path.join(data_dir, "memory.db")
        db = sqlite3.connect(db_path)
        user_types = ("semantic", "episodic", "procedural", "pending", "zettel",
                      "reasoning", "opinion", "observation", "decision")
        placeholders = ",".join("?" * len(user_types))
        rows = db.execute(
            f"SELECT item_id, properties, memory_type FROM nodes "
            f"WHERE memory_type IN ({placeholders})",
            user_types,
        ).fetchall()
        db.close()

        if not rows:
            return {"extracted": 0, "entities_created": 0, "total": 0, "elapsed_s": 0}

        # Get EntityRuler stage from the pipeline
        from smartmemory.pipeline.stages.entity_ruler import EntityRulerStage, _get_nlp
        from smartmemory.pipeline.state import PipelineState

        nlp = _get_nlp()
        pattern_manager = getattr(mem, "_entity_ruler_patterns", None)
        ruler = EntityRulerStage(nlp=nlp, pattern_manager=pattern_manager)

        # Build a minimal pipeline config for entity_ruler
        pipeline_config = mem._build_pipeline_config()

        t0 = time.time()
        extracted = 0
        entities_created = 0
        skipped = 0
        backend = mem._graph.backend

        for item_id, props_json, memory_type in rows:
            props = json.loads(props_json) if props_json else {}
            content = props.get("content", "")
            if not content or len(content.strip()) < 3:
                skipped += 1
                continue

            # Check if this memory already has entity edges
            existing_edges = backend.get_edges_for_node(item_id)
            has_entities = any(
                e.get("edge_type") in ("CONTAINS_ENTITY", "MENTIONED_IN")
                for e in existing_edges
            )
            if has_entities:
                skipped += 1
                continue

            # Run EntityRuler on the content
            try:
                state = PipelineState(text=content, memory_type=memory_type)
                state = ruler.execute(state, pipeline_config)
                entities = state.ruler_entities or []

                if not entities:
                    skipped += 1
                    continue

                # Build entity_nodes for add_dual_node
                entity_nodes = []
                for ent in entities:
                    name = ent.get("name", "")
                    etype = ent.get("entity_type", "concept")
                    if not name:
                        continue
                    entity_nodes.append({
                        "entity_type": etype,
                        "properties": {
                            "name": name,
                            "confidence": ent.get("confidence", 0.85),
                            "source": "reextract",
                        },
                    })

                if entity_nodes:
                    # Create entity nodes + edges (memory node already exists)
                    for en in entity_nodes:
                        ename = en["properties"]["name"]
                        etype = en["entity_type"]
                        canonical_key = f"{ename.lower()}::{etype.lower()}"

                        # Find or create entity node
                        existing_eid = backend._find_entity_by_canonical_key(canonical_key)
                        if existing_eid:
                            eid = existing_eid
                        else:
                            import uuid
                            eid = str(uuid.uuid4())
                            backend.add_node(eid, {
                                "content": ename,
                                "name": ename,
                                "entity_type": etype,
                                "canonical_key": canonical_key,
                                "memory_type": "entity",
                            }, memory_type="entity")
                            entities_created += 1

                        backend.add_edge(item_id, eid, "CONTAINS_ENTITY", {})
                        backend.add_edge(eid, item_id, "MENTIONED_IN", {})

                    extracted += 1
            except Exception as e:
                logger.warning("Re-extraction failed for %s: %s", item_id, e)
                skipped += 1

        elapsed = time.time() - t0

    return {
        "extracted": extracted,
        "entities_created": entities_created,
        "skipped": skipped,
        "total": len(rows),
        "elapsed_s": round(elapsed, 1),
    }


# ── DIST-DAEMON-1: Memory operation endpoints ──────────────────────────────


class IngestRequest(BaseModel):
    content: str
    memory_type: str = "episodic"
    context: Optional[dict] = None
    properties: Optional[dict] = None  # user-supplied key-value properties
    profile_name: Optional[str] = None  # accepted for service contract alignment, ignored in lite
    extractor_name: Optional[str] = None  # DIST-OBSIDIAN-LITE-1: SDK contract; ignored in lite (env-driven tier split decides)
    cwd: Optional[str] = None  # HOOK-RECALL-RELEVANCE-1 G3.B: workspace_id derivation source
    workspace_id: Optional[str] = None  # explicit override; else derived from cwd


@api.post("/ingest")
def ingest_endpoint(body: IngestRequest) -> dict:
    """Ingest content through the pipeline. Two-tier when LLM key available.

    Tier 1 (sync, ~4ms): spaCy + EntityRuler → returns item_id immediately.
    Tier 2 (async, ~740ms): background LLM extraction if API key is set.

    HOOK-RECALL-RELEVANCE-1 G3.B: stamps `metadata.workspace_id` on the item
    so future workspace-scoped recall can filter. Derivation priority:
      explicit body.workspace_id > derive_workspace_id(body.cwd) >
      SMARTMEMORY_WORKSPACE_ID env var > None (legacy untagged).
    """
    import os
    import time
    from smartmemory_app.recall_format import derive_workspace_id

    memory_type = body.memory_type
    if body.context and "memory_type" in body.context:
        memory_type = body.context["memory_type"]

    # Auto-stamp workspace_id so this item is filterable in scoped recall.
    workspace_id = body.workspace_id or derive_workspace_id(body.cwd)
    properties = dict(body.properties or {})
    if workspace_id and "workspace_id" not in properties:
        properties["workspace_id"] = workspace_id

    has_llm = bool(os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"))

    if has_llm:
        # Two-tier: Tier 1 sync (spaCy), enqueue Tier 2 (LLM) via SQLite queue.
        # A separate worker process drains the queue — no threading issues.
        with _rw_lock:
            from smartmemory_app.storage import ingest
            result = ingest(body.content, memory_type, sync=False, properties=properties)
            item_id = result["item_id"] if isinstance(result, dict) else result
            raw_ids = result.get("entity_ids", {}) if isinstance(result, dict) else {}
            entity_ids = {k.lower(): v for k, v in raw_ids.items()} if raw_ids else {}
            already_queued = result.get("queued", False) if isinstance(result, dict) else False
            if not already_queued:
                from smartmemory_app.enrichment_queue import enqueue
                enqueue(item_id, entity_ids)
        return {"item_id": item_id}
    else:
        # No LLM — full sync pipeline
        with _rw_lock:
            from smartmemory_app.storage import ingest
            item_id = ingest(body.content, memory_type, properties=properties)
        return {"item_id": item_id}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[dict] = None  # property filters (e.g. {"project": "atlas"})
    enable_hybrid: bool = True  # DIST-OBSIDIAN-LITE-1: SDK contract; no-op in lite (FTS5+vector always on)
    memory_type: Optional[str] = None  # DIST-OBSIDIAN-LITE-1: SDK contract; folded into filters


@api.post("/search")
def search_endpoint(body: SearchRequest) -> dict:
    """Search memories. Returns {items: [...]} matching post-CORE-CRUD-LIST service contract."""
    from fastapi import HTTPException

    # DIST-OBSIDIAN-LITE-1: fold memory_type into filters so storage.search sees
    # a single filter dict regardless of which contract surface the caller used.
    filters = dict(body.filters or {})
    if body.memory_type:
        filters.setdefault("memory_type", body.memory_type)

    with _rw_lock:
        from smartmemory_app.storage import search
        try:
            results = search(body.query, body.top_k, filters=filters or None)
        except NotImplementedError as e:
            raise HTTPException(status_code=501, detail=str(e))
    return {"items": results}


# recall endpoint moved above /{memory_id} to avoid wildcard route capture


# ── DIST-OBSIDIAN-LITE-1: PATCH + DELETE for SDK CRUD parity ───────────────


class UpdateRequest(BaseModel):
    """CORE-CRUD-UPDATE-1 contract; matches SDK MemoryAPI.update body."""
    content: Optional[str] = None
    metadata: Optional[dict] = None
    properties: Optional[dict] = None
    write_mode: Optional[str] = "merge"  # 'merge' | 'replace'


@api.patch("/{memory_id}")
def update_memory_item(memory_id: str, body: UpdateRequest) -> dict:
    """Update a memory's properties. Maps to SmartMemory.update_properties.

    DIST-OBSIDIAN-LITE-1: enables the Obsidian plugin's metadata-only re-ingest
    path (PATCH instead of POST) when running against the daemon.
    """
    mem = _get_mem()
    from smartmemory_app.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        # RemoteMemory has no update_node — proxy mode is out of scope for
        # DIST-OBSIDIAN-LITE-1. Surface explicitly rather than AttributeError.
        raise HTTPException(status_code=501, detail="PATCH not available in remote-proxy mode")
    with _rw_lock:
        # CORE-CRUD-UPDATE-1: properties wins over content/metadata conveniences
        props = dict(body.properties or {})
        if body.content is not None and "content" not in props:
            props["content"] = body.content
        if body.metadata is not None:
            # Flat-merge metadata into properties; deep-merge happens on the
            # service side, but daemon's storage is single-level by convention.
            for k, v in body.metadata.items():
                props.setdefault(k, v)
        try:
            mem.update_properties(memory_id, props, write_mode=body.write_mode or "merge")
        except ValueError as e:
            # smart_memory.py:2012 → memory/pipeline/stages/crud.py:409 raises
            # ValueError("Node {item_id} not found in graph.") on missing item.
            # Match on the message so other ValueError categories (validation
            # failures, etc.) bubble as 500 rather than being mis-attributed
            # to a missing item.
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail="Memory not found")
            raise
    return {"item_id": memory_id, "updated": True}


@api.delete("/{memory_id}")
def delete_memory_item(memory_id: str) -> Response:
    """Delete a memory item.

    DIST-OBSIDIAN-LITE-1: lifted from the prior 405 ('Local viewer is read-only')
    policy. Single-user local deployment — no governance reason to refuse delete.
    Plugin uses this for the 'Danger: purge Obsidian-origin memories' command and
    for content-change re-ingest dedupe (workaround for CORE-INGEST-DEDUPE-1).

    Note: graph router's DELETE /graph/nodes/{id} stays 405 — that's a different
    surface (entity-node ops, not memory CRUD).
    """
    mem = _get_mem()
    from smartmemory_app.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        return Response(status_code=501, content="DELETE not available in remote-proxy mode")
    with _rw_lock:
        # SmartMemory.delete() at smart_memory.py:2182 delegates to crud.delete()
        # at crud.py:287, which already cascades to vector store + Vec_* nodes.
        # Going through backend.remove_node() directly would leak vector entries.
        ok = mem.delete(memory_id)
    return Response(status_code=204 if ok else 404)
