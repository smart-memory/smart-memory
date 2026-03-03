"""DIST-LITE-4/5: Graph API — local SQLite or remote hosted API.

Mounted at /memory by viewer_server.py — routes here are relative to that mount.
GET  /graph/full         → /memory/graph/full
POST /graph/edges        → /memory/graph/edges
GET  /list               → /memory/list
GET  /{id}/neighbors     → /memory/{id}/neighbors  (BEFORE /{id} — declaration order matters)
GET  /{id}               → /memory/{id}
DELETE /{id}             → /memory/{id}            (405 — local viewer is read-only)
DELETE /graph/nodes/{id} → /memory/graph/nodes/{id} (405 — local viewer is read-only)

DIST-LITE-5: each endpoint dispatches to RemoteMemory graph methods when mode=remote.
Unconfigured state → HTTP 503 (UnconfiguredError caught by _get_mem()).
"""
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Response
from pydantic import BaseModel

from smartmemory_app.config import UnconfiguredError
from smartmemory_app.storage import get_memory

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
    backend = _get_backend()
    snapshot = backend.serialize()
    nodes = [_flatten_node(n) for n in snapshot.get("nodes", [])]
    edges = snapshot.get("edges", [])
    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
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
        # Remote graph/full used as source; paginate client-side
        full = mem.get_graph_full()
        nodes = full.get("nodes", [])
        paginated = nodes[offset: offset + limit]
        return {"items": paginated, "total": len(nodes), "limit": limit, "offset": offset}
    backend = _get_backend()
    snapshot = backend.serialize()
    nodes = [_flatten_node(n) for n in snapshot.get("nodes", [])]
    paginated = nodes[offset: offset + limit]
    return {"items": paginated, "total": len(nodes), "limit": limit, "offset": offset}


# IMPORTANT: /{memory_id}/neighbors MUST be declared BEFORE /{memory_id}.
# FastAPI matches in declaration order — the bare /{memory_id} would otherwise
# capture /neighbors as the memory_id value.

@api.get("/{memory_id}/neighbors")
def get_neighbors(memory_id: str) -> dict:
    """get_neighbors() uses _row_to_node() — output is already flat, no transformation needed.

    Response key MUST be "neighbors", not "nodes".
    createFetchAdapter.getNeighbors reads: const neighbors = res?.neighbors || []
    (useGraphInteraction.js:106-107). Returning {"nodes": ...} produces an empty expansion.
    """
    mem = _get_mem()
    from smartmemory_app.remote_backend import RemoteMemory
    if isinstance(mem, RemoteMemory):
        return mem.get_neighbors(memory_id)
    backend = _get_backend()
    neighbors = backend.get_neighbors(memory_id)
    edges = backend.get_edges_for_node(memory_id)
    return {"neighbors": neighbors, "edges": edges}


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
    backend = _get_backend()
    node = backend.get_node(memory_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return node


@api.delete("/{memory_id}")
def delete_node_405(memory_id: str) -> Response:
    """Local viewer is read-only. Memory node deletes return 405."""
    return Response(status_code=405, content="Local viewer is read-only.")
