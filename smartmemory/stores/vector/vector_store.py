import json
from datetime import datetime
from time import perf_counter

from smartmemory.observability.instrumentation import make_emitter
from smartmemory.stores.vector.backends.base import create_backend
from smartmemory.utils import get_config
from smartmemory.utils.context import get_user_id, get_workspace_id


class VectorStore:
    """
    Backend-agnostic vector store for semantic memory.
    Supports add, upsert, search, delete, get, and clear operations with metadata.

    Each instance is independent - no singleton pattern for better testability and isolation.
    
    Example usage:
        vector_store = VectorStore(collection_name="my_collection")
        vector_store.upsert(
            item_id="item123",
            embedding=embedding,
            node_ids=["nodeA", "nodeB"],
            chunk_ids=["chunk3", "chunk7"],
            metadata={"source": "summary"}
        )
        vector_store.clear()  # Deletes all vectors from the collection
    """

    # Preconfigured emitter for vector operations
    VEC_EMIT = make_emitter(component="vector_store", default_type="vector_operation")

    def _vec_data(
            self,
            *,
            item_id=None,
            embedding=None,
            meta=None,
            top_k=None,
            returned=None,
            deleted_count=None,
            t0=None,
    ):
        """Build a standard payload for vector operations, omitting None values.

        This centralizes repeated logic like computing dim, counts, collection name,
        and duration, so call sites can pass only what's relevant.
        """

        def _count(val):
            if isinstance(val, list):
                return len(val)
            return 1 if val else 0

        node_ids = meta.get("node_ids") if isinstance(meta, dict) else None
        chunk_ids = meta.get("chunk_ids") if isinstance(meta, dict) else None

        duration_ms = None
        if t0 is not None:
            try:
                duration_ms = (perf_counter() - t0) * 1000.0
            except Exception:
                duration_ms = None

        data = {
            "collection": getattr(self.collection, "name", "unknown"),
            "id": str(item_id) if item_id is not None else None,
            "dim": (len(embedding) if hasattr(embedding, "__len__") else None) if embedding is not None else None,
            "node_ids_count": _count(node_ids) if node_ids is not None else None,
            "chunk_ids_count": _count(chunk_ids) if chunk_ids is not None else None,
            "top_k": top_k,
            "returned": returned,
            "deleted_count": deleted_count,
            "duration_ms": duration_ms,
        }

        # Omit None entries
        return {k: v for k, v in data.items() if v is not None}

    def __init__(self, collection_name=None, persist_directory=None):
        """Construct a VectorStore that delegates to a configured backend."""
        vector_cfg = get_config('vector') or {}
        full_cfg = get_config()

        # Resolve effective collection name with namespace support
        ns = full_cfg.get('active_namespace') if isinstance(full_cfg, dict) else None
        use_ws_ns = bool(vector_cfg.get("use_workspace_namespace", False))
        ws = get_workspace_id() if use_ws_ns else None

        base_collection = collection_name or vector_cfg.get("collection_name") or "semantic_memory"
        # Chroma collections cannot contain ':'; use '_' as delimiter
        eff_collection = base_collection
        if ns:
            eff_collection = f"{eff_collection}_{ns}"
        if ws:
            eff_collection = f"{eff_collection}_{ws}"

        # Resolve persist directory for Chroma only
        if persist_directory is None:
            persist_directory = vector_cfg.get("persist_directory", ".chroma")

        backend_name = (vector_cfg.get('backend') or 'falkordb').lower()
        self._backend = create_backend(backend_name, eff_collection, persist_directory)
        self._collection_name = eff_collection
        # Expose collection for compatibility with _vec_data method
        self.collection = self._backend

    def add(self, item_id, embedding, metadata=None, node_ids=None, chunk_ids=None, is_global=False, workspace_id=None):
        """
        Add an embedding to the vector store. Supports cross-referencing multiple node_ids and chunk_ids.
        - item_id: unique vector entry ID (string)
        - embedding: list of floats
        - metadata: dict of additional metadata
        - node_ids: single string or list of graph node IDs
        - chunk_ids: single string or list of chunk IDs
        - user_id: user context for multi-tenancy (None for global)
        - is_global: if True, forcibly remove user_id from metadata
        All IDs are stored in metadata as lists for robust cross-referencing.
        """
        meta = metadata.copy() if metadata else {}
        meta["item_id"] = str(item_id)
        # Flatten 'properties' dict if present
        node = meta.pop("_node", None)
        if node:
            properties = node.pop('properties', None)
            if properties and isinstance(properties, dict):
                for k, v in properties.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    elif isinstance(v, datetime):
                        meta[k] = v.isoformat()
                    else:
                        meta[k] = json.dumps(v)
        # Normalize node_ids and chunk_ids to lists
        if node_ids is not None:
            meta["node_ids"] = node_ids if isinstance(node_ids, list) else [node_ids]
        if chunk_ids is not None:
            meta["chunk_ids"] = chunk_ids if isinstance(chunk_ids, list) else [chunk_ids]

        # Ensure all metadata values are ChromaDB-compatible (str, int, float, bool)
        def chroma_safe(val):
            if isinstance(val, (list, dict)):
                return json.dumps(val, default=str)
            if isinstance(val, datetime):
                return val.isoformat()
            return val

        meta = {k: chroma_safe(v) for k, v in meta.items() if v is not None}
        meta = {k: v for k, v in meta.items() if v is not None}
        # Multi-tenancy logic
        effective_user_id = get_user_id()
        effective_workspace_id = workspace_id if workspace_id is not None else get_workspace_id()
        if "user_id" in meta:
            # If user_id is specified in metadata, always use it (metadata wins)
            pass
        elif effective_user_id is None:
            # No user context, always treat as global
            meta.pop("user_id", None)
        elif is_global:
            meta.pop("user_id", None)
        else:
            meta["user_id"] = effective_user_id
        # workspace scope: best-effort tagging; allow explicit override in metadata
        if "workspace_id" in meta:
            pass
        elif effective_workspace_id:
            meta["workspace_id"] = effective_workspace_id
        t0 = perf_counter()
        self._backend.add(item_id=str(item_id), embedding=embedding, metadata=meta)
        # Best-effort emit with automatic context
        VectorStore.VEC_EMIT(None, "add", self._vec_data(item_id=item_id, embedding=embedding, meta=meta, t0=t0))

    def upsert(self, item_id, embedding, metadata=None, node_ids=None, chunk_ids=None, is_global=False, workspace_id=None):
        """
        Upsert an embedding to the vector store. Overwrites if the id exists, inserts if not.
        - user_id: user context for multi-tenancy (None for global)
        - is_global: if True, forcibly remove user_id from metadata
        """
        meta = metadata.copy() if metadata else {}
        if node_ids is not None:
            meta["node_ids"] = node_ids if isinstance(node_ids, list) else [node_ids]
        if chunk_ids is not None:
            meta["chunk_ids"] = chunk_ids if isinstance(chunk_ids, list) else [chunk_ids]

        # Ensure all metadata values are ChromaDB-compatible (str, int, float, bool)
        def chroma_safe(val):
            if isinstance(val, (list, dict)):
                return json.dumps(val, default=str)
            if isinstance(val, datetime):
                return val.isoformat()
            return val

        meta = {k: chroma_safe(v) for k, v in meta.items() if v is not None}
        meta = {k: v for k, v in meta.items() if v is not None}
        from smartmemory.utils.context import get_user_id, get_workspace_id
        effective_user_id = get_user_id()
        effective_workspace_id = workspace_id if workspace_id is not None else get_workspace_id()
        if "user_id" in meta:
            # If user_id is specified in metadata, always use it (metadata wins)
            pass
        elif effective_user_id is None:
            # No user context, always treat as global
            meta.pop("user_id", None)
        elif is_global:
            meta.pop("user_id", None)
        else:
            meta["user_id"] = effective_user_id
        # workspace scope tagging
        if "workspace_id" in meta:
            pass
        elif effective_workspace_id:
            meta["workspace_id"] = effective_workspace_id
        t0 = perf_counter()
        self._backend.upsert(item_id=str(item_id), embedding=embedding, metadata=meta)
        VectorStore.VEC_EMIT(None, "upsert", self._vec_data(item_id=item_id, embedding=embedding, meta=meta, t0=t0))

    def get(self, item_id, include_metadata: bool = True):
        """
        Fetch a single vector item by id. Returns a dict with keys:
        {"id", "embedding" (if backend provides), "metadata", "node_ids", "is_global"} when available,
        or None if not found or not supported by backend.
        """
        getter = getattr(self._backend, "get", None)
        if callable(getter):
            try:
                # Backend-specific signature may vary; prefer a simple get by id
                res = getter(item_id)
                if isinstance(res, list):
                    # Some backends return list; take first
                    res = res[0] if res else None
                return res
            except Exception as e:
                print(f"Warning: Vector backend get failed: {e}")
                return None
        # Not supported by backend
        return None

    def delete(self, item_id) -> bool:
        """
        Delete a single vector item by id. Returns True if deletion was attempted and backend acknowledged,
        False if not supported or failed.
        """
        deleter = getattr(self._backend, "delete", None)
        if callable(deleter):
            try:
                deleter(item_id)
                VectorStore.VEC_EMIT(None, "delete", self._vec_data(item_id=item_id))
                return True
            except Exception as e:
                print(f"Warning: Vector backend delete failed: {e}")
                return False
        return False

    def search(self, query_embedding, top_k=5, is_global=False, workspace_id=None):
        """
        Search the vector store. If is_global is True, return all results. If not, filter to user_id.
        """
        t0 = perf_counter()
        backend_results = self._backend.search(query_embedding=query_embedding, top_k=top_k * 2)
        hits = []
        count = 0
        from smartmemory.utils.context import get_user_id, get_workspace_id
        effective_user_id = get_user_id()
        effective_workspace_id = workspace_id if workspace_id is not None else get_workspace_id()
        for i, res in enumerate(backend_results):
            id_ = res.get("id")
            meta = res.get("metadata", {})
            # Global searches return everything regardless of scope
            if effective_user_id is None or is_global:
                if "user_id" in meta:
                    continue  # skip user-scoped vectors
            else:
                if meta.get("user_id") != effective_user_id:
                    continue
            # If a workspace_id is set, enforce match
            if effective_workspace_id is not None and meta.get("workspace_id") != effective_workspace_id:
                continue
            hit = {"id": id_}
            hit["metadata"] = self.deserialize_metadata(meta)
            if "score" in res:
                hit["score"] = res["score"]
            hits.append(hit)
            count += 1
            if count >= top_k:
                break
        VectorStore.VEC_EMIT(None, "search", self._vec_data(top_k=top_k, returned=len(hits), t0=t0))
        return hits

    def clear(self):
        """
        Delete all embeddings from the vector store collection.
        This operation is idempotent and safe. Useful for tests or resetting state.
        """
        try:
            t0 = perf_counter()
            # Delegate to backend; not all backends return a count
            self._backend.clear()
            VectorStore.VEC_EMIT(None, "clear", self._vec_data(deleted_count=None, t0=t0))
        except Exception as e:
            print(f"Warning: Vector store clear encountered error: {e}")

    @staticmethod
    def deserialize_metadata(meta):
        """
        Convert metadata values back to their original types if possible.
        Attempts to json.loads strings that look like lists/dicts, and parse ISO8601 dates.
        """
        import json
        from dateutil.parser import parse as parse_date
        def try_load(val):
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    try:
                        return parse_date(val)
                    except Exception:
                        return val
            return val

        return {k: try_load(v) for k, v in meta.items()}
