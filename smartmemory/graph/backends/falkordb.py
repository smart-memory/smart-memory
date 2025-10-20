import logging
from typing import Any, Dict, List, Optional, Tuple

from falkordb import FalkorDB
from smartmemory.graph.backends.backend import SmartGraphBackend
from smartmemory.utils import unflatten_dict, flatten_dict, get_config

logger = logging.getLogger(__name__)


class FalkorDBBackend(SmartGraphBackend):
    """Minimal RedisGraph/FalkorDB backend implementing the SmartGraphBackend interface.

    This adapter lets Agentic Memory switch from Neo4j to FalkorDB by flipping
    `graph_db.backend_class` in `config.json` to ``FalkorDBBackend``.

    Notes
    -----
    * FalkorDB re-uses openCypher; parameter placeholders (e.g. `$name`) are supported.
    * The backend keeps the entire graph inside the Redis module. Ensure sufficient
      RAM or shard via Redis Cluster.
    * Only core CRUD and simple search are implemented for now. Add advanced
      analytics or vector search later if needed.
    """

    def __init__(
            self,
            host: Optional[str] = None,
            port: Optional[int] = None,
            graph_name: str = "smartmemory",
            config_path: str = "config.json",
    ):
        config = get_config("graph_db")

        # Use direct config access with explicit parameter override
        self.host = host or config.host
        self.port = port or config.port
        self.graph_name = graph_name or config.get("graph_name", "smartmemory")

        self.db = FalkorDB(host=self.host, port=self.port)
        self.graph = self.db.select_graph(self.graph_name)

    # ---------- Capability Checks ----------
    def has_capability(self, name: str) -> bool:
        if name in {"vector"}:  # RedisGraph lacks native vector ops
            return False
        return False

    # ---------- Utility ----------

    def _query(self, cypher: str, params: Optional[Dict[str, Any]] = None):
        """Run a Cypher query and return raw records list."""
        res = self.graph.query(cypher, params or {})
        return res.result_set if hasattr(res, "result_set") else []

    # ---------- Bulk helpers ----------
    def add_nodes_bulk(self, nodes: List[Dict[str, Any]]):
        if not nodes:
            return
        for n in nodes:
            item_id = n.get("item_id")
            label = n.get("memory_type", "Node").capitalize()
            props = flatten_dict(n)
            query = f"MERGE (n:{label} {{item_id: $item_id}}) SET n += $props"
            self.graph.query(query, {"item_id": item_id, "props": props})

    def add_edges_bulk(self, edges: List[Tuple[str, str, str, Dict[str, Any]]]):
        if not edges:
            return
        for src, tgt, etype, props in edges:
            query = f"MATCH (a {{item_id: $src}}), (b {{item_id: $tgt}}) MERGE (a)-[r:{etype.upper()}]->(b) SET r += $props"
            self.graph.query(query, {"src": src, "tgt": tgt, "props": flatten_dict(props)})

    # ---------- CRUD ----------

    def clear(self):
        try:
            self.graph.delete()
        except Exception as e:
            # Ignore error if graph doesn't exist yet
            if "Invalid graph operation on empty key" not in str(e):
                raise

    def add_node(
            self,
            item_id: Optional[str],
            properties: Dict[str, Any],
            valid_time: Optional[Tuple] = None,
            created_at: Optional[Tuple] = None,
            memory_type: Optional[str] = None,
            is_global: bool = False,
    ):
        from smartmemory.utils.context import get_user_id, get_workspace_id

        label = memory_type.capitalize() if memory_type else "Node"
        props = flatten_dict(properties)

        # Detect write mode marker (used by CRUD.update_memory_node for replace semantics)
        write_mode = props.pop('_write_mode', None)

        # Add user_id/workspace_id/tenant_id for scoped nodes
        user_id = get_user_id()
        workspace_id = get_workspace_id()
        # tenant_id should match workspace_id for tenant isolation
        tenant_id = workspace_id  # In multi-tenant setup, workspace_id IS the tenant_id
        if not is_global and user_id:
            props["user_id"] = user_id
        if not is_global and workspace_id:
            props["workspace_id"] = workspace_id
        if not is_global and tenant_id:
            props["tenant_id"] = tenant_id

        # Store is_global in properties (will be used internally but not persisted)
        props["is_global"] = is_global

        # Build individual SET clauses for each property to avoid parameter expansion issues
        set_clauses = []
        params = {"item_id": item_id}

        for key, value in props.items():
            # Skip problematic property types that cause FalkorDB issues
            if value is None:
                continue

            # Skip embedding arrays entirely as they cause FalkorDB Cypher syntax errors
            if key == 'embedding' or isinstance(value, (list, tuple, dict)):
                continue

            # Convert datetime objects to strings
            if hasattr(value, 'isoformat'):
                value = value.isoformat()

            # Skip empty strings
            if isinstance(value, str) and value == "":
                continue

            # Only allow basic scalar types (str, int, float, bool)
            if not isinstance(value, (str, int, float, bool)):
                continue

            param_key = f"prop_{key}"
            set_clauses.append(f"n.{key} = ${param_key}")
            params[param_key] = value

        # If replace semantics requested, remove existing properties first (except item_id)
        if write_mode == 'replace':
            try:
                # Get existing keys
                existing_res = self._query("MATCH (n {item_id: $item_id}) RETURN n LIMIT 1", {"item_id": item_id})
                if existing_res and existing_res[0]:
                    node_obj = existing_res[0][0]
                    if hasattr(node_obj, 'properties'):
                        existing_keys = list(node_obj.properties.keys())
                    else:
                        existing_keys = [k for k in vars(node_obj).keys() if not k.startswith('_')]
                    # Exclude item_id
                    keys_to_remove = [k for k in existing_keys if k != 'item_id']
                    if keys_to_remove:
                        remove_clause = ", ".join([f"n.{k}" for k in keys_to_remove])
                        remove_query = f"MATCH (n:{label} {{item_id: $item_id}}) REMOVE {remove_clause}"
                        self._query(remove_query, {"item_id": item_id})
            except Exception:
                # Best-effort removal; continue to set new properties
                pass

        if set_clauses:
            set_clause = "SET " + ", ".join(set_clauses)
            query = f"MERGE (n:{label} {{item_id: $item_id}}) {set_clause} RETURN n"
        else:
            query = f"MERGE (n:{label} {{item_id: $item_id}}) RETURN n"

        self._query(query, params)
        return {"item_id": item_id, "properties": props}

    def add_edge(
            self,
            source_id: str,
            target_id: str,
            edge_type: str,
            properties: Dict[str, Any],
            valid_time: Optional[Tuple] = None,
            created_at: Optional[Tuple] = None,
            memory_type: Optional[str] = None,
    ):
        from smartmemory.utils.context import get_workspace_id

        # Attach workspace_id to relationship properties if present
        props_in = dict(properties or {})
        ws = get_workspace_id()
        if ws and "workspace_id" not in props_in:
            props_in["workspace_id"] = ws

        params = {
            "source": source_id,
            "target": target_id,
            "props": flatten_dict(props_in),
        }
        query = (
            f"MATCH (a {{item_id: $source}}), (b {{item_id: $target}}) "
            f"MERGE (a)-[r:{edge_type.upper()}]->(b) "
            f"SET r += $props"
        )

        # CRITICAL DEBUG: Add logging to understand edge creation failures
        try:
            result = self._query(query, params)
            # Verify the edge was actually created
            verify_query = f"MATCH (a {{item_id: $source}})-[r:{edge_type.upper()}]->(b {{item_id: $target}}) RETURN count(r) as edge_count"
            verify_result = self._query(verify_query, {"source": source_id, "target": target_id})
            edge_count = verify_result[0][0] if verify_result and verify_result[0] else 0

            if edge_count == 0:
                # Edge creation failed - check if nodes exist
                source_check = self._query("MATCH (n {item_id: $id}) RETURN count(n) as count", {"id": source_id})
                target_check = self._query("MATCH (n {item_id: $id}) RETURN count(n) as count", {"id": target_id})
                source_exists = source_check[0][0] if source_check and source_check[0] else 0
                target_exists = target_check[0][0] if target_check and target_check[0] else 0

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "EDGE CREATION FAILED: %s --[%s]--> %s | source_exists=%s target_exists=%s",
                        source_id, edge_type, target_id, bool(source_exists), bool(target_exists)
                    )
                return False
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("EDGE CREATED: %s --[%s]--> %s", source_id, edge_type, target_id)
                return True

        except Exception as e:
            logger.warning("EDGE CREATION ERROR: %s --[%s]--> %s: %s", source_id, edge_type, target_id, e)
            return False

    def get_node(self, item_id: str, as_of_time: Optional[str] = None):
        res = self._query("MATCH (n {item_id: $item_id}) RETURN n LIMIT 1", {"item_id": item_id})
        if not res or not res[0]:
            return None

        # Extract properties from FalkorDB Node object
        node = res[0][0]
        if hasattr(node, 'properties'):
            props = dict(node.properties)
        else:
            # Fallback to direct attribute access if properties attribute is not available
            props = {k: v for k, v in vars(node).items()
                     if not k.startswith('_') and k != 'properties'}

        # Ensure item_id is included in the returned properties at the top level
        props['item_id'] = item_id

        # Remove internal properties that shouldn't be exposed
        props.pop('is_global', None)

        # ISOMORPHIC TRANSFORM: Unflatten the properties to restore original nested structure
        # This makes the backend a true black box - you get back what you put in
        from smartmemory.utils import unflatten_dict
        
        # Separate system properties from user metadata
        system_props = {}
        user_metadata = {}
        
        for key, value in props.items():
            # Keep system properties flat
            if key in ['item_id', 'node_category', 'memory_type', 'content', 'user_id']:
                system_props[key] = value
            else:
                # User metadata gets unflattened to restore nested structure
                user_metadata[key] = value
        
        # Unflatten user metadata to restore original structure
        if user_metadata:
            unflattened_metadata = unflatten_dict(user_metadata)
            # Merge system props with unflattened metadata
            system_props.update(unflattened_metadata)
        
        return system_props

    def get_neighbors(
            self,
            item_id: str,
            edge_type: Optional[str] = None,
            as_of_time: Optional[str] = None,
    ):
        if edge_type:
            query = (
                f"MATCH (n {{item_id: $item_id}})-[:{edge_type.upper()}]-(m) RETURN m"
            )
        else:
            query = "MATCH (n {item_id: $item_id})--(m) RETURN m"
        res = self._query(query, {"item_id": item_id})
        out = []
        for record in res:
            node = record[0]
            # Handle both dict-like and Node objects
            if hasattr(node, 'keys') and hasattr(node, 'values'):
                props = dict(zip(node.keys(), node.values()))
            elif hasattr(node, 'properties'):
                # FalkorDB Node object
                props = dict(node.properties)
            elif isinstance(node, dict):
                props = node
            else:
                # Fallback: try to convert to dict
                try:
                    props = dict(node)
                except:
                    continue
            out.append(unflatten_dict(props))
        return out

    def get_all_edges(self):
        """Get all edges in the graph for debugging purposes."""
        try:
            query = "MATCH (a)-[r]->(b) RETURN a.item_id as source_id, type(r) as edge_type, b.item_id as target_id, r"
            result = self._query(query, {})
            edges = []
            for record in result:
                # FalkorDB returns results as tuples/lists
                if len(record) >= 4:
                    source_id = record[0] if record[0] else 'unknown'
                    edge_type = record[1] if record[1] else 'unknown'
                    target_id = record[2] if record[2] else 'unknown'
                    edge_obj = record[3]

                    # Extract edge properties
                    edge_props = {}
                    if hasattr(edge_obj, 'properties'):
                        edge_props = dict(edge_obj.properties)

                    edge_info = {
                        'source_id': source_id,
                        'target_id': target_id,
                        'edge_type': edge_type,
                        'properties': edge_props
                    }
                    edges.append(edge_info)
            return edges
        except Exception as e:
            logger.debug("Error getting all edges: %s", e)
            return []

    def remove_node(self, item_id: str):
        self._query("MATCH (n {item_id: $item_id}) DETACH DELETE n", {"item_id": item_id})
        return True

    def remove_edge(self, source_id: str, target_id: str, edge_type: Optional[str] = None):
        if edge_type:
            query = (
                f"MATCH (a {{item_id: $source}})-[r:{edge_type.upper()}]->(b {{item_id: $target}}) DELETE r"
            )
        else:
            query = (
                "MATCH (a {item_id: $source})-[r]->(b {item_id: $target}) DELETE r"
            )
        self._query(query, {"source": source_id, "target": target_id})
        return True

    # ---------- Read helpers for transactional layer ----------

    def node_exists(self, item_id: str) -> bool:
        try:
            res = self._query("MATCH (n {item_id: $item_id}) RETURN count(n)", {"item_id": item_id})
            if res and res[0]:
                val = res[0][0]
                return int(val) > 0
        except Exception:
            return False
        return False

    def get_properties(self, item_id: str) -> Dict[str, Any]:
        props = self.get_node(item_id) or {}
        # Remove internal flags
        props.pop('is_global', None)
        return props

    def set_properties(self, item_id: str, properties: Dict[str, Any]) -> bool:
        # Update only provided scalar properties
        from smartmemory.utils import flatten_dict
        props = flatten_dict(properties or {})
        if not props:
            return True
        set_parts = []
        params: Dict[str, Any] = {"item_id": item_id}
        for k, v in props.items():
            if v is None:
                continue
            if hasattr(v, 'isoformat'):
                v = v.isoformat()
            if not isinstance(v, (str, int, float, bool)):
                continue
            pk = f"p_{k}"
            set_parts.append(f"n.{k} = ${pk}")
            params[pk] = v
        if not set_parts:
            return True
        q = f"MATCH (n {{item_id: $item_id}}) SET {', '.join(set_parts)}"
        self._query(q, params)
        return True

    def edge_exists(self, source_id: str, target_id: str, relation_type: str) -> bool:
        try:
            q = (
                f"MATCH (a {{item_id: $s}})-[r:{relation_type.upper()}]->(b {{item_id: $t}}) RETURN count(r)"
            )
            res = self._query(q, {"s": source_id, "t": target_id})
            if res and res[0]:
                return int(res[0][0]) > 0
        except Exception:
            return False
        return False

    def get_edge_properties(self, source_id: str, target_id: str, relation_type: str) -> Optional[Dict[str, Any]]:
        try:
            q = (
                f"MATCH (a {{item_id: $s}})-[r:{relation_type.upper()}]->(b {{item_id: $t}}) RETURN r LIMIT 1"
            )
            res = self._query(q, {"s": source_id, "t": target_id})
            if res and res[0] and res[0][0] is not None:
                r = res[0][0]
                if hasattr(r, 'properties'):
                    return dict(r.properties)
                # best-effort
                return {k: v for k, v in vars(r).items() if not k.startswith('_')}
        except Exception:
            return None
        return None

    def list_edges(self, item_id: str) -> List[Dict[str, Any]]:
        try:
            return self.get_edges_for_node(item_id)
        except Exception:
            return []

    def archive(self, item_id: str) -> bool:
        self._query("MATCH (n {item_id: $id}) SET n.archived = true", {"id": item_id})
        return True

    def unarchive(self, item_id: str) -> bool:
        self._query("MATCH (n {item_id: $id}) SET n.archived = false", {"id": item_id})
        return True

    # ---------- Vector similarity ----------
    def vector_similarity_search(self, embedding: List[float], top_k: int = 5, prop_key: str = "embedding"):
        # Fallback to base implementation (Python-side) until Redis vector module available
        return super().vector_similarity_search(embedding, top_k, prop_key)

    # ---------- Query helpers ----------

    def search_nodes(self, query: Dict[str, Any], is_global: bool = False):
        """Search for nodes matching query properties.
        
        Args:
            query: Dictionary of property filters (simple equality only)
            is_global: Whether to search globally or scope to user/workspace
            
        Returns:
            List of node dictionaries matching the query
        """
        from smartmemory.utils.context import get_user_id, get_workspace_id

        clauses = []
        params = {}

        # Add query clauses (simple equality only)
        for idx, (k, v) in enumerate(query.items()):
            param_key = f"p{idx}"
            clauses.append(f"n.{k} = ${param_key}")
            params[param_key] = v

        # Add user/workspace scoping clauses
        user_id = get_user_id()
        workspace_id = get_workspace_id()
        if is_global:
            # For global queries, exclude user-scoped nodes
            if user_id:
                clauses.append("(NOT EXISTS(n.user_id) OR n.user_id IS NULL)")
            if workspace_id:
                clauses.append("(NOT EXISTS(n.workspace_id) OR n.workspace_id IS NULL)")
        else:
            # For user queries with actual query parameters, apply user scoping
            if user_id and query:  # Only apply user scoping if there are query parameters
                clauses.append(f"n.user_id = $user_id")
                params["user_id"] = user_id
            if workspace_id and query:
                clauses.append("n.workspace_id = $workspace_id")
                params["workspace_id"] = workspace_id
            # If no user_id or empty query, return all nodes (unscoped search)

        if clauses:
            where_clause = " AND ".join(clauses)
            cypher = f"MATCH (n) WHERE {where_clause} RETURN n"
        else:
            cypher = "MATCH (n) RETURN n"
        res = self._query(cypher, params)
        result = []
        for record in res:
            node = record[0]
            if hasattr(node, 'properties'):
                props = dict(node.properties)
            else:
                # Fallback to direct attribute access if properties attribute is not available
                props = {k: v for k, v in vars(node).items()
                         if not k.startswith('_') and k != 'properties'}

            # Remove internal properties that shouldn't be exposed
            props.pop('is_global', None)

            # Don't use unflatten_dict to preserve flat structure with item_id
            result.append(props)
        return result

    def search_nodes_by_type_or_tag(self, type_or_tag: str, is_global: bool = False) -> List[Dict[str, Any]]:
        """Search for nodes by type OR tag using proper Cypher OR logic.
        
        This replaces the MongoDB-style {"$or": [{"type": x}, {"tags": x}]} pattern.
        
        Args:
            type_or_tag: The type or tag value to search for
            is_global: Whether to search globally or scope to user/workspace
            
        Returns:
            List of node dictionaries matching the query
        """
        from smartmemory.utils.context import get_user_id, get_workspace_id

        clauses = []
        params = {"type_or_tag": type_or_tag}

        # Build OR condition for type or tags
        or_clause = "(n.type = $type_or_tag OR n.tags = $type_or_tag)"
        clauses.append(or_clause)

        # Add user/workspace scoping clauses
        user_id = get_user_id()
        workspace_id = get_workspace_id()
        if is_global:
            if user_id:
                clauses.append("(NOT EXISTS(n.user_id) OR n.user_id IS NULL)")
            if workspace_id:
                clauses.append("(NOT EXISTS(n.workspace_id) OR n.workspace_id IS NULL)")
        else:
            if user_id:
                clauses.append("n.user_id = $user_id")
                params["user_id"] = user_id
            if workspace_id:
                clauses.append("n.workspace_id = $workspace_id")
                params["workspace_id"] = workspace_id

        where_clause = " AND ".join(clauses)
        cypher = f"MATCH (n) WHERE {where_clause} RETURN n"
        res = self._query(cypher, params)
        
        result = []
        for record in res:
            node = record[0]
            if hasattr(node, 'properties'):
                props = dict(node.properties)
            else:
                props = {k: v for k, v in vars(node).items()
                         if not k.startswith('_') and k != 'properties'}
            props.pop('is_global', None)
            result.append(props)
        return result

    def get_all_nodes(self):
        """Get all nodes in the graph."""
        query = "MATCH (n) RETURN n"
        result = self._query(query)
        nodes = []
        for record in result:
            if record and len(record) > 0:
                node = record[0]
                if hasattr(node, 'properties'):
                    node_dict = dict(node.properties)
                    nodes.append(node_dict)
                elif isinstance(node, dict):
                    nodes.append(node)
        return nodes

    def get_edges_for_node(self, item_id: str) -> List[Dict[str, Any]]:
        """Get all edges (relationships) involving a specific node.
        
        Returns a list of edge dictionaries with 'source', 'target', and 'type' keys.
        """
        query = """
        MATCH (n {item_id: $item_id})-[r]-(m)
        RETURN n.item_id as source, type(r) as rel_type, m.item_id as target, 
               startNode(r).item_id as start_node, endNode(r).item_id as end_node
        """
        result = self._query(query, {"item_id": item_id})

        edges = []
        for record in result:
            if record and len(record) >= 5:
                # Use the actual start/end node info to determine direction
                start_node = record[3]
                end_node = record[4]
                rel_type = record[1]

                edges.append({
                    "source": start_node,
                    "target": end_node,
                    "type": rel_type
                })

        return edges

    # ---------- Stats helpers ----------
    def get_node_count(self) -> int:
        """Return total number of nodes in the graph."""
        try:
            res = self._query("MATCH (n) RETURN count(n)")
            if res and res[0]:
                val = res[0][0]
                try:
                    return int(val)
                except Exception:
                    # Some drivers return dict-like or typed values
                    return int(str(val))
        except Exception:
            pass
        return 0

    def get_edge_count(self) -> int:
        """Return total number of edges (relationships) in the graph."""
        try:
            res = self._query("MATCH ()-[r]->() RETURN count(r)")
            if res and res[0]:
                val = res[0][0]
                try:
                    return int(val)
                except Exception:
                    return int(str(val))
        except Exception:
            pass
        return 0

    def get_counts(self) -> Dict[str, int]:
        """Return a dict with node_count and edge_count for fast stats emission."""
        return {
            "node_count": self.get_node_count(),
            "edge_count": self.get_edge_count(),
        }

    # ---------- (De)Serialization ----------

    def serialize(self) -> Any:
        nodes_res = self._query("MATCH (n) RETURN n")
        edges_res = self._query("MATCH (a)-[r]->(b) RETURN a.item_id, b.item_id, type(r), r")
        nodes = []
        for rec in nodes_res:
            props = dict(zip(rec[0].keys(), rec[0].values()))  # type: ignore[index]
            nodes.append(props)
        edges = []
        for src, tgt, etype, rprops in edges_res:
            edges.append(
                {
                    "source": src,
                    "target": tgt,
                    "type": etype,
                    "properties": rprops,
                }
            )
        return {"nodes": nodes, "edges": edges}

    def deserialize(self, data: Any):
        self.clear()
        for node in data.get("nodes", []):
            item_id = node.pop("item_id")
            self.add_node(item_id, node)
        for edge in data.get("edges", []):
            self.add_edge(edge["source"], edge["target"], edge["type"], edge.get("properties") or {})

    def add_dual_node(
            self,
            item_id: str,
            memory_properties: Dict[str, Any],
            memory_type: str,
            entity_nodes: List[Dict[str, Any]] = None,
            is_global: bool = False,
    ):
        """
        Add a dual-node structure: one memory node + related entity nodes.
        
        Args:
            item_id: Unique identifier for the memory node
            memory_properties: Properties for the memory node (content, metadata, etc.)
            memory_type: Memory type for the memory node label (semantic, episodic, etc.)
            entity_nodes: List of entity node dicts with {entity_type, properties, relationships}
            is_global: Whether nodes are global or user-scoped
            
        Returns:
            Dict with memory_node_id and list of entity_node_ids
        """
        from smartmemory.utils.context import get_user_id

        # Prepare memory node
        memory_label = memory_type.capitalize()
        memory_props = flatten_dict(memory_properties)

        # Add user_id for user-scoped nodes
        user_id = get_user_id()
        if not is_global and user_id:
            memory_props["user_id"] = user_id
        memory_props["is_global"] = is_global

        # Build transaction query for atomic dual-node creation
        queries = []
        params = {"memory_id": item_id}

        # 1. Create memory node
        memory_set_clauses = []
        for key, value in memory_props.items():
            if self._is_valid_property(key, value):
                param_key = f"mem_{key}"
                memory_set_clauses.append(f"m.{key} = ${param_key}")
                params[param_key] = self._serialize_value(value)

        if memory_set_clauses:
            memory_query = f"CREATE (m:{memory_label} {{item_id: $memory_id}}) SET {', '.join(memory_set_clauses)}"
        else:
            memory_query = f"CREATE (m:{memory_label} {{item_id: $memory_id}})"
        queries.append(memory_query)

        # 2. Create entity nodes and relationships
        entity_ids = []
        if entity_nodes:
            for i, entity_node in enumerate(entity_nodes):
                entity_type = entity_node.get('entity_type', 'Entity')
                entity_props = entity_node.get('properties') or {}
                entity_relationships = entity_node.get('relations', [])

                # Generate unique entity ID
                entity_id = f"{item_id}_entity_{i}"
                entity_ids.append(entity_id)
                entity_label = entity_type.capitalize()

                # Add user context to entity
                if not is_global and user_id:
                    entity_props["user_id"] = user_id
                entity_props["is_global"] = is_global
                # SECURITY: Add tenant metadata from memory properties for tenant isolation
                if 'tenant_id' in memory_props:
                    entity_props["tenant_id"] = memory_props['tenant_id']
                if 'workspace_id' in memory_props:
                    entity_props["workspace_id"] = memory_props['workspace_id']

                # Build entity creation query
                entity_set_clauses = []
                for key, value in entity_props.items():
                    if self._is_valid_property(key, value):
                        param_key = f"ent_{i}_{key}"
                        entity_set_clauses.append(f"e{i}.{key} = ${param_key}")
                        params[param_key] = self._serialize_value(value)

                params[f"entity_id_{i}"] = entity_id

                if entity_set_clauses:
                    entity_query = f"CREATE (e{i}:{entity_label} {{item_id: $entity_id_{i}}}) SET {', '.join(entity_set_clauses)}"
                else:
                    entity_query = f"CREATE (e{i}:{entity_label} {{item_id: $entity_id_{i}}})"
                queries.append(entity_query)

                # Create CONTAINS_ENTITY relationship
                queries.append(f"CREATE (m)-[:CONTAINS_ENTITY]->(e{i})")

                # Create semantic relationships between entities
                for rel in entity_relationships:
                    target_idx = rel.get('target_index')
                    rel_type = rel.get('relation_type', 'RELATED')
                    if target_idx is not None and target_idx < len(entity_nodes):
                        queries.append(f"CREATE (e{i})-[:{rel_type}]->(e{target_idx})")

        # Create memory node first with all properties
        memory_params = {'memory_id': item_id}
        # Add node_category for proper querying
        memory_params['mem_node_category'] = 'memory'
        # Preserve user scoping on memory node (second-phase creation path)
        if not is_global and user_id:
            memory_params['mem_user_id'] = user_id
        for key, value in memory_properties.items():
            if self._is_valid_property(key, value):
                memory_params[f'mem_{key}'] = value

        # Build memory node creation query
        memory_set_parts = ['m.node_category = $mem_node_category']
        if not is_global and user_id:
            memory_set_parts.append('m.user_id = $mem_user_id')
        for key in memory_properties.keys():
            if f'mem_{key}' in memory_params:
                memory_set_parts.append(f'm.{key} = $mem_{key}')

        if memory_set_parts:
            memory_query = f"CREATE (m:{memory_label} {{item_id: $memory_id}}) SET {', '.join(memory_set_parts)} RETURN m"
        else:
            memory_query = f"CREATE (m:{memory_label} {{item_id: $memory_id}}) RETURN m"

        memory_result = self._query(memory_query, memory_params)
        entity_node_ids = []

        # Create entity nodes and relationships separately
        if entity_nodes:
            for i, entity_node in enumerate(entity_nodes):
                entity_type = entity_node.get('entity_type', 'Entity')
                entity_label = entity_type.capitalize()
                entity_id = f"{item_id}_entity_{i}"
                entity_node_ids.append(entity_id)

                # Prepare entity properties
                entity_properties = entity_node.get('properties') or {}
                entity_params = {'entity_id': entity_id}
                # Add node_category for proper querying
                entity_params['ent_node_category'] = 'entity'
                # Preserve user scoping on entity nodes
                if not is_global and user_id:
                    entity_params['ent_user_id'] = user_id
                # SECURITY: Add tenant metadata from memory properties for tenant isolation
                if 'tenant_id' in memory_props:
                    entity_params['ent_tenant_id'] = memory_props['tenant_id']
                if 'workspace_id' in memory_props:
                    entity_params['ent_workspace_id'] = memory_props['workspace_id']
                    
                entity_set_parts = ['e.node_category = $ent_node_category']
                if not is_global and user_id:
                    entity_set_parts.append('e.user_id = $ent_user_id')
                if 'tenant_id' in memory_props:
                    entity_set_parts.append('e.tenant_id = $ent_tenant_id')
                if 'workspace_id' in memory_props:
                    entity_set_parts.append('e.workspace_id = $ent_workspace_id')

                for key, value in entity_properties.items():
                    if self._is_valid_property(key, value):
                        entity_params[f'ent_{key}'] = value
                        entity_set_parts.append(f'e.{key} = $ent_{key}')

                # Create entity node
                if entity_set_parts:
                    entity_query = f"CREATE (e:{entity_label} {{item_id: $entity_id}}) SET {', '.join(entity_set_parts)}"
                else:
                    entity_query = f"CREATE (e:{entity_label} {{item_id: $entity_id}})"

                self._query(entity_query, entity_params)

                # Create CONTAINS_ENTITY relationship
                contains_query = f"MATCH (m:{memory_label} {{item_id: $memory_id}}), (e:{entity_label} {{item_id: $entity_id}}) CREATE (m)-[:CONTAINS_ENTITY]->(e)"
                self._query(contains_query, {'memory_id': item_id, 'entity_id': entity_id})

        # Create semantic relationships between entities (after all entities exist)
        if entity_nodes:
            for i, entity_node in enumerate(entity_nodes):
                entity_type = entity_node.get('entity_type', 'Entity')
                entity_label = entity_type.capitalize()
                entity_id = f"{item_id}_entity_{i}"

                entity_relationships = entity_node.get('relations', [])
                for rel in entity_relationships:
                    target_idx = rel.get('target_index')
                    rel_type = rel.get('relation_type', 'RELATED')
                    if target_idx is not None and target_idx < len(entity_nodes):
                        target_entity_id = f"{item_id}_entity_{target_idx}"
                        target_entity_type = entity_nodes[target_idx].get('entity_type', 'Entity')
                        target_label = target_entity_type.capitalize()

                        rel_query = f"MATCH (e1:{entity_label} {{item_id: $source_id}}), (e2:{target_label} {{item_id: $target_id}}) CREATE (e1)-[:{rel_type}]->(e2)"
                        self._query(rel_query, {'source_id': entity_id, 'target_id': target_entity_id})

        result = memory_result

        return {
            "memory_node_id": item_id,
            "entity_node_ids": entity_node_ids,
            "memory_type": memory_type,
            "entity_count": len(entity_node_ids)
        }

    def _is_valid_property(self, key: str, value: Any) -> bool:
        """Check if a property is valid for FalkorDB storage."""
        if value is None or key == 'embedding':
            return False
        if isinstance(value, (list, tuple, dict)):
            return False
        if isinstance(value, str) and value == "":
            return False
        return isinstance(value, (str, int, float, bool))

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for FalkorDB storage."""
        if hasattr(value, 'isoformat'):
            return value.isoformat()
        return value
