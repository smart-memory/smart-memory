import logging
from typing import Optional, List, Union, Any, Dict

from smartmemory.conversation.context import ConversationContext
from smartmemory.graph.smartgraph import SmartGraph
from smartmemory.integration.archive.archive_provider import get_archive_provider
from smartmemory.memory.base import MemoryBase
from smartmemory.memory.ingestion.flow import MemoryIngestionFlow
from smartmemory.memory.pipeline.stages.crud import CRUD
from smartmemory.memory.pipeline.stages.enrichment import Enrichment
from smartmemory.memory.pipeline.stages.evolution import EvolutionOrchestrator
from smartmemory.memory.pipeline.stages.graph_operations import GraphOperations
from smartmemory.memory.pipeline.stages.grounding import Grounding
from smartmemory.memory.pipeline.stages.linking import Linking
from smartmemory.memory.pipeline.stages.monitoring import Monitoring
from smartmemory.memory.pipeline.stages.personalization import Personalization
from smartmemory.memory.pipeline.stages.search import Search
from smartmemory.models.memory_item import MemoryItem
from smartmemory.plugins.resolvers.external_resolver import ExternalResolver

logger = logging.getLogger(__name__)


class SmartMemory(MemoryBase):
    """
    Unified agentic memory store combining semantic, episodic, procedural, and working memory.
    Delegates responsibilities to submodules for store management, linking, enrichment, grounding, and personalization.

    The canonical agentic ingestion flow is exposed via the ingest() method, which runs the full flowchart pipeline:
    adapt → classify → route → create → extract semantics → link → activate → enrich/feedback.
    This is the recommended entry point for all agentic workflows. Use add() only for direct low-level node insertion.

    All linking operations should be accessed via SmartMemory methods only.
    Do not use Linking directly; it is an internal implementation detail.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph = SmartGraph()  # Direct SmartGraph backend is the canonical store

        # Initialize stages with proper delegation
        self._graph_ops = GraphOperations(self._graph)
        self._crud = CRUD(self._graph)
        self._linking = Linking(self._graph)
        self._enrichment = Enrichment(self._graph)
        self._grounding = Grounding(self._graph)
        self._personalization = Personalization(self._graph)
        self._search = Search(self._graph)
        self._monitoring = Monitoring(self._graph)
        self._evolution = EvolutionOrchestrator(self)
        self._external_resolver = ExternalResolver()

        # Initialize temporal queries
        from smartmemory.temporal.queries import TemporalQueries
        self.temporal = TemporalQueries(self)
        self._temporal_context = None  # For time-travel context manager

        # Defer flow construction until needed (lazy)
        self._ingestion_flow = None
        
        # Set self on graph search module for SSG traversal
        self._graph.search.set_smart_memory(self)

    def clear(self):
        """Clear all memory from ALL storage backends comprehensively."""
        print("🧹 Clearing all memory storage backends...")

        # 1. Clear the graph backend (FalkorDB)
        self._graph_ops.clear_all()
        print("✅ Cleared Graph Database (FalkorDB)")

        # 2. Clear the vector database (ChromaDB)
        from smartmemory.stores.vector.vector_store import VectorStore
        try:
            # Create vector store instance directly
            vector_store = VectorStore()
            vector_store.clear()
            print("✅ Cleared Vector Store (ChromaDB)")
        except Exception as e:
            print(f"⚠️  Vector Store clear failed: {e}")

        # 3. Clear ALL Redis cache types
        from smartmemory.utils.cache import get_cache
        cache = get_cache()

        cache_types = ['embedding', 'search', 'entity_extraction', 'similarity', 'graph_query']
        total_cleared = 0
        for cache_type in cache_types:
            cleared_count = cache.clear_type(cache_type)
            total_cleared += cleared_count

        # Clear any remaining keys with our prefix
        pattern = f"{cache.prefix}:*"
        remaining_keys = cache.redis.keys(pattern)
        if remaining_keys:
            cache.redis.delete(*remaining_keys)
            total_cleared += len(remaining_keys)

        print(f"✅ Cleared Redis Cache ({total_cleared} keys)")

        # 4. Clear working memory buffer
        if hasattr(self, '_working_buffer'):
            self._working_buffer.clear()
            print("✅ Cleared Working Memory Buffer")

        # 5. Clear canonical memory store
        if hasattr(self, '_canonical_store'):
            self._canonical_store.clear()
            print("✅ Cleared Canonical Memory Store")

        # 6. Clear in-memory caches and mixins
        if hasattr(self, '_cache'):
            self._cache.clear()

        if hasattr(self, '_operation_stats'):
            for key in self._operation_stats:
                self._operation_stats[key] = 0

        print("✅ Cleared In-memory Caches")

        # 7. Clear any remaining memory type stores
        memory_types = ['semantic', 'episodic', 'procedural', 'zettelkasten']
        for memory_type in memory_types:
            if hasattr(self, f'_{memory_type}_store'):
                store = getattr(self, f'_{memory_type}_store')
                if hasattr(store, 'clear'):
                    store.clear()
                    print(f"✅ Cleared {memory_type.title()} Memory Store")

        print("🎉 All memory storage backends cleared successfully!")
        return True

    def ingest(self,
               item,
               context=None,
               adapter_name=None,
               converter_name=None,
               extractor_name=None,
               sync: Optional[bool] = None,
               conversation_context: Optional[Union[ConversationContext, Dict[str, Any]]] = None,
               user_id: Optional[str] = None):
        """Ingest item. If sync is True or mode is local, run full pipeline; otherwise persist and enqueue background processing.
        
        Args:
            item: Content to ingest
            context: Additional context for processing
            adapter_name: Specific adapter to use
            converter_name: Specific converter to use
            extractor_name: Specific extractor to use
            sync: If True, run full pipeline synchronously; if False, enqueue for background processing
            conversation_context: Conversation context for conversational memory
            user_id: User ID for user isolation (sets item.user_id)
            
        Returns:
            Union[Dict[str, Any], Any]: Pipeline result if sync=True, or {"item_id": str, "queued": bool} if sync=False
        """
        # Determine effective mode
        if sync is None:
            try:
                from smartmemory.utils import get_config
                mode = ((get_config('ingestion') or {}).get('mode') or '').lower()
            except Exception:
                mode = ''
            sync = (mode == 'local')
        
        # Normalize item and set user_id if provided
        from smartmemory.models.memory_item import MemoryItem
        if not isinstance(item, MemoryItem):
            item = self._crud.normalize_item(item)
        if user_id is not None:
            item.user_id = user_id

        if sync:
            if self._ingestion_flow is None:
                self._ingestion_flow = MemoryIngestionFlow(
                    self,
                    linking=self._linking,
                    enrichment=self._enrichment
                )
            # Merge conversation context into pipeline context if provided
            if conversation_context is not None:
                try:
                    if context is None:
                        context = {}
                    if isinstance(conversation_context, ConversationContext):
                        context['conversation'] = conversation_context.to_dict()
                    elif isinstance(conversation_context, dict):
                        context['conversation'] = dict(conversation_context)
                except Exception as e:
                    logger.debug(f"Failed to merge conversation_context into context: {e}")

            result = self._ingestion_flow.run(
                item,
                context=context,
                adapter_name=adapter_name,
                converter_name=converter_name,
                extractor_name=extractor_name,
                enricher_names=getattr(self, '_enricher_pipeline', []),
            )
            self._evolution.run_evolution_cycle()
            return result

        # Async path: quick persist, emit message, do not run local background
        normalized_item = self._crud.normalize_item(item)
        add_result = self._crud.add(normalized_item)
        if isinstance(add_result, dict):
            item_id = add_result.get('memory_node_id')
        else:
            item_id = add_result
        try:
            from smartmemory.observability.events import RedisStreamQueue
            q = RedisStreamQueue.for_enrich()
            q.enqueue({'job_type': 'enrich', 'item_id': item_id})
            queued = True
        except Exception:
            queued = False
        return {'item_id': item_id, 'queued': queued}

    # Store management
    def add(self,
            item,
            context=None,
            adapter_name=None,
            converter_name=None,
            extractor_name=None,
            enricher_names=None,
            conversation_context: Optional[Union[ConversationContext, Dict[str, Any]]] = None,
            user_id: Optional[str] = None,
            **kwargs) -> str:
        """
        Primary entry point for adding items to SmartMemory.
        Runs the full canonical agentic memory ingestion flow with enrichment, linking, and semantic extraction.
        For basic storage without the full pipeline, use _add_basic() internally.
        
        Args:
            item: Content to add (str, dict, or MemoryItem)
            context: Additional context for processing
            adapter_name: Specific adapter to use
            converter_name: Specific converter to use
            extractor_name: Specific extractor to use
            enricher_names: Specific enrichers to use
            conversation_context: Conversation context for conversational memory
            user_id: User ID for user isolation (sets normalized_item.user_id)
            **kwargs: Additional processing options
            
        Returns:
            str: The item_id of the stored memory item
        """
        # Normalize item using CRUD component (eliminates mixed abstraction)
        normalized_item = self._crud.normalize_item(item)
        
        # Set user_id if provided (backward compatible)
        if user_id is not None:
            normalized_item.user_id = user_id

        # Annotate with conversation metadata if provided (non-invasive)
        try:
            if conversation_context is not None:
                convo_dict: Dict[str, Any]
                if isinstance(conversation_context, ConversationContext):
                    convo_dict = conversation_context.to_dict()
                elif isinstance(conversation_context, dict):
                    convo_dict = dict(conversation_context)
                else:
                    convo_dict = {}
                # MemoryItem guarantees a metadata dict; normalize just in case
                if not isinstance(normalized_item.metadata, dict):
                    logger.warning("MemoryItem.metadata was not a dict; normalizing to empty dict")
                    normalized_item.metadata = {}
                # Only store lightweight identifiers to avoid bloat
                conv_id = convo_dict.get('conversation_id')
                if conv_id is not None:
                    normalized_item.metadata.setdefault('conversation_id', conv_id)
                    normalized_item.metadata.setdefault('provenance', {})
                    try:
                        if isinstance(normalized_item.metadata['provenance'], dict):
                            normalized_item.metadata['provenance'].setdefault('source', 'conversation')
                    except Exception as e:
                        logger.debug(f"Failed to set conversation provenance on metadata: {e}")
        except Exception as e:
            logger.warning(f"Failed to annotate item with conversation metadata: {e}")

        # Special handling for working memory - bypass ingestion pipeline
        memory_type = getattr(normalized_item, 'memory_type', 'semantic')
        if memory_type == 'working':
            return self._add_basic(normalized_item, **kwargs)

        # If this is called from internal code that needs basic storage, check for bypass flag
        if kwargs.pop('_bypass_ingestion', False):
            return self._add_basic(normalized_item, **kwargs)

        # Run full ingestion pipeline for public API
        pipeline = enricher_names if enricher_names is not None else getattr(self, '_enricher_pipeline', [])
        if self._ingestion_flow is None:
            self._ingestion_flow = MemoryIngestionFlow(
                self,
                linking=self._linking,
                enrichment=self._enrichment
            )
        # Merge conversation context into pipeline context if provided
        if conversation_context is not None:
            try:
                if context is None:
                    context = {}
                if isinstance(conversation_context, ConversationContext):
                    context['conversation'] = conversation_context.to_dict()
                elif isinstance(conversation_context, dict):
                    context['conversation'] = dict(conversation_context)
            except Exception as e:
                logger.debug(f"Failed to merge conversation_context into context: {e}")

        result = self._ingestion_flow.run(
            normalized_item,
            context=context,
            adapter_name=adapter_name,
            converter_name=converter_name,
            extractor_name=extractor_name,
            enricher_names=pipeline,
        )

        # Delegate evolution to EvolutionOrchestrator (fail fast)
        self._evolution.run_evolution_cycle()

        # Return the item_id from the result
        if isinstance(result, dict) and 'item' in result:
            item = result['item']
            return item.item_id if hasattr(item, 'item_id') else str(item)
        elif isinstance(result, dict) and 'item_id' in result:
            return result['item_id']
        elif hasattr(result, 'item_id'):
            return result.item_id
        else:
            return str(result) if result else None

    def _add_basic(self, item, **kwargs) -> str:
        """
        Internal method for basic storage without the full ingestion pipeline.
        Used by ingestion flow and evolution algorithms to avoid recursion.
        """
        # Convert to MemoryItem if needed
        if hasattr(item, 'to_memory_item'):
            item = item.to_memory_item()
        elif not isinstance(item, MemoryItem):
            # Convert dict or other types to MemoryItem
            if isinstance(item, dict):
                item = MemoryItem(**item)
            else:
                item = MemoryItem(content=str(item))

        # Route to appropriate memory store based on memory_type
        memory_type = getattr(item, 'memory_type', 'semantic')
        if memory_type == 'working':
            # Determine persistence behavior from config (default: persist working memory)
            persist_enabled = False
            try:
                from smartmemory.utils import get_config
                wm_cfg = (get_config('working_memory') or {})
                persist_enabled = bool(wm_cfg.get('persist', True))
            except Exception as e:
                logger.debug(f"Failed to read working_memory.persist from config; defaulting to True. Error: {e}")
                persist_enabled = True

            if persist_enabled:
                # Persist via CRUD/graph only (no vectorization by default)
                try:
                    # Ensure metadata contains memory_type and optional provenance
                    if not isinstance(item.metadata, dict):
                        logger.warning("MemoryItem.metadata was not a dict for working item; normalizing to empty dict")
                        item.metadata = {}
                    item.metadata.setdefault('memory_type', 'working')
                    item.metadata.setdefault('provenance', {})
                    if isinstance(item.metadata.get('provenance'), dict):
                        item.metadata['provenance'].setdefault('source', 'working_memory')
                except Exception as e:
                    logger.warning(f"Failed to set default working memory metadata: {e}")
                return self._crud.add(item, **kwargs)
            else:
                # Fallback: in-memory buffer with NO hard max length
                if not hasattr(self, '_working_buffer'):
                    from collections import deque
                    self._working_buffer = deque()  # unbounded
                self._working_buffer.append(item)
                return item.item_id
        else:
            # For other memory types, use the standard CRUD
            return self._crud.add(item, **kwargs)

    def _add_impl(self, item, **kwargs) -> str:
        return self._crud.add(item, **kwargs)

    def _get_impl(self, key: str):
        return self._crud.get(key)

    def _update_impl(self, item, **kwargs):
        """Implement type-specific update logic by delegating to CRUD component."""
        return self._crud.update(item, **kwargs)

    def get(self, item_id: str, **kwargs):
        return self._crud.get(item_id)

    # Archive facades (provider hidden behind SmartMemory interface)
    def archive_put(self, conversation_id: str, payload: Union[bytes, Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, str]:
        """Persist a raw conversation artifact durably and return { archive_uri, content_hash }.

        Delegates to the configured ArchiveProvider. Errors are logged and re-raised; callers should handle.
        """
        try:
            if metadata is None:
                metadata = {}
            result = get_archive_provider().put(conversation_id, payload, metadata)
            if not isinstance(result, dict) or 'archive_uri' not in result or 'content_hash' not in result:
                raise RuntimeError("ArchiveProvider.put returned invalid result; expected keys 'archive_uri' and 'content_hash'")
            return result
        except Exception as e:
            logger.error(f"archive_put failed for conversation_id={conversation_id}: {e}")
            raise

    def archive_get(self, archive_uri: str) -> Union[bytes, Dict[str, Any]]:
        """Retrieve an archived artifact by its URI via the configured ArchiveProvider."""
        try:
            return get_archive_provider().get(archive_uri)
        except Exception as e:
            logger.error(f"archive_get failed for uri={archive_uri}: {e}")
            raise

    # Public wrappers for internal operations to preserve encapsulation
    def add_basic(self, item, **kwargs) -> str:
        """Public wrapper for basic add (dual-node aware)."""
        return self._add_basic(item, **kwargs)

    def update_properties(self, item_id: str, properties: dict, write_mode: str | None = None):
        """Public wrapper to update memory node properties with merge/replace semantics."""
        return self._crud.update_memory_node(item_id, properties, write_mode)

    # Backward-compatible alias (deprecated)
    def update(self, item: Union[MemoryItem, dict], **kwargs):
        """Can take either dict or MemoryItem."""
        return self._crud.update(item)

    def run_evolution_cycle(self):
        """Public wrapper to run a single evolution cycle."""
        return self._evolution.run_evolution_cycle()

    def ground_context(self, context: dict):
        """Public wrapper to ground using a pre-built context dict."""
        return self._grounding.ground(context)

    def add_edge(self, source_id: str, target_id: str, relation_type: str, properties: dict | None = None):
        """Public wrapper to add an edge between two nodes."""
        properties = properties or {}
        return self._graph.add_edge(source_id=source_id, target_id=target_id, edge_type=relation_type, properties=properties)

    def create_or_merge_node(self, item_id: str, properties: dict, memory_type: str | None = None):
        """Public wrapper to upsert a raw node with properties. Returns item_id."""
        self._graph.add_node(item_id=item_id, properties=properties, memory_type=memory_type)
        return item_id

    def delete(self, item_id: str, **kwargs) -> bool:
        return self._crud.delete(item_id)

    def resolve_external(self, node: MemoryItem) -> Optional[list]:
        """Delegate resolve_external to ExternalResolver submodule."""
        return self._external_resolver.resolve_external(node)

    def search(self, query: str, top_k: int = 5, memory_type: str = None, user_id: str = None, conversation_context: Optional[Union[ConversationContext, Dict[str, Any]]] = None):
        """Search using canonical search component."""
        if memory_type == 'working':
            # Check if persistence is enabled; if so, use canonical search on persisted working items
            persist_enabled = False
            try:
                from smartmemory.utils import get_config
                wm_cfg = (get_config('working_memory') or {})
                persist_enabled = bool(wm_cfg.get('persist', True))
            except Exception:
                persist_enabled = True

            if persist_enabled:
                # Use canonical search for working memory stored in graph
                results = self._search.search(query, top_k=top_k * 2, memory_type='working')
                # Optional: filter by conversation_id when provided
                conv_id = None
                try:
                    if isinstance(conversation_context, ConversationContext):
                        conv_id = conversation_context.conversation_id
                    elif isinstance(conversation_context, dict):
                        conv_id = conversation_context.get('conversation_id')
                except Exception as e:
                    logger.debug(f"Failed to extract conversation_id from conversation_context: {e}")
                    conv_id = None
                if conv_id and results:
                    results = [r for r in results if getattr(r, 'metadata', {}).get('conversation_id') == conv_id]
                # Apply user_id filtering consistent with non-working path
                if user_id and results:
                    filtered_results = []
                    for item in results:
                        item_metadata = getattr(item, 'metadata', {})
                        if item_metadata.get('user_id') == user_id:
                            filtered_results.append(item)
                    results = filtered_results[:top_k] if filtered_results else []
                elif user_id:
                    results = []
                return results[:top_k]
            else:
                # Fallback to in-memory buffer behavior
                if hasattr(self, '_working_buffer') and self._working_buffer:
                    results = []
                    # If a conversation context is provided, bias towards that conversation_id
                    conv_id = None
                    try:
                        if isinstance(conversation_context, ConversationContext):
                            conv_id = conversation_context.conversation_id
                        elif isinstance(conversation_context, dict):
                            conv_id = conversation_context.get('conversation_id')
                    except Exception as e:
                        logger.debug(f"Failed to extract conversation_id from conversation_context (buffer path): {e}")
                        conv_id = None

                    for item in self._working_buffer:
                        # Prefer items from the same conversation when available
                        if conv_id and hasattr(item, 'metadata') and isinstance(item.metadata, dict):
                            if item.metadata.get('conversation_id') == conv_id:
                                if query == "*" or query == "" or len(query.strip()) == 0:
                                    results.append(item)
                                    continue
                                if query.lower() in str(item.content).lower():
                                    results.append(item)
                                    continue
                                if hasattr(item, 'metadata') and item.metadata:
                                    metadata_str = str(item.metadata).lower()
                                    if query.lower() in metadata_str:
                                        results.append(item)
                                        continue
                        if query == "*" or query == "" or len(query.strip()) == 0:
                            results.append(item)
                        elif query.lower() in str(item.content).lower():
                            results.append(item)
                        elif hasattr(item, 'metadata') and item.metadata:
                            metadata_str = str(item.metadata).lower()
                            if query.lower() in metadata_str:
                                results.append(item)
                    if not results and self._working_buffer:
                        results = list(self._working_buffer)[-top_k:]
                    return results[:top_k]
                return []

        # Use canonical search component directly
        results = self._search.search(query, top_k=top_k * 2, memory_type=memory_type)  # Overfetch for filtering

        # Apply user_id filtering if provided
        if user_id and results:
            filtered_results = []
            for item in results:
                item_metadata = getattr(item, 'metadata', {})
                item_user_id = getattr(item, 'user_id', None)
                # Check both user_id attribute and metadata for user_id
                # Also restore user_id attribute if it's missing but exists in metadata
                if not item_user_id and item_metadata.get('user_id'):
                    item.user_id = item_metadata.get('user_id')
                    item_user_id = item.user_id

                if item_user_id == user_id or item_metadata.get('user_id') == user_id:
                    filtered_results.append(item)
                # Skip memories with different user_ids or no user_id (legacy items)
            results = filtered_results[:top_k] if filtered_results else []
        elif user_id:
            # If user_id is provided but no results match, return empty instead of all results
            results = []

        return results[:top_k]

    # Linking
    def link(self, source_id: str, target_id: str, link_type: Union[str, "LinkType"] = "RELATED") -> str:
        """
        Link two memory items. (Delegates to internal Linking helper.)
        Always use this method instead of accessing Linking directly.
        """
        # Accept both Enum and string for link_type
        if hasattr(link_type, 'value'):
            link_type = link_type.value
        return self._linking.link(source_id, target_id, link_type)

    def get_links(self, item_id: str, memory_type: str = "semantic") -> List[str]:
        """
        Get all links (triples) for a memory item. (Delegates to internal Linking helper.)
        Always use this method instead of accessing Linking directly.
        """
        return self._linking.get_links(item_id, memory_type)

    def get_neighbors(self, item_id: str):
        """
        Return neighboring MemoryItems for a node.
        Delegates to GraphOperations component for proper abstraction.
        """
        neighbors = self._graph_ops.get_neighbors(item_id)
        neighbor_items = []
        for neighbor_id in neighbors:
            neighbor_item = self.get(neighbor_id)
            if neighbor_item:
                neighbor_items.append(neighbor_item)
        return neighbor_items

    # Enrichment & Transformation
    def enrich(self, item_id: str, routines: Optional[List[str]] = None) -> None:
        """
        Enrich a memory item using registered enrichment routines.
        Args:
            item_id: ID of the memory item to enrich.
            routines: Optional list of enrichment routines to apply.
        """
        return self._enrichment.enrich(item_id, routines)

    # Grounding & Provenance
    def ground(self, item_id: str, source_url: str, validation: Optional[dict] = None) -> None:
        """
        Ground a memory item to an external source (e.g., for provenance).
        Args:
            item_id: ID of the memory item to ground.
            source_url: URL of the external source.
            validation: Optional validation metadata.
        """
        context = {
            "item_id": item_id,
            "source_url": source_url,
            "validation": validation
        }
        return self._grounding.ground(context)

    # Personalization & Feedback
    def personalize(self, user_id: str, traits: dict = None, preferences: dict = None) -> None:
        return self._personalization.personalize(user_id, traits, preferences)

    def update_from_feedback(self, feedback: dict, memory_type: str = "semantic") -> None:
        return self._personalization.update_from_feedback(feedback, memory_type)

    # Existing summary/monitoring methods can be refactored to use store_manager or moved to a new analytics module if needed.
    def summary(self) -> dict:
        """Delegate summary to Monitoring submodule."""
        return self._monitoring.summary()

    def orphaned_notes(self) -> list:
        """Delegate orphaned_notes to Monitoring submodule."""
        return self._monitoring.orphaned_notes()

    def prune(self, strategy="old", days=365, **kwargs):
        """Delegate prune to Monitoring submodule."""
        return self._monitoring.prune(strategy, days, **kwargs)

    def find_old_notes(self, days: int = 365) -> list:
        """Delegate find_old_notes to Monitoring submodule."""
        return self._monitoring.find_old_notes(days)

    def self_monitor(self) -> dict:
        """Delegate self_monitor to Monitoring submodule."""
        return self._monitoring.self_monitor()

    def reflect(self, top_k: int = 5) -> dict:
        """Delegate reflect to Monitoring submodule."""
        return self._monitoring.reflect(top_k)

    def summarize(self, max_items: int = 10) -> dict:
        """Delegate summarize to Monitoring submodule."""
        return self._monitoring.summarize(max_items)

    def embeddings_search(self, embedding, top_k=5):
        """
        Search for memory items most similar to the given embedding using the vector store.
        Delegates to Search component for proper abstraction.
        """
        return self._search.embeddings_search(embedding, top_k=top_k)

    def add_tags(self, item_id: str, tags: list) -> bool:
        """Delegate add_tags to CRUD submodule."""
        return self._crud.add_tags(item_id, tags)

    # Evolution operations - delegate to EvolutionOrchestrator
    def commit_working_to_episodic(self, remove_from_source: bool = True) -> List[str]:
        """Delegate evolution to EvolutionOrchestrator component."""
        return self._evolution.commit_working_to_episodic(remove_from_source)

    def commit_working_to_procedural(self, remove_from_source: bool = True) -> List[str]:
        """Delegate evolution to EvolutionOrchestrator component."""
        return self._evolution.commit_working_to_procedural(remove_from_source)

    # Debug and troubleshooting methods
    def debug_search(self, query: str, top_k: int = 5) -> dict:
        """Debug search functionality with detailed logging."""
        debug_info = {
            'query': query,
            'top_k': top_k,
            'graph_backend': str(type(self._graph.backend)),
            'search_component': str(type(self._search)),
            'results': []
        }

        # Test direct graph search
        graph_results = self._graph.search(query, top_k=top_k)
        debug_info['graph_search_count'] = len(graph_results)
        debug_info['graph_search_results'] = [
            {
                'item_id': getattr(r, 'item_id', 'No ID'),
                'content_preview': str(getattr(r, 'content', 'No content'))[:50] + '...',
                'type': str(type(r))
            } for r in graph_results[:3]
        ]

        # Test search component
        search_results = self._search.search(query, top_k=top_k)
        debug_info['search_component_count'] = len(search_results)
        debug_info['search_component_results'] = [
            {
                'item_id': getattr(r, 'item_id', 'No ID'),
                'content_preview': str(getattr(r, 'content', 'No content'))[:50] + '...',
                'type': str(type(r))
            } for r in search_results[:3]
        ]

        debug_info['results'] = search_results
        return debug_info

    def get_all_items_debug(self, tenant_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """Get all items for debugging, optionally filtered by tenant/user.
        
        Args:
            tenant_id: Filter by tenant/workspace ID
            user_id: Filter by user ID
        """
        debug_info = {
            'total_items': 0,
            'items_by_type': {},
            'sample_items': []
        }
        
        # Get all node IDs - self._graph.nodes is SmartGraphNodes object with nodes() method
        if hasattr(self._graph, 'nodes'):
            all_node_ids = self._graph.nodes.nodes()  # Call nodes() method on nodes property
            debug_info['total_node_ids'] = len(all_node_ids)
            
            # Filter and collect items
            seen_ids = set()
            for node_id in all_node_ids:
                item = self._graph.get_node(node_id)
                if not item:
                    continue
                
                # Get item_id for deduplication
                item_id = getattr(item, 'item_id', node_id)
                if item_id in seen_ids:
                    continue
                
                # Filter by tenant/user if specified
                if tenant_id or user_id:
                    item_metadata = getattr(item, 'metadata', {})
                    item_workspace = getattr(item, 'workspace_id', None) or item_metadata.get('workspace_id')
                    item_user = getattr(item, 'user_id', None) or item_metadata.get('user_id')
                    
                    # Skip if doesn't match filter
                    if tenant_id and item_workspace != tenant_id:
                        continue
                    if user_id and item_user != user_id:
                        continue
                
                seen_ids.add(item_id)
                debug_info['total_items'] += 1
                item_type = getattr(item, 'memory_type', 'unknown')
                debug_info['items_by_type'][item_type] = debug_info['items_by_type'].get(item_type, 0) + 1

                if len(debug_info['sample_items']) < 3:
                    debug_info['sample_items'].append({
                        'item_id': item_id,
                        'content_preview': str(getattr(item, 'content', 'No content'))[:50] + '...',
                        'memory_type': item_type,
                        'type': str(type(item))
                    })

        return debug_info

    def fix_search_if_broken(self) -> dict:
        """Attempt to fix search functionality if it's broken."""
        fix_info = {
            'fixes_applied': [],
            'test_search_count': 0
        }

        # Reinitialize search component
        self._search = Search(self._graph)
        fix_info['fixes_applied'].append('Reinitialized search component')

        # Clear any caches
        if hasattr(self._graph, 'clear_cache'):
            self._graph.clear_cache()
            fix_info['fixes_applied'].append('Cleared graph cache')

        # Test search after fixes
        test_results = self._search.search("test", top_k=1)
        fix_info['test_search_count'] = len(test_results)

        return fix_info
    
    # Temporal query methods
    
    def time_travel(self, to: str):
        """
        Time travel context manager.
        
        All queries executed within this context will use the specified
        time point as their reference.
        
        Args:
            to: Time point to travel to (ISO format: "2024-09-01" or "2024-09-01T00:00:00")
            
        Returns:
            Context manager
            
        Example:
            with memory.time_travel("2024-09-01"):
                # All queries in this block use Sept 1st as reference
                results = memory.search("Python")
                user = memory.get("user123")
        """
        from smartmemory.temporal.context import time_travel
        return time_travel(self, to)
    
    @property
    def graph(self):
        """Access to underlying graph for temporal queries."""
        return self._graph
