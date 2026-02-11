import logging
from time import perf_counter
from typing import Optional, Union, Any, Dict, List

from smartmemory.conversation.context import ConversationContext
from smartmemory.memory.pipeline.config import PipelineConfigBundle
from smartmemory.observability.instrumentation import make_emitter
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
from smartmemory.clustering.global_cluster import GlobalClustering
from smartmemory.temporal.queries import TemporalQueries
from smartmemory.temporal.version_tracker import VersionTracker
from smartmemory.models.memory_item import MemoryItem
from smartmemory.plugins.resolvers.external_resolver import ExternalResolver
from smartmemory.interfaces import ScopeProvider
from smartmemory.managers.debug import DebugManager
from smartmemory.managers.enrichment import EnrichmentManager
from smartmemory.managers.evolution import EvolutionManager
from smartmemory.managers.monitoring import MonitoringManager

logger = logging.getLogger(__name__)


class SmartMemory(MemoryBase):
    """
    Unified agentic memory store combining semantic, episodic, procedural, and working memory.
    Delegates responsibilities to submodules for store management, linking, enrichment, grounding, and personalization.

    API:
        ingest(item) - Full agentic pipeline: extract → store → link → enrich → ground → evolve.
                       Use for user-facing ingestion with all processing stages.
        add(item)    - Simple storage: normalize → store → embed.
                       Use for internal operations, derived items, and when pipeline is not needed.

    All linking operations should be accessed via SmartMemory methods only.
    Do not use Linking directly; it is an internal implementation detail.
    """

    # Preconfigured emitter for memory operations
    _MEM_EMIT = make_emitter(component="smart_memory", default_type="memory_operation")

    def __init__(
        self,
        scope_provider: Optional[ScopeProvider] = None,
        *args,
        # Optional dependency injection (all default to None = create defaults)
        graph: Optional["SmartGraph"] = None,
        crud: Optional["CRUD"] = None,
        search: Optional["Search"] = None,
        linking: Optional["Linking"] = None,
        enrichment: Optional["Enrichment"] = None,
        grounding: Optional["Grounding"] = None,
        personalization: Optional["Personalization"] = None,
        monitoring: Optional["Monitoring"] = None,
        evolution: Optional["EvolutionOrchestrator"] = None,
        clustering: Optional["GlobalClustering"] = None,
        external_resolver: Optional["ExternalResolver"] = None,
        version_tracker: Optional["VersionTracker"] = None,
        temporal: Optional["TemporalQueries"] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Use DefaultScopeProvider if none provided (for OSS usage)
        # Service layer should always provide a secure ScopeProvider
        if scope_provider is None:
            # Lazy import to avoid circular dependency
            from smartmemory.scope_provider import DefaultScopeProvider

            scope_provider = DefaultScopeProvider()

        self.scope_provider = scope_provider  # Store for use in filtering methods
        self._graph = graph or SmartGraph(scope_provider=scope_provider)

        # Initialize stages with proper delegation (use injected or create defaults)
        self._graph_ops = GraphOperations(self._graph)
        self._crud = crud or CRUD(self._graph)
        self._linking = linking or Linking(self._graph)
        self._enrichment = enrichment or Enrichment(self._graph)
        self._grounding = grounding or Grounding(self._graph)
        self._personalization = personalization or Personalization(self._graph)
        self._search = search or Search(self._graph)
        self._monitoring = monitoring or Monitoring(self._graph)
        self._evolution = evolution or EvolutionOrchestrator(self)
        self._clustering = clustering or GlobalClustering(self)
        self._external_resolver = external_resolver or ExternalResolver()

        # Initialize temporal queries
        self.version_tracker = version_tracker or VersionTracker(self._graph)
        self.temporal = temporal or TemporalQueries(self)
        self._temporal_context = None  # For time-travel context manager

        # Defer flow construction until needed (lazy)
        self._ingestion_flow = None
        self._pipeline_runner = None
        self._last_token_summary: dict | None = None

        # Set self on graph search module for SSG traversal
        self._graph.search.set_smart_memory(self)

        # Initialize sub-managers
        self._monitoring_mgr = MonitoringManager(self._monitoring)
        self._evolution_mgr = EvolutionManager(self._evolution, self._clustering)
        self._debug_mgr = DebugManager(self._graph, self._search, self.scope_provider)
        self._enrichment_mgr = EnrichmentManager(self._enrichment, self._grounding, self._external_resolver)

    @property
    def graph(self):
        """Access to underlying graph storage."""
        return self._graph

    # =========================================================================
    # Pipeline v2 assembly
    # =========================================================================

    def _create_pipeline_runner(self):
        """Assemble the v2 pipeline runner from stage wrappers."""
        from smartmemory.pipeline.runner import PipelineRunner
        from smartmemory.pipeline.transport import InProcessTransport
        from smartmemory.pipeline.stages.classify import ClassifyStage
        from smartmemory.pipeline.stages.coreference import CoreferenceStageCommand
        from smartmemory.pipeline.stages.simplify import SimplifyStage
        from smartmemory.pipeline.stages.entity_ruler import EntityRulerStage
        from smartmemory.pipeline.stages.llm_extract import LLMExtractStage
        from smartmemory.pipeline.stages.ontology_constrain import OntologyConstrainStage
        from smartmemory.pipeline.stages.store import StoreStage
        from smartmemory.pipeline.stages.link import LinkStage
        from smartmemory.pipeline.stages.enrich import EnrichStage
        from smartmemory.pipeline.stages.ground import GroundStage
        from smartmemory.pipeline.stages.evolve import EvolveStage
        from smartmemory.memory.ingestion.enrichment import EnrichmentPipeline
        from smartmemory.memory.ingestion.observer import IngestionObserver
        from smartmemory.graph.ontology_graph import OntologyGraph

        # Reuse the existing ingestion flow for classify (it has the logic)
        if self._ingestion_flow is None:
            self._ingestion_flow = MemoryIngestionFlow(self, linking=self._linking, enrichment=self._enrichment)

        # Build enrichment pipeline
        observer = IngestionObserver()
        enrichment_pipeline = EnrichmentPipeline(self, self._enrichment, observer)

        # Build ontology graph for constraint stage
        workspace_id = "default"
        try:
            scope = self.scope_provider.get_scope()
            workspace_id = getattr(scope, "workspace_id", None) or "default"
        except Exception:
            pass
        ontology_graph = OntologyGraph(workspace_id=workspace_id)

        # Phase 4: PatternManager for learned entity pattern dictionary scan
        from smartmemory.ontology.pattern_manager import PatternManager

        pattern_manager = PatternManager(ontology_graph, workspace_id=workspace_id)

        # Phase 4: EntityPairCache for relation caching
        from smartmemory.ontology.entity_pair_cache import EntityPairCache

        entity_pair_cache = EntityPairCache()

        # Phase 4: Promotion queue (optional — requires Redis)
        promotion_queue = None
        try:
            from smartmemory.observability.events import RedisStreamQueue

            promotion_queue = RedisStreamQueue.for_promote()
        except Exception:
            pass

        stages = [
            ClassifyStage(self._ingestion_flow),
            CoreferenceStageCommand(),
            SimplifyStage(),
            EntityRulerStage(pattern_manager=pattern_manager),
            LLMExtractStage(),
            OntologyConstrainStage(
                ontology_graph, promotion_queue=promotion_queue, entity_pair_cache=entity_pair_cache
            ),
            StoreStage(self),
            LinkStage(self._linking),
            EnrichStage(enrichment_pipeline),
            GroundStage(self),
            EvolveStage(self),
        ]

        from smartmemory.pipeline.metrics import PipelineMetricsEmitter

        emitter = PipelineMetricsEmitter(workspace_id=workspace_id)
        return PipelineRunner(stages, InProcessTransport(), metrics_emitter=emitter)

    @property
    def last_token_summary(self) -> dict | None:
        """Token usage summary from the most recent ``ingest()`` call, or None."""
        return self._last_token_summary

    def create_pipeline_runner(self):
        """Create a configured PipelineRunner for breakpoint execution.

        Public API for Studio and other consumers that need run_to()/run_from().
        Returns cached runner instance (stateless — state flows through PipelineState).
        """
        if self._pipeline_runner is None:
            self._pipeline_runner = self._create_pipeline_runner()
        return self._pipeline_runner

    def _build_pipeline_config(
        self,
        extractor_name=None,
        enricher_names=None,
        pipeline_config=None,
        sync=True,
    ):
        """Map legacy ingest() parameters to a v2 PipelineConfig."""
        from smartmemory.pipeline.config import PipelineConfig

        config = PipelineConfig()

        if pipeline_config is not None:
            # Merge legacy PipelineConfigBundle fields into v2 config
            if pipeline_config.extraction and pipeline_config.extraction.extractor_name:
                config.extraction.extractor_name = pipeline_config.extraction.extractor_name
            if pipeline_config.extraction and pipeline_config.extraction.model:
                config.extraction.llm_extract.model = pipeline_config.extraction.model

        if extractor_name:
            config.extraction.extractor_name = extractor_name
        if enricher_names:
            config.enrich.enricher_names = list(enricher_names)
        if not sync:
            config.mode = "async"

        # Pull workspace from scope provider
        try:
            scope = self.scope_provider.get_scope()
            config.workspace_id = getattr(scope, "workspace_id", None)
        except Exception:
            pass

        return config

    # =========================================================================
    # Core operations (ingest, add, get, search, update, delete, link)
    # These stay inline as the hot path.
    # =========================================================================

    def ingest(
        self,
        item,
        context=None,
        adapter_name=None,
        converter_name=None,
        extractor_name=None,
        enricher_names=None,
        conversation_context: Optional[Union[ConversationContext, Dict[str, Any]]] = None,
        pipeline_config: Optional[PipelineConfigBundle] = None,
        sync: Optional[bool] = None,
        auto_challenge: Optional[bool] = None,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Primary entry point for ingesting items into SmartMemory.
        Runs the full canonical agentic memory ingestion flow with extraction, linking, enrichment, and evolution.
        For basic storage without the full pipeline, use add().

        Args:
            item: Content to ingest (str, dict, or MemoryItem)
            context: Additional context for processing
            adapter_name: Specific adapter to use
            converter_name: Specific converter to use
            extractor_name: Specific extractor to use
            enricher_names: Specific enrichers to use
            conversation_context: Conversation context for conversational memory
            sync: If True, run full pipeline synchronously; if False, persist and queue for background processing.
                  If None, defaults to sync=True (local mode).
            auto_challenge: If True, automatically challenge factual claims before ingesting.
                           If None, uses smart triggering (only challenges semantic facts with factual patterns).
            **kwargs: Additional processing options

        Returns:
            str: The item_id of the stored memory item (sync=True)
            dict: {"item_id": str, "queued": bool} (sync=False)
        """
        # Determine effective mode
        if sync is None:
            try:
                from smartmemory.utils import get_config

                mode = ((get_config("ingestion") or {}).get("mode") or "").lower()
                sync = mode != "async"  # Default to sync unless explicitly async
            except Exception:
                sync = True  # Default to sync

        # Async path: quick persist + queue for background processing
        if not sync:
            normalized_item = self._crud.normalize_item(item)
            add_result = self._crud.add(normalized_item)
            if isinstance(add_result, dict):
                item_id = add_result.get("memory_node_id")
            else:
                item_id = add_result
            try:
                from smartmemory.observability.events import RedisStreamQueue

                q = RedisStreamQueue.for_enrich()
                q.enqueue({"job_type": "enrich", "item_id": item_id})
                queued = True
            except Exception:
                queued = False
            return {"item_id": item_id, "queued": queued}
        # Normalize item using CRUD component (eliminates mixed abstraction)
        normalized_item = self._crud.normalize_item(item)

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
                conv_id = convo_dict.get("conversation_id")
                if conv_id is not None:
                    normalized_item.metadata.setdefault("conversation_id", conv_id)
                    normalized_item.metadata.setdefault("provenance", {})
                    try:
                        if isinstance(normalized_item.metadata["provenance"], dict):
                            normalized_item.metadata["provenance"].setdefault("source", "conversation")
                    except Exception as e:
                        logger.debug(f"Failed to set conversation provenance on metadata: {e}")
        except Exception as e:
            logger.warning(f"Failed to annotate item with conversation metadata: {e}")

        # Special handling for working memory - bypass ingestion pipeline
        memory_type = getattr(normalized_item, "memory_type", "semantic")
        if memory_type == "working":
            return self.add(normalized_item, **kwargs)

        # Auto-challenge: detect contradictions before ingesting
        # Only runs when: auto_challenge=True, or auto_challenge=None and smart triggering says yes
        try:
            from smartmemory.reasoning.challenger import should_challenge, AssertionChallenger

            content = normalized_item.content or ""
            metadata = normalized_item.metadata or {}
            source = metadata.get("source") or (context or {}).get("source")

            do_challenge = auto_challenge
            if do_challenge is None:
                # Smart triggering: only challenge semantic facts with factual patterns
                do_challenge = should_challenge(
                    content=content, memory_type=memory_type, source=source, metadata=metadata
                )

            if do_challenge:
                challenger = AssertionChallenger(self, use_llm=False)  # Heuristics for speed
                challenge_result = challenger.challenge(content, memory_type=memory_type)

                if challenge_result.has_conflicts:
                    # Store challenge result in metadata for downstream handling
                    normalized_item.metadata["challenge_result"] = {
                        "has_conflicts": True,
                        "conflict_count": len(challenge_result.conflicts),
                        "overall_confidence": challenge_result.overall_confidence,
                        "conflicts": [c.to_dict() for c in challenge_result.conflicts[:3]],  # Limit stored
                    }
                    logger.info(
                        f"Auto-challenge found {len(challenge_result.conflicts)} conflicts for: {content[:50]}..."
                    )
        except Exception as e:
            logger.debug(f"Auto-challenge failed (non-fatal): {e}")

        # Check for explicit versioning via original_id
        # This supports the temporal test patterns where add() is used to create new versions
        if normalized_item.metadata and "original_id" in normalized_item.metadata:
            original_id = normalized_item.metadata["original_id"]
            try:
                version = self.version_tracker.create_version(
                    item_id=original_id,
                    content=normalized_item.content,
                    metadata=normalized_item.metadata,
                    change_reason="updated via add()",
                )
                # Return a unique ID for the version node, consistent with the expectation that
                # add() returns a unique ID for the new "item" (version)
                return f"version_{original_id}_{version.version_number}"
            except Exception as e:
                logger.warning(f"Failed to create version for {original_id}: {e}")
                # Fallback to normal add if versioning fails (e.g. original not found)

        # If this is called from internal code that needs basic storage, check for bypass flag
        if kwargs.pop("_bypass_ingestion", False):
            return self.add(normalized_item, **kwargs)

        # Run full ingestion pipeline (v2)
        if self._pipeline_runner is None:
            self._pipeline_runner = self._create_pipeline_runner()

        pipeline_cfg = self._build_pipeline_config(
            extractor_name=extractor_name,
            enricher_names=enricher_names if enricher_names is not None else getattr(self, "_enricher_pipeline", []),
            pipeline_config=pipeline_config,
            sync=True,
        )

        # Build metadata dict for the pipeline
        meta = dict(normalized_item.metadata or {})
        if conversation_context is not None:
            try:
                if isinstance(conversation_context, ConversationContext):
                    meta["conversation"] = conversation_context.to_dict()
                elif isinstance(conversation_context, dict):
                    meta["conversation"] = dict(conversation_context)
            except Exception as e:
                logger.debug(f"Failed to merge conversation_context into metadata: {e}")

        state = self._pipeline_runner.run(
            text=normalized_item.content or "",
            config=pipeline_cfg,
            metadata=meta,
        )

        # Save token usage summary for retrieval by service layer (CFS-1)
        if state.token_tracker:
            self._last_token_summary = state.token_tracker.summary()
        else:
            self._last_token_summary = None

        item_id = state.item_id

        # Auto-version initial creation
        if item_id and self.version_tracker:
            try:
                # Only create if no versions exist (avoid double creation if logic changes)
                # We check cache first to avoid DB hit if possible, but get_versions does that
                versions = self.version_tracker.get_versions(item_id)
                if not versions:
                    self.version_tracker.create_version(
                        item_id=item_id,
                        content=normalized_item.content,
                        metadata=normalized_item.metadata,
                        change_reason="initial creation",
                    )
            except Exception as e:
                logger.debug(f"Failed to create initial version for {item_id}: {e}")

        return item_id

    def add(self, item, **kwargs) -> str:
        """
        Simple storage without the full ingestion pipeline.

        Use this for:
        - Internal operations (evolution, enrichment creating derived items)
        - Direct storage when extraction/linking/enrichment is not needed
        - Avoiding recursion in pipeline stages

        For full agentic ingestion with all processing stages, use ingest().

        Args:
            item: Content to store (str, dict, or MemoryItem)
            **kwargs: Additional storage options

        Returns:
            str: The item_id of the stored memory item
        """
        # Convert to MemoryItem if needed
        if hasattr(item, "to_memory_item"):
            item = item.to_memory_item()
        elif not isinstance(item, MemoryItem):
            # Convert dict or other types to MemoryItem
            if isinstance(item, dict):
                item = MemoryItem(**item)
            else:
                item = MemoryItem(content=str(item))

        # Route to appropriate memory store based on memory_type
        memory_type = getattr(item, "memory_type", "semantic")
        if memory_type == "working":
            # Determine persistence behavior from config (default: persist working memory)
            persist_enabled = False
            try:
                from smartmemory.utils import get_config

                wm_cfg = get_config("working_memory") or {}
                persist_enabled = bool(wm_cfg.get("persist", True))
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
                    item.metadata.setdefault("memory_type", "working")
                    item.metadata.setdefault("provenance", {})
                    if isinstance(item.metadata.get("provenance"), dict):
                        item.metadata["provenance"].setdefault("source", "working_memory")
                except Exception as e:
                    logger.warning(f"Failed to set default working memory metadata: {e}")
                return self._crud.add(item, **kwargs)
            else:
                # Fallback: in-memory buffer with NO hard max length
                if not hasattr(self, "_working_buffer"):
                    from collections import deque

                    self._working_buffer = deque()  # unbounded
                self._working_buffer.append(item)
                return item.item_id
        else:
            # For other memory types, use the standard CRUD
            t0 = perf_counter()
            result = self._crud.add(item, **kwargs)
            # Emit memory_store event for observability
            # noinspection PyArgumentList
            SmartMemory._MEM_EMIT(
                "memory_store",
                "add",
                {
                    "item_id": result,
                    "memory_type": memory_type,
                    "content_length": len(getattr(item, "content", "") or ""),
                    "duration_ms": (perf_counter() - t0) * 1000.0,
                },
            )
            return result

    def _add_impl(self, item, **kwargs) -> str:
        """Implement abstract method from MemoryBase."""
        return self._crud.add(item, **kwargs)

    def _get_impl(self, key: str):
        return self._crud.get(key)

    def _update_impl(self, item, **kwargs):
        """Implement type-specific update logic by delegating to CRUD component."""
        return self._crud.update(item, **kwargs)

    def get(self, item_id: str, **kwargs):
        t0 = perf_counter()
        result = self._crud.get(item_id)
        # Emit memory_retrieve event for observability
        # noinspection PyArgumentList
        SmartMemory._MEM_EMIT(
            "memory_retrieve",
            "get",
            {"item_id": item_id, "found": result is not None, "duration_ms": (perf_counter() - t0) * 1000.0},
        )
        return result

    # Archive facades (provider hidden behind SmartMemory interface)
    def archive_put(
        self, conversation_id: str, payload: Union[bytes, Dict[str, Any]], metadata: Dict[str, Any]
    ) -> Dict[str, str]:
        """Persist a raw conversation artifact durably and return { archive_uri, content_hash }."""
        try:
            if metadata is None:
                metadata = {}
            result = get_archive_provider().put(conversation_id, payload, metadata)
            if not isinstance(result, dict) or "archive_uri" not in result or "content_hash" not in result:
                raise RuntimeError(
                    "ArchiveProvider.put returned invalid result; expected keys 'archive_uri' and 'content_hash'"
                )
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

    def find_shortest_path(self, start_id: str, end_id: str, max_hops: int = 5) -> Optional["MemoryPath"]:
        """Find the shortest path between two nodes in the knowledge graph.

        Args:
            start_id: Item ID of the start node.
            end_id: Item ID of the end node.
            max_hops: Maximum number of hops to traverse (capped at 10).

        Returns:
            MemoryPath with populated Triple objects, or None if no path found.
        """
        from smartmemory.graph.core.memory_path import MemoryPath, Triple

        max_hops = min(max_hops, 10)
        query = (
            "MATCH (start), (end) "
            "WHERE start.item_id = $start_id AND end.item_id = $end_id "
            f"MATCH path = shortestPath((start)-[*..{max_hops}]-(end)) "
            "RETURN nodes(path) as nodes, relationships(path) as rels"
        )
        try:
            results = self._graph.execute_query(query, {"start_id": start_id, "end_id": end_id})
        except Exception as e:
            logger.error(f"find_shortest_path failed: {e}")
            return None

        if not results:
            return None

        row = results[0]
        nodes_data = row[0] if isinstance(row, (list, tuple)) else row.get("nodes", [])
        rels_data = row[1] if isinstance(row, (list, tuple)) else row.get("rels", [])

        triples: list[Triple] = []
        for i, rel in enumerate(rels_data):
            src_node = nodes_data[i]
            tgt_node = nodes_data[i + 1]

            src_id = (
                src_node.properties.get("item_id", str(src_node.id))
                if hasattr(src_node, "properties")
                else str(src_node)
            )
            tgt_id = (
                tgt_node.properties.get("item_id", str(tgt_node.id))
                if hasattr(tgt_node, "properties")
                else str(tgt_node)
            )
            rel_type = rel.relation if hasattr(rel, "relation") else str(rel)

            triples.append(Triple(subject=src_id, predicate=rel_type, object=tgt_id))

        return MemoryPath(triples=triples)

    def execute_cypher(self, query: str, params: dict | None = None) -> list:
        """Execute a raw Cypher query against the graph backend.

        Args:
            query: Cypher query string with optional $param placeholders.
            params: Parameter dict for parameterized queries.

        Returns:
            List of result rows from the backend.
        """
        return self._graph.execute_query(query, params)

    def update_properties(self, item_id: str, properties: dict, write_mode: str | None = None):
        """Public wrapper to update memory node properties with merge/replace semantics."""
        return self._crud.update_memory_node(item_id, properties, write_mode)

    # Backward-compatible alias (deprecated)
    def update(self, item: Union[MemoryItem, dict], **kwargs):
        """Can take either dict or MemoryItem."""
        return self._crud.update(item)

    def add_edge(self, source_id: str, target_id: str, relation_type: str, properties: dict | None = None):
        """Public wrapper to add an edge between two nodes."""
        properties = properties or {}
        return self._graph.add_edge(
            source_id=source_id, target_id=target_id, edge_type=relation_type, properties=properties
        )

    def create_or_merge_node(self, item_id: str, properties: dict, memory_type: str | None = None):
        """Public wrapper to upsert a raw node with properties. Returns item_id."""
        self._graph.add_node(item_id=item_id, properties=properties, memory_type=memory_type)
        return item_id

    def delete(self, item_id: str, **kwargs) -> bool:
        t0 = perf_counter()
        result = self._crud.delete(item_id)
        # Emit memory_delete event for observability
        # noinspection PyArgumentList
        SmartMemory._MEM_EMIT(
            "memory_delete",
            "delete",
            {"item_id": item_id, "success": result, "duration_ms": (perf_counter() - t0) * 1000.0},
        )
        return result

    def search(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str = None,
        conversation_context: Optional[Union[ConversationContext, Dict[str, Any]]] = None,
        **kwargs,
    ):
        """
        Search using canonical search component with automatic scope filtering.

        Filtering by tenant/workspace/user is handled automatically by ScopeProvider.
        This ensures consistent isolation across all search operations.
        """
        if memory_type == "working":
            # Check if persistence is enabled; if so, use canonical search on persisted working items
            persist_enabled = False
            try:
                from smartmemory.utils import get_config

                wm_cfg = get_config("working_memory") or {}
                persist_enabled = bool(wm_cfg.get("persist", True))
            except Exception:
                persist_enabled = True

            if persist_enabled:
                # Use canonical search for working memory stored in graph
                # ScopeProvider handles tenant/user filtering automatically
                results = self._search.search(query, top_k=top_k, memory_type="working", **kwargs)

                # Optional: filter by conversation_id when provided
                conv_id = None
                try:
                    if isinstance(conversation_context, ConversationContext):
                        conv_id = conversation_context.conversation_id
                    elif isinstance(conversation_context, dict):
                        conv_id = conversation_context.get("conversation_id")
                except Exception as e:
                    logger.debug(f"Failed to extract conversation_id from conversation_context: {e}")
                    conv_id = None
                if conv_id and results:
                    results = [r for r in results if getattr(r, "metadata", {}).get("conversation_id") == conv_id]

                return results[:top_k]
            else:
                # Fallback to in-memory buffer behavior
                if hasattr(self, "_working_buffer") and self._working_buffer:
                    results = []
                    # If a conversation context is provided, bias towards that conversation_id
                    conv_id = None
                    try:
                        if isinstance(conversation_context, ConversationContext):
                            conv_id = conversation_context.conversation_id
                        elif isinstance(conversation_context, dict):
                            conv_id = conversation_context.get("conversation_id")
                    except Exception as e:
                        logger.debug(f"Failed to extract conversation_id from conversation_context (buffer path): {e}")
                        conv_id = None

                    for item in self._working_buffer:
                        # Prefer items from the same conversation when available
                        if conv_id and hasattr(item, "metadata") and isinstance(item.metadata, dict):
                            if item.metadata.get("conversation_id") == conv_id:
                                if query == "*" or query == "" or len(query.strip()) == 0:
                                    results.append(item)
                                    continue
                                if query.lower() in str(item.content).lower():
                                    results.append(item)
                                    continue
                                if hasattr(item, "metadata") and item.metadata:
                                    metadata_str = str(item.metadata).lower()
                                    if query.lower() in metadata_str:
                                        results.append(item)
                                        continue
                        if query == "*" or query == "" or len(query.strip()) == 0:
                            results.append(item)
                        elif query.lower() in str(item.content).lower():
                            results.append(item)
                        elif hasattr(item, "metadata") and item.metadata:
                            metadata_str = str(item.metadata).lower()
                            if query.lower() in metadata_str:
                                results.append(item)
                    if not results and self._working_buffer:
                        results = list(self._working_buffer)[-top_k:]
                    return results[:top_k]
                return []

        # Use canonical search component directly
        # ScopeProvider handles all tenant/workspace/user filtering automatically
        t0 = perf_counter()
        results = self._search.search(query, top_k=top_k, memory_type=memory_type, **kwargs)

        # Emit memory_retrieve event for observability
        # noinspection PyArgumentList
        SmartMemory._MEM_EMIT(
            "memory_retrieve",
            "search",
            {
                "query_length": len(query) if query else 0,
                "top_k": top_k,
                "memory_type": memory_type,
                "results_count": len(results) if results else 0,
                "duration_ms": (perf_counter() - t0) * 1000.0,
            },
        )

        return results[:top_k]

    # Linking
    def link(self, source_id: str, target_id: str, link_type: Union[str, "LinkType"] = "RELATED") -> str:
        """Link two memory items."""
        if hasattr(link_type, "value"):
            link_type = link_type.value
        return self._linking.link(source_id, target_id, link_type)

    def get_links(self, item_id: str, memory_type: str = "semantic") -> List[str]:
        """Get all links (triples) for a memory item."""
        return self._linking.get_links(item_id, memory_type)

    def get_neighbors(self, item_id: str):
        """Return neighboring MemoryItems for a node with link types."""
        return self._graph_ops.get_neighbors(item_id)

    # Personalization & Feedback
    def personalize(self, traits: dict = None, preferences: dict = None) -> None:
        """Personalize the memory system for the current user."""
        return self._personalization.personalize(traits, preferences)

    def update_from_feedback(self, feedback: dict, memory_type: str = "semantic") -> None:
        return self._personalization.update_from_feedback(feedback, memory_type)

    def embeddings_search(self, embedding, top_k=5):
        """Search for memory items most similar to the given embedding."""
        return self._search.embeddings_search(embedding, top_k=top_k)

    def add_tags(self, item_id: str, tags: list) -> bool:
        """Delegate add_tags to CRUD submodule."""
        return self._crud.add_tags(item_id, tags)

    def challenge(self, assertion: str, memory_type: str = "semantic", use_llm: bool = True):
        """Challenge an assertion against existing knowledge to detect contradictions."""
        from smartmemory.reasoning.challenger import AssertionChallenger

        challenger = AssertionChallenger(self, use_llm=use_llm)
        return challenger.challenge(assertion, memory_type=memory_type)

    # =========================================================================
    # Delegated to MonitoringManager
    # =========================================================================

    def summary(self) -> dict:
        return self._monitoring_mgr.summary()

    def orphaned_notes(self) -> list:
        return self._monitoring_mgr.orphaned_notes()

    def prune(self, strategy="old", days=365, **kwargs):
        return self._monitoring_mgr.prune(strategy, days, **kwargs)

    def find_old_notes(self, days: int = 365) -> list:
        return self._monitoring_mgr.find_old_notes(days)

    def self_monitor(self) -> dict:
        return self._monitoring_mgr.self_monitor()

    def reflect(self, top_k: int = 5) -> dict:
        return self._monitoring_mgr.reflect(top_k)

    def summarize(self, max_items: int = 10) -> dict:
        return self._monitoring_mgr.summarize(max_items)

    # =========================================================================
    # Delegated to EvolutionManager
    # =========================================================================

    def run_evolution_cycle(self):
        return self._evolution_mgr.run_evolution_cycle()

    def commit_working_to_episodic(self, remove_from_source: bool = True) -> List[str]:
        return self._evolution_mgr.commit_working_to_episodic(remove_from_source)

    def commit_working_to_procedural(self, remove_from_source: bool = True) -> List[str]:
        return self._evolution_mgr.commit_working_to_procedural(remove_from_source)

    def run_clustering(self) -> dict:
        return self._evolution_mgr.run_clustering()

    # =========================================================================
    # Delegated to EnrichmentManager
    # =========================================================================

    def enrich(self, item_id: str, routines: Optional[List[str]] = None) -> None:
        return self._enrichment_mgr.enrich(item_id, routines)

    def ground(self, item_id: str, source_url: str, validation: Optional[dict] = None) -> None:
        return self._enrichment_mgr.ground(item_id, source_url, validation)

    def ground_context(self, context: dict):
        return self._enrichment_mgr.ground_context(context)

    def resolve_external(self, node: MemoryItem) -> Optional[list]:
        return self._enrichment_mgr.resolve_external(node)

    # =========================================================================
    # Delegated to DebugManager
    # =========================================================================

    def debug_search(self, query: str, top_k: int = 5) -> dict:
        return self._debug_mgr.debug_search(query, top_k)

    def get_all_items_debug(self) -> Dict[str, Any]:
        return self._debug_mgr.get_all_items_debug()

    def fix_search_if_broken(self) -> dict:
        fix_info, new_search = self._debug_mgr.fix_search_if_broken()
        self._search = new_search
        return fix_info

    def clear(self):
        """Clear all memory from ALL storage backends comprehensively."""
        result = self._debug_mgr.clear()

        # Clear working memory buffer
        if hasattr(self, "_working_buffer"):
            self._working_buffer.clear()

        # Clear canonical memory store
        if hasattr(self, "_canonical_store"):
            self._canonical_store.clear()

        # Clear in-memory caches and mixins
        if hasattr(self, "_cache"):
            self._cache.clear()

        if hasattr(self, "_operation_stats"):
            for key in self._operation_stats:
                self._operation_stats[key] = 0

        # Clear any remaining memory type stores
        memory_types = ["semantic", "episodic", "procedural", "zettelkasten"]
        for memory_type in memory_types:
            if hasattr(self, f"_{memory_type}_store"):
                store = getattr(self, f"_{memory_type}_store")
                if hasattr(store, "clear"):
                    store.clear()

        return result

    # =========================================================================
    # Temporal query methods
    # =========================================================================

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

    # =========================================================================
    # Graph Integrity Operations
    # =========================================================================

    def delete_run(self, run_id: str) -> int:
        """Delete all entities created by a specific pipeline run.

        Use this to clean up entities from a failed or re-run pipeline execution.
        All nodes with the matching run_id in their metadata will be deleted.

        Args:
            run_id: The pipeline run identifier (UUID string)

        Returns:
            Number of entities deleted
        """
        return self._graph.delete_by_run_id(run_id)

    def rename_entity_type(self, old_type: str, new_type: str) -> int:
        """Rename an entity type across all matching entities.

        Use this when the ontology evolves and entity types need to be renamed.
        All entities with entity_type=old_type will be updated to new_type.

        Args:
            old_type: Current entity type name
            new_type: New entity type name

        Returns:
            Number of entities updated
        """
        return self._graph.rename_entity_type(old_type, new_type)

    def merge_entity_types(self, source_types: List[str], target_type: str) -> int:
        """Merge multiple entity types into a single target type.

        Use this to consolidate fragmented or similar entity types.

        Args:
            source_types: List of entity types to merge
            target_type: The target entity type name

        Returns:
            Total number of entities updated
        """
        return self._graph.merge_entity_types(source_types, target_type)
