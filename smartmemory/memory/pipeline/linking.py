"""LinkingEngine component for componentized memory ingestion pipeline.
Handles entity linking and deduplication with configurable algorithms.
"""
import logging
from typing import Dict, Any, List

from smartmemory.memory.pipeline.components import PipelineComponent, ComponentResult
from smartmemory.memory.pipeline.config import LinkingConfig
from smartmemory.memory.pipeline.state import StorageState
from smartmemory.memory.pipeline.transactions.change_set import ChangeOp, ChangeSet
from smartmemory.utils.pipeline_utils import create_error_result

logger = logging.getLogger(__name__)


class LinkingEngine(PipelineComponent[LinkingConfig]):
    """
    Component responsible for entity linking and deduplication.
    Uses the memory system's linking module for cross-reference resolution.
    """

    def __init__(self, memory_instance):
        self.memory = memory_instance
        # Bind to SmartMemory's internal linking service
        self.linking = self.memory._linking
        if self.linking is None:
            raise RuntimeError("SmartMemory does not provide a linking service (_linking/linking)")

    def _activate_working_memory(self, context: Dict[str, Any]):
        """Activate working memory for the context (stub implementation)"""
        # This is currently a stub in the original code
        # Future implementation would handle working memory activation
        context['working_memory_activated'] = True

    def _save_to_vector_and_graph(self, context: Dict[str, Any]):
        """Save original item in vector store and graph"""
        try:
            from smartmemory.stores.vector.vector_store import VectorStore
            from smartmemory.plugins.embedding import create_embeddings

            item = context.get('item')
            if not item:
                logger.warning("No item found in context for vector/graph saving")
                context['vector_saved'] = False
                context['graph_saved'] = False
                return

            # Use VectorStore instance
            vector_store = VectorStore()

            # Ensure embedding is generated if not present
            if not hasattr(item, 'embedding') or item.embedding is None:
                try:
                    item.embedding = create_embeddings(str(item.content))
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")
                    item.embedding = None

            # Add to vector store if embedding was successfully generated
            if item.embedding is not None:
                try:
                    vector_store.upsert(
                        item_id=str(item.item_id),
                        embedding=item.embedding.tolist() if hasattr(item.embedding, 'tolist') else item.embedding,
                        metadata=item.metadata or {},
                        node_ids=[str(item.item_id)],  # Link vector to graph node
                        is_global=True  # Make searchable globally
                    )
                    logger.info(f"Upserted embedding to vector store for item: {item.item_id}")
                    context['vector_saved'] = True
                except Exception as e:
                    logger.warning(f"Failed to upsert embedding to vector store: {e}")
                    context['vector_saved'] = False
            else:
                logger.warning(f"No embedding generated for item: {item.item_id}")
                context['vector_saved'] = False

            # Graph storage is handled separately by the memory system
            context['graph_saved'] = True  # Assume success since it's handled elsewhere

        except Exception as e:
            logger.error(f"Failed to save to vector/graph: {e}")
            context['vector_saved'] = False
            context['graph_saved'] = False

    def validate_config(self, config: LinkingConfig) -> bool:
        """Validate LinkingEngine configuration using typed config"""
        try:
            # similarity_threshold range
            if not (0.0 <= float(getattr(config, 'similarity_threshold', 0.8)) <= 1.0):
                return False
            # linking_algorithm presence (optional fixed set in future)
            la = getattr(config, 'linking_algorithm', 'default')
            if not isinstance(la, str):
                return False
            # booleans
            for b in ('deduplication_enabled', 'cross_reference_resolution', 'working_memory_activation'):
                val = getattr(config, b, True)
                if not isinstance(val, bool):
                    return False
            return True
        except Exception:
            return False

    def run(self, storage_state: StorageState, config: LinkingConfig) -> ComponentResult:
        """
        Execute LinkingEngine with given storage state and configuration.
        
        Args:
            storage_state: StorageState from previous stage with stored nodes/relations
            config: Linking configuration dict
        
        Returns:
            ComponentResult with linking results and metadata
        """
        try:
            if not storage_state or not storage_state.success:
                return create_error_result('linking_engine', ValueError('Invalid or failed storage state'))

            # Build context from storage state
            context = {
                'item_id': storage_state.data.get('item_id'),
                'entity_ids': storage_state.data.get('entity_ids', {}),
                'edges_created': storage_state.data.get('edges_created', 0),
                'stored_nodes': storage_state.data.get('stored_nodes', []),
                'stored_relations': storage_state.data.get('stored_relations', []),
                'storage_metadata': storage_state.data.get('storage_metadata', {})
            }

            # Add memory item if available (should be passed through pipeline)
            memory_item = storage_state.metadata.get('memory_item')
            if memory_item:
                context['item'] = memory_item

            # Activate working memory (stub)
            self._activate_working_memory(context)

            # Ensure context['entity_ids'] has the structure expected by stages/Linking
            entity_ids = context.get('entity_ids') or {}
            if not isinstance(entity_ids, dict):
                entity_ids = {}
            # semantic_entities
            if 'semantic_entities' not in entity_ids:
                semantic_entities = []
                for n in context.get('stored_nodes', []) or []:
                    if isinstance(n, dict):
                        # Prefer explicit id/item_id/name if available
                        node_id = n.get('id') or n.get('item_id') or n.get('name')
                        ent = dict(n)
                        if node_id and 'id' not in ent:
                            ent['id'] = node_id
                        if ent:
                            semantic_entities.append(ent)
                    else:
                        # Fallback: wrap raw value as id
                        semantic_entities.append({'id': str(n)})
                entity_ids['semantic_entities'] = semantic_entities
            # semantic_relations
            if 'semantic_relations' not in entity_ids:
                semantic_relations = []
                for e in context.get('stored_relations', []) or []:
                    if isinstance(e, dict):
                        src = e.get('source') or e.get('src') or e.get('from')
                        tgt = e.get('target') or e.get('tgt') or e.get('to')
                        typ = e.get('type') or e.get('edge_type') or 'RELATED'
                        if src and tgt:
                            semantic_relations.append({'source': src, 'target': tgt, 'type': typ})
                entity_ids['semantic_relations'] = semantic_relations
            context['entity_ids'] = entity_ids

            # Optional preview mode (non-mutating). When enabled, we return a change_set with proposed ops
            preview_mode = bool(getattr(config, 'preview', False))
            proposed_ops: List[ChangeOp] = []

            if not preview_mode:
                # Perform entity linking (may mutate internally)
                try:
                    self.linking.link_new_item(context)
                    linking_success = True
                    linking_error = None
                except Exception as e:
                    linking_success = False
                    linking_error = str(e)
                    logger.warning(f"Linking failed: {e}")

                # Save to vector store and graph
                self._save_to_vector_and_graph(context)
            else:
                linking_success = True
                linking_error = None

            # Ensure node_ids is available for downstream stages
            context['node_ids'] = context.get('entity_ids') or {}

            # Minimal dedup/canonicalization
            sem_entities = (context.get('entity_ids') or {}).get('semantic_entities', [])
            sem_relations = (context.get('entity_ids') or {}).get('semantic_relations', [])

            def _norm_label(s: str) -> str:
                try:
                    return (s or '').strip().lower()
                except Exception:
                    return str(s)

            # Build clusters by explicit id or normalized name
            clusters = {}
            for ent in sem_entities:
                ent_id = ent.get('id') or ent.get('item_id')
                name = ent.get('name') or ent.get('label') or ent.get('title')
                key = ent_id or _norm_label(name)
                if not key:
                    # fallback to any stable repr
                    key = _norm_label(str(ent))
                clusters.setdefault(key, []).append(ent)

            canonical_map = {}  # original_id/name -> canonical_id
            is_a_edges_added = 0
            try:
                for key, group in clusters.items():
                    if not group:
                        continue
                    # choose canonical: existing graph node if any, else first with explicit id, else first
                    candidate_ids = [g.get('id') or g.get('item_id') or g.get('name') for g in group]
                    candidate_ids = [str(c) for c in candidate_ids if c]
                    canonical_id = None
                    # prefer an id that already exists in graph
                    for cid in candidate_ids:
                        try:
                            if self.memory.get(cid):
                                canonical_id = cid
                                break
                        except Exception:
                            # get() may raise if missing backend; ignore
                            pass
                    if canonical_id is None:
                        canonical_id = candidate_ids[0] if candidate_ids else str(key)

                    # map all members to canonical and add IS_A edges
                    for cid in candidate_ids:
                        canonical_map[cid] = canonical_id
                        if cid != canonical_id:
                            if preview_mode:
                                # Propose IS_A edge and property update as ops
                                proposed_ops.append(ChangeOp(op_type='add_edge', args={
                                    'source': cid, 'target': canonical_id, 'relation_type': 'IS_A', 'properties': {}
                                }))
                                proposed_ops.append(ChangeOp(op_type='set_properties', args={
                                    'node_id': cid, 'properties': {'canonical_id': canonical_id}
                                }))
                                is_a_edges_added += 1
                            else:
                                try:
                                    # Non-destructive: add IS_A edge
                                    self.memory.add_edge(cid, canonical_id, relation_type='IS_A', properties={})
                                    is_a_edges_added += 1
                                except Exception:
                                    pass
                                try:
                                    # annotate node with canonical_id for fast lookup
                                    self.memory.update_properties(cid, {'canonical_id': canonical_id})
                                except Exception:
                                    pass
            except Exception as _:
                # Do not fail pipeline on dedup errors
                pass

            # annotate context with canonical mapping for downstream
            context['canonical_map'] = canonical_map

            # Extract linking results
            linked_entities = []
            # Build a quick index of nodes that appear in any relation
            related_ids = set()
            for rel in sem_relations:
                src = rel.get('source')
                tgt = rel.get('target')
                if src:
                    related_ids.add(str(src))
                if tgt:
                    related_ids.add(str(tgt))
            for ent in sem_entities:
                ent_id = ent.get('id') or ent.get('name')
                if ent_id:
                    linked_entities.append({
                        'name': ent.get('name') or ent_id,
                        'id': ent_id,
                        'canonical_id': canonical_map.get(str(ent_id), str(ent_id)),
                        'linked': str(ent_id) in related_ids
                    })

            # Build deduplication results
            dedup_results = {
                'entities_processed': len(context.get('entity_ids', {})),
                'links_found': len(sem_relations),
                'is_a_edges_added': is_a_edges_added,
                'unique_entities_canonical': len(set(canonical_map.values())) if canonical_map else len(sem_entities),
                'deduplication_enabled': config.deduplication_enabled
            }

            # Build linking metadata
            linking_metadata = {
                'linking_algorithm': config.linking_algorithm,
                'similarity_threshold': config.similarity_threshold,
                'working_memory_activated': context.get('working_memory_activated', False),
                'vector_saved': context.get('vector_saved', False) if not preview_mode else False,
                'graph_saved': context.get('graph_saved', False) if not preview_mode else False,
                'linking_success': linking_success,
                'linking_error': linking_error
            }

            result_data = {
                'linked_entities': linked_entities,
                'dedup_results': dedup_results,
                'linking_metadata': linking_metadata,
                'context': context  # Pass context for downstream stages
            }

            # Attach a change_set when in preview mode (non-mutating)
            if preview_mode and proposed_ops:
                cs = ChangeSet.new(stage='linking', plugin='LinkingEngine', run_id=config.run_id)
                cs.ops.extend(proposed_ops)
                result_data['change_set'] = {
                    'change_set_id': cs.change_set_id,
                    'ops_count': len(cs.ops),
                }
                # Expose ops for API layer to persist the ChangeSet
                result_data['change_set_ops'] = [
                    {
                        'op_type': op.op_type,
                        'args': op.args,
                    }
                    for op in proposed_ops
                ]

            return ComponentResult(
                success=True,
                data=result_data,
                metadata={
                    'stage': 'linking_engine',
                    'entities_linked': len(linked_entities),
                    'links_found': len(sem_relations)
                }
            )

        except Exception as e:
            return create_error_result('linking_engine', e)
