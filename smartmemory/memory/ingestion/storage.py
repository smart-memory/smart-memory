"""
Storage pipeline module for ingestion flow.

This module handles all storage operations including:
- Vector store operations
- Graph storage operations
- Triple processing and relationship creation
- Entity node creation and management
"""
from typing import Dict, List, Any

from smartmemory.memory.ingestion import utils as ingestion_utils
from smartmemory.models.memory_item import MemoryItem
from smartmemory.plugins.embedding import create_embeddings


class StoragePipeline:
    """
    Handles all storage operations for the ingestion pipeline.
    
    Manages vector store operations, graph storage, and relationship creation.
    """

    def __init__(self, memory, observer):
        """
        Initialize storage pipeline.
        
        Args:
            memory: SmartMemory instance for storage operations
            observer: IngestionObserver instance for event emission
        """
        self.memory = memory
        self.observer = observer

    def save_to_vector_and_graph(self, context: Dict[str, Any]):
        """Delegate vector store and graph saving to their native modules."""
        from smartmemory.stores.vector.vector_store import VectorStore

        # Create VectorStore instance directly
        vector_store = VectorStore()
        item = context['item']

        # Ensure embedding is generated if not present
        if not hasattr(item, 'embedding') or item.embedding is None:
            try:
                item.embedding = create_embeddings(str(item.content))
            except Exception as e:
                print(f"Warning: Failed to generate embedding: {e}")
                # Continue without embedding - graph storage will still work
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
                print(f"✅ Upserted embedding to vector store for item: {item.item_id}")
            except Exception as e:
                print(f"Warning: Failed to upsert embedding to vector store: {e}")
        else:
            print(f"⚠️  No embedding generated for item: {item.item_id}")

        # Graph storage is handled separately by the memory system
        # No need to duplicate storage here

    def process_extracted_relations(self, context: Dict[str, Any], item_id: str, relations: List[Any]):
        """
        Process extracted relations to create relationships in the graph.
        This method is always active regardless of ontology settings.
        """
        if not relations:
            return

        # Use SmartGraph API for relationship creation (handles validation/caching)
        graph = self.memory._graph

        for relation in relations:
            try:
                # Handle different relation formats
                if isinstance(relation, dict):
                    subject = relation.get('source_id') or relation.get('subject') or relation.get('source')
                    predicate = relation.get('relation_type') or relation.get('predicate') or relation.get('type')
                    object_node = relation.get('target_id') or relation.get('object') or relation.get('target')
                elif isinstance(relation, (list, tuple)) and len(relation) == 3:
                    subject, predicate, object_node = relation
                else:
                    continue  # Skip invalid relations

                if not all([subject, predicate, object_node]):
                    continue  # Skip incomplete relations

                # Sanitize relationship type
                predicate = ingestion_utils.sanitize_relation_type(predicate)

                # Create nodes for subject and object if they don't exist
                subject_id = self.ensure_entity_node(subject, context)
                object_id = self.ensure_entity_node(object_node, context)

                # Emit edge creation event
                self.observer.emit_edge_creation_start(
                    subject=subject,
                    predicate=predicate,
                    object_node=object_node
                )

                # Create the relationship
                edge_id = graph.add_edge(
                    source_id=subject_id,
                    target_id=object_id,
                    edge_type=predicate,
                    properties={
                        'created_from_triple': True,
                        'source_item': item_id,
                        'created_at': context['item'].created_at.isoformat() if hasattr(context['item'], 'created_at') else None
                    }
                )

                # Track edges created
                context['edges_created'] = context.get('edges_created', 0) + 1

                # Emit edge creation complete event
                self.observer.emit_edge_created(
                    subject=subject,
                    predicate=predicate,
                    object_node=object_node,
                    edge_id=edge_id
                )

            except Exception as e:
                print(f"⚠️  Failed to process triple {triple}: {e}")
                continue  # Skip failed triples but continue processing others

    def ensure_entity_node(self, entity_name: str, context: Dict[str, Any]) -> str:
        """
        Ensure an entity node exists in the graph, creating it if necessary.
        Returns the node ID.
        """
        # Normalize entity name for disambiguation
        normalized_name = str(entity_name).strip().lower()

        # Check if entity already exists in context (use normalized name for lookup)
        entity_ids = context.get('entity_ids') or {}
        if normalized_name in entity_ids:
            # Emit entity reuse event
            self.observer.emit_entity_reused(
                entity_name=entity_name,
                normalized_name=normalized_name,
                existing_node_id=entity_ids[normalized_name]
            )
            return entity_ids[normalized_name]

        # Emit entity creation start event
        self.observer.emit_entity_creation_start(
            entity_name=entity_name,
            normalized_name=normalized_name
        )

        # Create a simple entity node
        entity_item = MemoryItem(
            content=entity_name,
            metadata={
                'name': entity_name,
                'normalized_name': normalized_name,  # Store normalized name for disambiguation
                'type': 'entity',
                'created_from_triple': True
            }
        )

        # Add to graph using SmartGraph API and ensure proper labeling as an Entity
        graph = self.memory._graph

        # Prepare properties for SmartGraph (flattened structure)
        properties = {
            'content': entity_item.content,
            'name': entity_name,
            'normalized_name': normalized_name,
            'type': 'entity',
            'node_category': 'entity',
            'entity_type': 'unknown',
            'created_from_triple': True
        }

        add_result = graph.add_node(
            item_id=entity_item.item_id,
            properties=properties,
            memory_type='entity'
        )

        # Extract the string node ID from the result
        node_id = entity_item.item_id
        if isinstance(add_result, dict) and add_result.get('item_id'):
            node_id = add_result['item_id']
        elif isinstance(add_result, str):
            node_id = add_result

        # Store in context for future reference using NORMALIZED name as key
        if 'entity_ids' not in context:
            context['entity_ids'] = {}
        context['entity_ids'][normalized_name] = node_id  # Use normalized name as key

        # Emit entity creation complete event
        self.observer.emit_entity_created(
            entity_name=entity_name,
            normalized_name=normalized_name,
            node_id=node_id
        )

        return node_id
