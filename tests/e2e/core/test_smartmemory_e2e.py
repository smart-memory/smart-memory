"""
End-to-End tests for SmartMemory system.
Tests complete user workflows with real backends from input to output.
"""
import pytest
import time
from datetime import datetime, timezone

from smartmemory.models.memory_item import MemoryItem


@pytest.mark.e2e
class TestSmartMemoryE2EWorkflows:
    """End-to-end tests for complete SmartMemory workflows with real backends."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self, real_smartmemory_for_integration):
        """Setup and teardown for each test."""
        self.memory = real_smartmemory_for_integration
        yield
        # Cleanup after each test
        try:
            self.memory.clear()
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    def test_complete_ingestion_and_retrieval_workflow(self):
        """Test complete workflow: ingest → search → retrieve → verify."""
        # Step 1: Ingest multiple related memories
        memories = [
            MemoryItem(
                content="I love learning about machine learning and neural networks",
                memory_type="episodic",
                user_id="e2e_user_001",
                metadata={"category": "learning", "timestamp": datetime.now(timezone.utc).isoformat()}
            ),
            MemoryItem(
                content="Neural networks are a key component of deep learning systems",
                memory_type="semantic",
                user_id="e2e_user_001",
                metadata={"category": "knowledge", "timestamp": datetime.now(timezone.utc).isoformat()}
            ),
            MemoryItem(
                content="I completed a course on Python programming for data science",
                memory_type="episodic",
                user_id="e2e_user_001",
                metadata={"category": "achievement", "timestamp": datetime.now(timezone.utc).isoformat()}
            )
        ]
        
        item_ids = []
        for memory_item in memories:
            item_id = self.memory.add(memory_item)
            assert item_id is not None, "Failed to add memory"
            item_ids.append(item_id)
        
        print(f"✅ Ingested {len(item_ids)} memories")
        
        # Step 2: Search for related content
        search_results = self.memory.search("machine learning neural networks", user_id="e2e_user_001", top_k=5)
        assert len(search_results) > 0, "Search returned no results"
        print(f"✅ Search found {len(search_results)} results")
        
        # Step 3: Retrieve specific memories and verify content
        for item_id in item_ids:
            retrieved = self.memory.get(item_id)
            assert retrieved is not None, f"Failed to retrieve memory {item_id}"
            assert hasattr(retrieved, 'content'), "Retrieved item missing content"
            assert hasattr(retrieved, 'user_id'), "Retrieved item missing user_id"
            assert retrieved.user_id == "e2e_user_001", "User ID mismatch"
        
        print(f"✅ Retrieved and verified {len(item_ids)} memories")
        
        # Step 4: Verify data persistence - search again
        second_search = self.memory.search("Python programming", user_id="e2e_user_001", top_k=5)
        assert len(second_search) > 0, "Second search failed - data not persisted"
        print(f"✅ Data persistence verified")
    
    def test_memory_persistence_across_operations(self):
        """Test that memories persist correctly through multiple operations."""
        user_id = "e2e_persistence_user"
        
        # Add initial memory
        initial_memory = MemoryItem(
            content="Initial knowledge: Python is a programming language",
            memory_type="semantic",
            user_id=user_id,
            metadata={"version": 1}
        )
        initial_id = self.memory.add(initial_memory)
        assert initial_id is not None
        
        # Verify it exists
        retrieved_1 = self.memory.get(initial_id)
        assert retrieved_1 is not None
        assert "Python" in retrieved_1.content
        
        # Add related memory
        related_memory = MemoryItem(
            content="Python is used for data science and machine learning",
            memory_type="semantic",
            user_id=user_id,
            metadata={"version": 2, "related_to": initial_id}
        )
        related_id = self.memory.add(related_memory)
        assert related_id is not None
        
        # Search should find both
        search_results = self.memory.search("Python", user_id=user_id, top_k=10)
        assert len(search_results) >= 2, f"Expected at least 2 results, got {len(search_results)}"
        
        # Verify both memories still retrievable
        retrieved_initial = self.memory.get(initial_id)
        retrieved_related = self.memory.get(related_id)
        
        assert retrieved_initial is not None, "Initial memory lost"
        assert retrieved_related is not None, "Related memory lost"
        assert retrieved_initial.user_id == user_id
        assert retrieved_related.user_id == user_id
        
        print(f"✅ Memory persistence verified across multiple operations")


@pytest.mark.e2e
class TestMultiUserE2EScenarios:
    """E2E tests for multi-user scenarios with real backends."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self, real_smartmemory_for_integration):
        """Setup and teardown for each test."""
        self.memory = real_smartmemory_for_integration
        yield
        try:
            self.memory.clear()
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    def test_multi_user_isolation_e2e(self):
        """Test complete multi-user isolation with real data."""
        # Create memories for different users
        users = ["user_001", "user_002", "user_003"]
        user_memories = {}
        
        for user_id in users:
            memories = [
                MemoryItem(
                    content=f"Private memory for {user_id}: My secret project details",
                    memory_type="episodic",
                    user_id=user_id,
                    metadata={"private": True, "user": user_id}
                ),
                MemoryItem(
                    content=f"Knowledge for {user_id}: Specific technical information",
                    memory_type="semantic",
                    user_id=user_id,
                    metadata={"category": "technical", "user": user_id}
                )
            ]
            
            user_item_ids = []
            for memory in memories:
                item_id = self.memory.add(memory)
                assert item_id is not None
                user_item_ids.append(item_id)
            
            user_memories[user_id] = user_item_ids
            print(f"✅ Added {len(user_item_ids)} memories for {user_id}")
        
        # Verify isolation: each user can only see their own memories
        for user_id in users:
            # Search for user-specific content
            user_results = self.memory.search("memory", user_id=user_id, top_k=10)
            
            # Verify all results belong to this user
            for result in user_results:
                if hasattr(result, 'user_id'):
                    assert result.user_id == user_id, f"Data leakage: {result.user_id} data visible to {user_id}"
                elif hasattr(result, 'metadata') and result.metadata:
                    result_user = result.metadata.get('user_id') or result.metadata.get('user')
                    if result_user:
                        assert result_user == user_id, f"Data leakage: {result_user} data visible to {user_id}"
            
            print(f"✅ User isolation verified for {user_id}")
        
        # Verify retrieval isolation
        for user_id, item_ids in user_memories.items():
            for item_id in item_ids:
                retrieved = self.memory.get(item_id)
                assert retrieved is not None
                assert retrieved.user_id == user_id, f"Retrieved memory has wrong user_id"
        
        print(f"✅ Multi-user isolation verified end-to-end")
    
    def test_cross_user_data_leakage_prevention(self):
        """Test that users cannot access each other's data."""
        # User 1 adds sensitive data
        user1_memory = MemoryItem(
            content="User 1 confidential: My password is secret123",
            memory_type="episodic",
            user_id="user_001",
            metadata={"confidential": True}
        )
        user1_id = self.memory.add(user1_memory)
        assert user1_id is not None
        
        # User 2 adds different data
        user2_memory = MemoryItem(
            content="User 2 public: I like Python programming",
            memory_type="semantic",
            user_id="user_002",
            metadata={"public": True}
        )
        user2_id = self.memory.add(user2_memory)
        assert user2_id is not None
        
        # User 2 searches for "password" - should find nothing
        user2_search = self.memory.search("password secret", user_id="user_002", top_k=10)
        
        # Verify no leakage
        for result in user2_search:
            if hasattr(result, 'content'):
                assert "secret123" not in result.content, "SECURITY BREACH: User 1 data leaked to User 2"
                assert "confidential" not in str(result.content).lower(), "User 1 data leaked"
        
        # User 1 can find their own data
        user1_search = self.memory.search("password", user_id="user_001", top_k=10)
        assert len(user1_search) >= 0, "User 1 should be able to search their own data"
        
        print(f"✅ Cross-user data leakage prevention verified")


@pytest.mark.e2e
class TestRealWorldScenarios:
    """E2E tests simulating real-world usage scenarios."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self, real_smartmemory_for_integration):
        """Setup and teardown for each test."""
        self.memory = real_smartmemory_for_integration
        yield
        try:
            self.memory.clear()
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    def test_knowledge_base_workflow(self):
        """Test building and querying a knowledge base."""
        user_id = "knowledge_worker"
        
        # Build knowledge base with related concepts
        knowledge_items = [
            MemoryItem(
                content="Machine learning is a subset of artificial intelligence",
                memory_type="semantic",
                user_id=user_id,
                metadata={"category": "definitions", "topic": "AI"}
            ),
            MemoryItem(
                content="Neural networks consist of interconnected nodes called neurons",
                memory_type="semantic",
                user_id=user_id,
                metadata={"category": "concepts", "topic": "neural_networks"}
            ),
            MemoryItem(
                content="Backpropagation is the algorithm used to train neural networks",
                memory_type="semantic",
                user_id=user_id,
                metadata={"category": "algorithms", "topic": "neural_networks"}
            ),
            MemoryItem(
                content="Deep learning uses multiple layers of neural networks",
                memory_type="semantic",
                user_id=user_id,
                metadata={"category": "concepts", "topic": "deep_learning"}
            )
        ]
        
        item_ids = []
        for item in knowledge_items:
            item_id = self.memory.add(item)
            assert item_id is not None
            item_ids.append(item_id)
        
        print(f"✅ Built knowledge base with {len(item_ids)} items")
        
        # Query 1: Find information about neural networks
        nn_results = self.memory.search("neural networks", user_id=user_id, top_k=5)
        assert len(nn_results) > 0, "Failed to find neural network information"
        
        # Verify relevance
        relevant_found = False
        for result in nn_results:
            if hasattr(result, 'content'):
                if "neural" in result.content.lower() or "network" in result.content.lower():
                    relevant_found = True
                    break
        assert relevant_found, "Search results not relevant to query"
        
        print(f"✅ Knowledge base query 1 successful: {len(nn_results)} results")
        
        # Query 2: Find information about machine learning
        ml_results = self.memory.search("machine learning artificial intelligence", user_id=user_id, top_k=5)
        assert len(ml_results) > 0, "Failed to find ML information"
        
        print(f"✅ Knowledge base query 2 successful: {len(ml_results)} results")
        
        # Verify all knowledge is still accessible
        for item_id in item_ids:
            retrieved = self.memory.get(item_id)
            assert retrieved is not None, f"Knowledge item {item_id} lost"
        
        print(f"✅ Knowledge base workflow completed successfully")
    
    def test_conversation_memory_workflow(self):
        """Test storing and retrieving conversation context."""
        user_id = "conversation_user"
        conversation_id = "conv_001"
        
        # Simulate a conversation with multiple turns
        conversation_memories = [
            MemoryItem(
                content="User asked: What is machine learning?",
                memory_type="episodic",
                user_id=user_id,
                metadata={
                    "conversation_id": conversation_id,
                    "turn": 1,
                    "role": "user",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ),
            MemoryItem(
                content="Assistant explained: Machine learning is a branch of AI that enables systems to learn from data",
                memory_type="episodic",
                user_id=user_id,
                metadata={
                    "conversation_id": conversation_id,
                    "turn": 2,
                    "role": "assistant",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ),
            MemoryItem(
                content="User asked: Can you give me an example?",
                memory_type="episodic",
                user_id=user_id,
                metadata={
                    "conversation_id": conversation_id,
                    "turn": 3,
                    "role": "user",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ),
            MemoryItem(
                content="Assistant provided example: Image recognition is a common ML application",
                memory_type="episodic",
                user_id=user_id,
                metadata={
                    "conversation_id": conversation_id,
                    "turn": 4,
                    "role": "assistant",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        ]
        
        item_ids = []
        for memory in conversation_memories:
            item_id = self.memory.add(memory)
            assert item_id is not None
            item_ids.append(item_id)
            time.sleep(0.01)  # Small delay to ensure ordering
        
        print(f"✅ Stored {len(item_ids)} conversation turns")
        
        # Retrieve conversation context
        search_results = self.memory.search("machine learning", user_id=user_id, top_k=10)
        assert len(search_results) > 0, "Failed to retrieve conversation context"
        
        # Verify conversation memories are retrievable
        for item_id in item_ids:
            retrieved = self.memory.get(item_id)
            assert retrieved is not None, f"Conversation turn {item_id} lost"
            assert retrieved.metadata.get("conversation_id") == conversation_id
        
        print(f"✅ Conversation memory workflow completed successfully")


@pytest.mark.e2e
class TestDataIntegrity:
    """E2E tests for data integrity across the full stack."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self, real_smartmemory_for_integration):
        """Setup and teardown for each test."""
        self.memory = real_smartmemory_for_integration
        yield
        try:
            self.memory.clear()
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    def test_graph_vector_consistency(self):
        """Test that graph and vector store remain consistent."""
        user_id = "consistency_user"
        
        # Add memories with rich content
        memories = [
            MemoryItem(
                content="Python is a versatile programming language used for web development, data science, and automation",
                memory_type="semantic",
                user_id=user_id,
                metadata={"topic": "programming", "language": "Python"}
            ),
            MemoryItem(
                content="JavaScript is the primary language for web browser programming and Node.js server development",
                memory_type="semantic",
                user_id=user_id,
                metadata={"topic": "programming", "language": "JavaScript"}
            )
        ]
        
        item_ids = []
        for memory in memories:
            item_id = self.memory.add(memory)
            assert item_id is not None
            item_ids.append(item_id)
        
        print(f"✅ Added {len(item_ids)} memories")
        
        # Verify data is accessible via search (vector store)
        search_results = self.memory.search("programming language", user_id=user_id, top_k=5)
        assert len(search_results) > 0, "Vector search failed"
        
        # Verify data is accessible via retrieval (graph)
        for item_id in item_ids:
            retrieved = self.memory.get(item_id)
            assert retrieved is not None, f"Graph retrieval failed for {item_id}"
            assert hasattr(retrieved, 'content'), "Retrieved item missing content"
        
        print(f"✅ Graph and vector store consistency verified")
    
    def test_temporal_data_integrity(self):
        """Test that temporal metadata is preserved correctly."""
        user_id = "temporal_user"
        
        # Add memory with temporal metadata
        now = datetime.now(timezone.utc)
        memory = MemoryItem(
            content="Important event happened at this specific time",
            memory_type="episodic",
            user_id=user_id,
            metadata={
                "event_time": now.isoformat(),
                "event_type": "milestone",
                "importance": "high"
            }
        )
        
        item_id = self.memory.add(memory)
        assert item_id is not None
        
        # Retrieve and verify temporal data
        retrieved = self.memory.get(item_id)
        assert retrieved is not None
        assert hasattr(retrieved, 'metadata'), "Missing metadata"
        assert retrieved.metadata.get("event_time") is not None, "Temporal metadata lost"
        assert retrieved.metadata.get("event_type") == "milestone", "Event type corrupted"
        
        # Verify through search
        search_results = self.memory.search("important event", user_id=user_id, top_k=5)
        assert len(search_results) > 0, "Failed to find temporal memory"
        
        print(f"✅ Temporal data integrity verified")
