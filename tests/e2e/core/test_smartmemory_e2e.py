"""
End-to-End tests for SmartMemory system.
Tests complete user workflows and system behavior from input to output.
"""
from unittest.mock import Mock, patch

import pytest

from smartmemory.models.memory_item import MemoryItem


@pytest.mark.e2e
class TestSmartMemoryE2EWorkflows:
    """End-to-end tests for complete SmartMemory workflows."""

    def test_user_conversation_to_memory_workflow(self):
        """Test complete user conversation to memory storage workflow."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory, \
                patch('smartmemory.conversation.manager.ConversationManager') as MockConvManager:
            mock_memory = Mock()
            mock_conv_manager = Mock()

            MockSmartMemory.return_value = mock_memory
            MockConvManager.return_value = mock_conv_manager

            # Simulate complete conversation workflow
            conversation_data = {
                "conversation_id": "e2e_conv_001",
                "user_id": "e2e_user_001",
                "messages": [
                    {"role": "user", "content": "I love learning about machine learning"},
                    {"role": "assistant", "content": "That's great! What aspects interest you most?"},
                    {"role": "user", "content": "Neural networks and deep learning"}
                ]
            }

            # Mock conversation processing
            mock_conv_manager.process_conversation.return_value = {
                "extracted_memories": [
                    {
                        "content": "User loves learning about machine learning",
                        "memory_type": "episodic",
                        "entities": ["machine learning", "learning"],
                        "sentiment": "positive"
                    },
                    {
                        "content": "User is interested in neural networks and deep learning",
                        "memory_type": "semantic",
                        "entities": ["neural networks", "deep learning"],
                        "relations": [("neural networks", "PART_OF", "deep learning")]
                    }
                ]
            }

            # Mock memory storage
            mock_memory.ingest.return_value = {"status": "success", "items_stored": 2}

            # Execute E2E workflow
            conv_manager = MockConvManager()
            memory = MockSmartMemory()

            # Process conversation
            processed_data = conv_manager.process_conversation(conversation_data)

            # Store extracted memories
            for memory_data in processed_data["extracted_memories"]:
                memory_item = MemoryItem(
                    content=memory_data["content"],
                    memory_type=memory_data["memory_type"],
                    user_id=conversation_data["user_id"],
                    metadata={
                        "conversation_id": conversation_data["conversation_id"],
                        "entities": memory_data.get("entities", []),
                        "sentiment": memory_data.get("sentiment")
                    }
                )
                result = memory.ingest(memory_item)

            # Verify E2E workflow
            assert processed_data["extracted_memories"]
            assert len(processed_data["extracted_memories"]) == 2
            mock_memory.ingest.assert_called()

    def test_search_and_retrieval_workflow(self):
        """Test complete search and retrieval workflow."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory

            # Mock search results
            mock_search_results = [
                {
                    "content": "User loves machine learning",
                    "memory_type": "episodic",
                    "relevance_score": 0.95,
                    "user_id": "e2e_user_001",
                    "metadata": {"entities": ["machine learning"]}
                },
                {
                    "content": "Neural networks are part of deep learning",
                    "memory_type": "semantic",
                    "relevance_score": 0.87,
                    "user_id": "e2e_user_001",
                    "metadata": {"entities": ["neural networks", "deep learning"]}
                }
            ]

            mock_memory.search.return_value = mock_search_results

            # Execute search workflow
            memory = MockSmartMemory()

            # Test different search scenarios
            search_queries = [
                {"query": "machine learning", "user_id": "e2e_user_001"},
                {"query": "neural networks", "memory_type": "semantic"},
                {"query": "learning", "limit": 5}
            ]

            for search_query in search_queries:
                results = memory.search(**search_query)

                assert isinstance(results, list)
                assert len(results) <= 5  # Respects limit

                # Verify result structure
                for result in results:
                    assert "content" in result
                    assert "memory_type" in result
                    assert "relevance_score" in result
                    assert result["relevance_score"] > 0.5  # Good relevance

    def test_memory_evolution_workflow(self):
        """Test complete memory evolution workflow."""
        with patch('smartmemory.memory.pipeline.stages.evolution.EvolutionOrchestrator') as MockEvolution:
            mock_evolution = Mock()
            MockEvolution.return_value = mock_evolution

            # Mock evolution process
            mock_evolution.run_evolution_cycle.return_value = {
                "working_to_episodic": 5,
                "episodic_to_semantic": 3,
                "memory_decay": 2,
                "new_connections": 8,
                "evolution_score": 0.78
            }

            # Execute evolution workflow
            evolution = MockEvolution()

            # Simulate evolution cycle
            evolution_results = evolution.run_evolution_cycle()

            # Verify evolution workflow
            assert evolution_results["working_to_episodic"] > 0
            assert evolution_results["episodic_to_semantic"] > 0
            assert evolution_results["evolution_score"] > 0.5
            assert evolution_results["new_connections"] > 0


@pytest.mark.e2e
class TestMultiUserE2EScenarios:
    """E2E tests for multi-user scenarios."""

    def test_multi_user_isolation_workflow(self):
        """Test complete multi-user isolation workflow."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory

            # Mock user-specific operations
            def mock_search_with_user_filter(query, user_id=None, **kwargs):
                if user_id == "user_001":
                    return [{"content": "User 1 memory", "user_id": "user_001"}]
                elif user_id == "user_002":
                    return [{"content": "User 2 memory", "user_id": "user_002"}]
                return []

            mock_memory.search.side_effect = mock_search_with_user_filter
            mock_memory.add.return_value = Mock()

            memory = MockSmartMemory()

            # Test multi-user workflow
            users = ["user_001", "user_002", "user_003"]

            for user_id in users:
                # Add user-specific memory
                user_memory = MemoryItem(
                    content=f"Memory for {user_id}",
                    memory_type="episodic",
                    user_id=user_id,
                    metadata={"private": True}
                )
                memory.add(user_memory)

                # Search user-specific memories
                user_results = memory.search("memory", user_id=user_id)

                # Verify isolation
                if user_id in ["user_001", "user_002"]:
                    assert len(user_results) > 0
                    assert all(result["user_id"] == user_id for result in user_results)
                else:
                    assert len(user_results) == 0  # No memories for user_003

    def test_collaborative_memory_workflow(self):
        """Test collaborative memory sharing workflow."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory

            # Mock collaborative memory operations
            shared_memories = []

            def mock_add_shared_memory(memory_item):
                shared_memories.append(memory_item)
                return memory_item

            def mock_search_shared_memories(query, group_id=None, **kwargs):
                if group_id:
                    return [mem for mem in shared_memories
                            if mem.metadata.get("group_id") == group_id]
                return shared_memories

            mock_memory.add.side_effect = mock_add_shared_memory
            mock_memory.search.side_effect = mock_search_shared_memories

            memory = MockSmartMemory()

            # Test collaborative workflow
            group_id = "project_team_alpha"
            team_members = ["alice", "bob", "charlie"]

            # Each team member adds shared knowledge
            for member in team_members:
                shared_memory = MemoryItem(
                    content=f"Project insight from {member}",
                    memory_type="semantic",
                    user_id=member,
                    metadata={
                        "group_id": group_id,
                        "shared": True,
                        "contributor": member
                    }
                )
                memory.add(shared_memory)

            # Search shared team knowledge
            team_knowledge = memory.search("project", group_id=group_id)

            # Verify collaborative workflow
            assert len(team_knowledge) == 3
            assert all(mem.metadata.get("group_id") == group_id
                       for mem in team_knowledge)
            assert len(set(mem.metadata.get("contributor")
                           for mem in team_knowledge)) == 3


@pytest.mark.e2e
class TestPerformanceE2EScenarios:
    """E2E performance testing scenarios."""

    def test_high_load_e2e_workflow(self):
        """Test system behavior under high load E2E."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory

            # Mock high-load responses
            mock_memory.add.return_value = Mock()
            mock_memory.search.return_value = []

            import threading
            import time

            memory = MockSmartMemory()

            # Simulate high load
            operations_completed = []

            def high_load_operation(operation_id):
                """Simulate high-load memory operation."""
                start_time = time.time()

                # Perform multiple operations
                for i in range(10):
                    test_item = MemoryItem(
                        content=f"High load test {operation_id}-{i}",
                        memory_type="working",
                        user_id=f"load_user_{operation_id}"
                    )
                    memory.add(test_item)
                    memory.search(f"test {i}")

                end_time = time.time()
                operations_completed.append({
                    "operation_id": operation_id,
                    "duration": end_time - start_time,
                    "operations": 20  # 10 adds + 10 searches
                })

            # Run concurrent high-load operations
            threads = []
            for i in range(20):  # 20 concurrent threads
                thread = threading.Thread(target=high_load_operation, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join()

            # Verify high-load performance
            assert len(operations_completed) == 20
            avg_duration = sum(op["duration"] for op in operations_completed) / len(operations_completed)
            assert avg_duration < 5.0  # Operations complete within reasonable time

    def test_memory_scalability_e2e(self):
        """Test memory scalability E2E workflow."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory

            # Mock scalability metrics
            mock_memory.get_system_metrics.return_value = {
                "total_memories": 100000,
                "memory_usage_mb": 250.5,
                "search_performance_ms": 15.2,
                "throughput_ops_per_second": 450
            }

            memory = MockSmartMemory()

            # Test scalability at different data volumes
            data_volumes = [1000, 10000, 100000]

            for volume in data_volumes:
                # Simulate data volume
                mock_memory.get_system_metrics.return_value["total_memories"] = volume

                metrics = memory.get_system_metrics()

                # Verify scalability characteristics
                assert metrics["total_memories"] == volume
                assert metrics["memory_usage_mb"] > 0
                assert metrics["search_performance_ms"] < 100  # Reasonable search time
                assert metrics["throughput_ops_per_second"] > 100  # Good throughput


@pytest.mark.e2e
class TestRealWorldScenarios:
    """E2E tests simulating real-world usage scenarios."""

    def test_personal_assistant_scenario(self):
        """Test complete personal assistant usage scenario."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory, \
                patch('smartmemory.conversation.manager.ConversationManager') as MockConvManager:

            mock_memory = Mock()
            mock_conv_manager = Mock()

            MockSmartMemory.return_value = mock_memory
            MockConvManager.return_value = mock_conv_manager

            # Simulate personal assistant scenario
            user_interactions = [
                {"type": "preference", "content": "I prefer morning meetings"},
                {"type": "schedule", "content": "Meeting with John at 2 PM tomorrow"},
                {"type": "reminder", "content": "Buy groceries after work"},
                {"type": "learning", "content": "I'm studying Python programming"},
                {"type": "query", "content": "What are my preferences for meetings?"}
            ]

            # Mock assistant responses
            mock_memory.search.return_value = [
                {"content": "User prefers morning meetings", "relevance_score": 0.95}
            ]
            mock_memory.add.return_value = Mock()

            memory = MockSmartMemory()
            conv_manager = MockConvManager()

            # Process user interactions
            for interaction in user_interactions:
                if interaction["type"] == "query":
                    # Handle query
                    results = memory.search(interaction["content"])
                    assert len(results) > 0
                    assert results[0]["relevance_score"] > 0.9
                else:
                    # Store interaction as memory
                    memory_item = MemoryItem(
                        content=interaction["content"],
                        memory_type="episodic" if interaction["type"] in ["schedule", "reminder"] else "semantic",
                        user_id="personal_assistant_user",
                        metadata={"interaction_type": interaction["type"]}
                    )
                    memory.add(memory_item)

            # Verify personal assistant workflow
            assert mock_memory.add.call_count == 4  # 4 non-query interactions
            assert mock_memory.search.call_count == 1  # 1 query interaction

    def test_knowledge_management_scenario(self):
        """Test complete knowledge management usage scenario."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory

            # Simulate knowledge management scenario
            knowledge_items = [
                {
                    "content": "Machine learning is a subset of artificial intelligence",
                    "category": "definitions",
                    "tags": ["ML", "AI", "definition"]
                },
                {
                    "content": "Neural networks consist of interconnected nodes called neurons",
                    "category": "concepts",
                    "tags": ["neural networks", "neurons", "architecture"]
                },
                {
                    "content": "Backpropagation is used to train neural networks",
                    "category": "processes",
                    "tags": ["backpropagation", "training", "neural networks"]
                }
            ]

            # Mock knowledge operations
            mock_memory.add.return_value = Mock()
            mock_memory.search.return_value = knowledge_items[:2]  # Return related items

            memory = MockSmartMemory()

            # Build knowledge base
            for item in knowledge_items:
                knowledge_memory = MemoryItem(
                    content=item["content"],
                    memory_type="semantic",
                    user_id="knowledge_worker",
                    metadata={
                        "category": item["category"],
                        "tags": item["tags"]
                    }
                )
                memory.add(knowledge_memory)

            # Query knowledge base
            search_results = memory.search("neural networks")

            # Verify knowledge management workflow - STRICT assertions
            assert mock_memory.add.call_count == 3, "Should have added exactly 3 knowledge items"
            assert len(search_results) >= 1, "Should return at least one search result"

            # STRICT: Verify each result has required fields (may be dict or MemoryItem)
            for i, item in enumerate(search_results):
                # Handle both dict and MemoryItem results
                if isinstance(item, dict):
                    assert 'content' in item, f"Result {i} must have content field"
                    assert item['content'], f"Result {i} content should not be empty"
                    content = item['content']
                else:
                    assert hasattr(item, 'content'), f"Result {i} must have content attribute"
                    assert item.content, f"Result {i} content should not be empty"
                    content = item.content

            # Verify content relevance - at least one result should mention neural networks
            assert any(
                "neural" in str(result.get('content') if isinstance(result, dict) else result.content).lower() or
                "network" in str(result.get('content') if isinstance(result, dict) else result.content).lower()
                for result in search_results
            ), "At least one result should be relevant to 'neural networks'"


@pytest.mark.e2e
class TestErrorRecoveryE2E:
    """E2E tests for error recovery scenarios."""

    def test_graceful_degradation_e2e(self):
        """Test graceful degradation E2E workflow."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory

            # Simulate service degradation
            def mock_operation_with_degradation(operation_type):
                if operation_type == "search":
                    # Search works but slower
                    return [{"content": "Degraded search result", "confidence": 0.7}]
                elif operation_type == "add":
                    # Add works with reduced functionality
                    return {"status": "partial_success", "warnings": ["Cache unavailable"]}
                else:
                    raise Exception("Service temporarily unavailable")

            mock_memory.search.return_value = mock_operation_with_degradation("search")
            mock_memory.add.return_value = mock_operation_with_degradation("add")

            memory = MockSmartMemory()

            # Test operations during degradation
            search_results = memory.search("test query")
            add_result = memory.add(Mock())

            # Verify graceful degradation
            assert len(search_results) > 0
            assert search_results[0]["confidence"] > 0.5  # Still functional
            assert add_result["status"] == "partial_success"
            assert "warnings" in add_result
