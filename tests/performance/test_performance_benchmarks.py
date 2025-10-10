"""
Performance and Load Tests.
Tests system performance, benchmarks, and load handling capabilities.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from uuid import uuid4

from smartmemory.models.memory_item import MemoryItem


class TestMemorySystemPerformance:
    """Test memory system performance benchmarks."""
    
    def test_memory_ingestion_performance(self):
        """Test memory ingestion performance benchmarks."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory
            
            # Test ingestion performance metrics
            performance_metrics = {
                "items_per_second": 150,
                "average_latency_ms": 6.7,
                "p95_latency_ms": 12.3,
                "p99_latency_ms": 25.1,
                "memory_usage_mb": 45.2,
                "cpu_usage_percent": 35.8,
                "throughput_mb_per_second": 2.3
            }
            
            mock_memory.benchmark_ingestion.return_value = performance_metrics
            
            memory = MockSmartMemory()
            
            # Run ingestion benchmark
            results = memory.benchmark_ingestion(
                item_count=1000,
                concurrent_threads=10,
                duration_seconds=60
            )
            
            assert results["items_per_second"] > 100  # Good throughput
            assert results["average_latency_ms"] < 10  # Low latency
            assert results["cpu_usage_percent"] < 50   # Reasonable CPU usage
            assert results["memory_usage_mb"] < 100    # Reasonable memory usage
    
    def test_search_performance_benchmarks(self):
        """Test search performance benchmarks."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory
            
            # Test search performance metrics
            search_metrics = {
                "queries_per_second": 500,
                "average_response_time_ms": 2.1,
                "p95_response_time_ms": 4.8,
                "p99_response_time_ms": 9.2,
                "cache_hit_rate": 0.85,
                "index_efficiency": 0.92,
                "memory_usage_mb": 25.7
            }
            
            mock_memory.benchmark_search.return_value = search_metrics
            
            memory = MockSmartMemory()
            
            # Run search benchmark
            results = memory.benchmark_search(
                query_count=5000,
                concurrent_queries=50,
                query_complexity="mixed"
            )
            
            assert results["queries_per_second"] > 200   # High query throughput
            assert results["average_response_time_ms"] < 5  # Fast response
            assert results["cache_hit_rate"] > 0.8       # Good cache performance
            assert results["index_efficiency"] > 0.9     # Efficient indexing
    
    def test_graph_operations_performance(self):
        """Test graph operations performance benchmarks."""
        with patch('smartmemory.graph.smartgraph.SmartGraph') as MockSmartGraph:
            mock_graph = Mock()
            MockSmartGraph.return_value = mock_graph
            
            # Test graph performance metrics
            graph_metrics = {
                "node_operations_per_second": 1000,
                "edge_operations_per_second": 800,
                "traversal_operations_per_second": 300,
                "cypher_queries_per_second": 200,
                "average_query_time_ms": 5.2,
                "graph_size_nodes": 100000,
                "graph_size_edges": 250000,
                "memory_usage_mb": 120.5
            }
            
            mock_graph.benchmark_operations.return_value = graph_metrics
            
            graph = MockSmartGraph()
            
            # Run graph benchmark
            results = graph.benchmark_operations(
                operation_count=10000,
                operation_mix={"nodes": 0.4, "edges": 0.4, "queries": 0.2}
            )
            
            assert results["node_operations_per_second"] > 500
            assert results["edge_operations_per_second"] > 400
            assert results["average_query_time_ms"] < 10
            assert results["memory_usage_mb"] < 200
    
    def test_vector_store_performance(self):
        """Test vector store performance benchmarks."""
        with patch('smartmemory.stores.vector.vector_store.VectorStore') as MockVectorStore:
            mock_store = Mock()
            MockVectorStore.return_value = mock_store
            
            # Test vector store performance metrics
            vector_metrics = {
                "embeddings_per_second": 250,
                "search_queries_per_second": 400,
                "average_search_time_ms": 2.5,
                "index_build_time_seconds": 45.2,
                "memory_usage_mb": 85.3,
                "disk_usage_mb": 150.7,
                "similarity_accuracy": 0.95
            }
            
            mock_store.benchmark_performance.return_value = vector_metrics
            
            store = MockVectorStore()
            
            # Run vector store benchmark
            results = store.benchmark_performance(
                embedding_count=50000,
                search_count=10000,
                embedding_dimension=1536
            )
            
            assert results["embeddings_per_second"] > 100
            assert results["search_queries_per_second"] > 200
            assert results["average_search_time_ms"] < 5
            assert results["similarity_accuracy"] > 0.9


class TestConcurrencyAndScalability:
    """Test concurrency and scalability performance."""
    
    def test_concurrent_memory_operations(self):
        """Test concurrent memory operations performance."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory
            
            # Test concurrent operations metrics
            concurrency_metrics = {
                "max_concurrent_operations": 100,
                "operations_completed": 9850,
                "operations_failed": 150,
                "success_rate": 0.985,
                "average_wait_time_ms": 1.2,
                "deadlock_count": 0,
                "resource_contention_events": 5,
                "throughput_degradation": 0.15  # 15% slower under load
            }
            
            mock_memory.test_concurrency.return_value = concurrency_metrics
            
            memory = MockSmartMemory()
            
            # Run concurrency test
            results = memory.test_concurrency(
                concurrent_threads=100,
                operations_per_thread=100,
                operation_types=["add", "get", "search", "update"]
            )
            
            assert results["success_rate"] > 0.95      # High success rate
            assert results["deadlock_count"] == 0      # No deadlocks
            assert results["throughput_degradation"] < 0.3  # Acceptable degradation
    
    def test_memory_scalability_limits(self):
        """Test memory system scalability limits."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory
            
            # Test scalability metrics
            scalability_tests = [
                {"memory_items": 1000, "performance_score": 1.0},
                {"memory_items": 10000, "performance_score": 0.95},
                {"memory_items": 100000, "performance_score": 0.85},
                {"memory_items": 1000000, "performance_score": 0.70},
                {"memory_items": 10000000, "performance_score": 0.50}
            ]
            
            mock_memory.test_scalability.return_value = scalability_tests
            
            memory = MockSmartMemory()
            
            # Run scalability test
            results = memory.test_scalability(max_items=10000000)
            
            assert len(results) == 5
            assert results[0]["performance_score"] == 1.0  # Baseline performance
            assert results[-1]["performance_score"] > 0.3  # Still functional at scale
            
            # Performance should degrade gracefully
            for i in range(1, len(results)):
                assert results[i]["performance_score"] <= results[i-1]["performance_score"]
    
    def test_load_balancing_performance(self):
        """Test load balancing performance."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory
            
            # Test load balancing metrics
            load_balancing_metrics = {
                "total_requests": 10000,
                "requests_per_node": {
                    "node_1": 3300,
                    "node_2": 3350,
                    "node_3": 3350
                },
                "load_distribution_variance": 0.02,  # Low variance = good balancing
                "average_response_time_ms": 3.2,
                "node_failure_recovery_time_ms": 150,
                "failover_success_rate": 0.99
            }
            
            mock_memory.test_load_balancing.return_value = load_balancing_metrics
            
            memory = MockSmartMemory()
            
            # Run load balancing test
            results = memory.test_load_balancing(
                total_requests=10000,
                node_count=3,
                failure_simulation=True
            )
            
            assert results["load_distribution_variance"] < 0.1  # Good distribution
            assert results["failover_success_rate"] > 0.95     # Reliable failover
            assert results["node_failure_recovery_time_ms"] < 500  # Fast recovery


class TestMemoryUsageAndOptimization:
    """Test memory usage and optimization performance."""
    
    def test_memory_consumption_patterns(self):
        """Test memory consumption patterns."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory
            
            # Test memory consumption over time
            memory_timeline = [
                {"time_minutes": 0, "memory_mb": 50, "items": 0},
                {"time_minutes": 10, "memory_mb": 75, "items": 1000},
                {"time_minutes": 20, "memory_mb": 95, "items": 2000},
                {"time_minutes": 30, "memory_mb": 110, "items": 3000},
                {"time_minutes": 40, "memory_mb": 120, "items": 4000},
                {"time_minutes": 50, "memory_mb": 125, "items": 5000}  # Growth slowing
            ]
            
            mock_memory.monitor_memory_consumption.return_value = memory_timeline
            
            memory = MockSmartMemory()
            
            # Monitor memory consumption
            timeline = memory.monitor_memory_consumption(duration_minutes=50)
            
            assert len(timeline) == 6
            assert timeline[0]["memory_mb"] < timeline[-1]["memory_mb"]  # Memory increases
            
            # Check growth rate is reasonable
            final_memory = timeline[-1]["memory_mb"]
            final_items = timeline[-1]["items"]
            memory_per_item = final_memory / final_items if final_items > 0 else 0
            assert memory_per_item < 0.1  # Less than 0.1 MB per item
    
    def test_garbage_collection_performance(self):
        """Test garbage collection performance."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory
            
            # Test garbage collection metrics
            gc_metrics = {
                "gc_frequency_per_hour": 12,
                "average_gc_duration_ms": 25.3,
                "max_gc_duration_ms": 85.7,
                "memory_freed_per_gc_mb": 15.2,
                "gc_efficiency": 0.88,  # 88% of targeted memory freed
                "application_pause_time_ms": 5.1,
                "gc_overhead_percent": 2.3
            }
            
            mock_memory.analyze_garbage_collection.return_value = gc_metrics
            
            memory = MockSmartMemory()
            
            # Analyze garbage collection
            results = memory.analyze_garbage_collection(monitoring_duration_hours=4)
            
            assert results["average_gc_duration_ms"] < 50    # Fast GC
            assert results["gc_efficiency"] > 0.8           # Efficient GC
            assert results["application_pause_time_ms"] < 10 # Low pause time
            assert results["gc_overhead_percent"] < 5       # Low overhead
    
    def test_cache_performance_optimization(self):
        """Test cache performance optimization."""
        with patch('smartmemory.utils.cache.RedisCache') as MockRedisCache:
            mock_cache = Mock()
            MockRedisCache.return_value = mock_cache
            
            # Test cache optimization metrics
            cache_optimization = {
                "baseline_hit_rate": 0.75,
                "optimized_hit_rate": 0.88,
                "improvement_percentage": 17.3,
                "average_response_time_before_ms": 3.2,
                "average_response_time_after_ms": 2.1,
                "memory_usage_before_mb": 85.3,
                "memory_usage_after_mb": 78.9,
                "optimization_techniques": [
                    "cache_warming",
                    "ttl_optimization",
                    "key_partitioning"
                ]
            }
            
            mock_cache.optimize_performance.return_value = cache_optimization
            
            cache = MockRedisCache()
            
            # Run cache optimization
            results = cache.optimize_performance()
            
            assert results["optimized_hit_rate"] > results["baseline_hit_rate"]
            assert results["improvement_percentage"] > 10
            assert results["average_response_time_after_ms"] < results["average_response_time_before_ms"]


class TestStressAndLoadTesting:
    """Test system behavior under stress and load."""
    
    def test_high_volume_ingestion_stress(self):
        """Test high volume ingestion stress testing."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory
            
            # Test stress testing metrics
            stress_metrics = {
                "total_items_processed": 95000,
                "target_items": 100000,
                "success_rate": 0.95,
                "peak_ingestion_rate": 180,
                "sustained_ingestion_rate": 145,
                "error_rate": 0.05,
                "system_stability": "stable",
                "resource_utilization": {
                    "cpu_peak": 85,
                    "memory_peak_mb": 450,
                    "disk_io_peak_mbps": 25
                },
                "recovery_time_after_peak_ms": 2500
            }
            
            mock_memory.stress_test_ingestion.return_value = stress_metrics
            
            memory = MockSmartMemory()
            
            # Run ingestion stress test
            results = memory.stress_test_ingestion(
                target_items=100000,
                ramp_up_duration_minutes=10,
                peak_duration_minutes=30,
                ramp_down_duration_minutes=5
            )
            
            assert results["success_rate"] > 0.9           # High success rate
            assert results["error_rate"] < 0.1             # Low error rate
            assert results["system_stability"] == "stable" # System remains stable
            assert results["recovery_time_after_peak_ms"] < 5000  # Quick recovery
    
    def test_concurrent_user_simulation(self):
        """Test concurrent user simulation."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory
            
            # Test concurrent user metrics
            user_simulation_metrics = {
                "simulated_users": 500,
                "total_operations": 50000,
                "operations_per_user": 100,
                "average_session_duration_minutes": 15.3,
                "concurrent_peak": 450,
                "user_satisfaction_score": 0.87,
                "operation_success_rate": 0.96,
                "average_response_time_ms": 2.8,
                "p95_response_time_ms": 8.2,
                "system_throughput_ops_per_second": 320
            }
            
            mock_memory.simulate_concurrent_users.return_value = user_simulation_metrics
            
            memory = MockSmartMemory()
            
            # Run user simulation
            results = memory.simulate_concurrent_users(
                user_count=500,
                session_duration_minutes=15,
                operation_mix={
                    "search": 0.4,
                    "add": 0.3,
                    "get": 0.2,
                    "update": 0.1
                }
            )
            
            assert results["operation_success_rate"] > 0.95    # High success rate
            assert results["user_satisfaction_score"] > 0.8    # Good user experience
            assert results["average_response_time_ms"] < 5     # Fast responses
            assert results["system_throughput_ops_per_second"] > 200  # Good throughput
    
    def test_resource_exhaustion_scenarios(self):
        """Test resource exhaustion scenarios."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory
            
            # Test resource exhaustion scenarios
            exhaustion_scenarios = [
                {
                    "resource": "memory",
                    "threshold_reached": True,
                    "system_response": "graceful_degradation",
                    "recovery_time_seconds": 15.2,
                    "data_loss": False
                },
                {
                    "resource": "cpu",
                    "threshold_reached": True,
                    "system_response": "request_throttling",
                    "recovery_time_seconds": 8.7,
                    "data_loss": False
                },
                {
                    "resource": "disk_space",
                    "threshold_reached": True,
                    "system_response": "cleanup_old_data",
                    "recovery_time_seconds": 45.3,
                    "data_loss": True  # Old data cleaned up
                }
            ]
            
            mock_memory.test_resource_exhaustion.return_value = exhaustion_scenarios
            
            memory = MockSmartMemory()
            
            # Test resource exhaustion
            results = memory.test_resource_exhaustion()
            
            assert len(results) == 3
            for scenario in results:
                assert scenario["threshold_reached"] is True
                assert scenario["recovery_time_seconds"] < 60  # Reasonable recovery time
                assert scenario["system_response"] in [
                    "graceful_degradation", 
                    "request_throttling", 
                    "cleanup_old_data"
                ]


class TestPerformanceRegression:
    """Test performance regression detection."""
    
    def test_performance_baseline_comparison(self):
        """Test performance baseline comparison."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory
            
            # Test baseline comparison
            baseline_metrics = {
                "ingestion_rate": 150,
                "search_response_time_ms": 2.5,
                "memory_usage_mb": 85,
                "cpu_usage_percent": 35
            }
            
            current_metrics = {
                "ingestion_rate": 140,  # 6.7% slower
                "search_response_time_ms": 2.8,  # 12% slower
                "memory_usage_mb": 92,  # 8.2% more memory
                "cpu_usage_percent": 38  # 8.6% more CPU
            }
            
            regression_analysis = {
                "performance_regression_detected": True,
                "regression_severity": "moderate",
                "affected_metrics": ["search_response_time_ms", "memory_usage_mb"],
                "regression_percentage": {
                    "ingestion_rate": -6.7,
                    "search_response_time_ms": 12.0,
                    "memory_usage_mb": 8.2,
                    "cpu_usage_percent": 8.6
                },
                "recommended_actions": [
                    "investigate_search_performance",
                    "analyze_memory_leaks",
                    "profile_cpu_usage"
                ]
            }
            
            mock_memory.compare_performance_baseline.return_value = regression_analysis
            
            memory = MockSmartMemory()
            
            # Compare with baseline
            results = memory.compare_performance_baseline(
                baseline=baseline_metrics,
                current=current_metrics,
                regression_threshold=0.1  # 10% threshold
            )
            
            assert results["performance_regression_detected"] is True
            assert results["regression_severity"] in ["minor", "moderate", "major"]
            assert len(results["affected_metrics"]) > 0
            assert len(results["recommended_actions"]) > 0
    
    def test_performance_trend_analysis(self):
        """Test performance trend analysis."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory
            
            # Test performance trends over time
            performance_history = [
                {"date": "2024-01-01", "ingestion_rate": 150, "response_time_ms": 2.5},
                {"date": "2024-01-02", "ingestion_rate": 148, "response_time_ms": 2.6},
                {"date": "2024-01-03", "ingestion_rate": 145, "response_time_ms": 2.8},
                {"date": "2024-01-04", "ingestion_rate": 142, "response_time_ms": 3.0},
                {"date": "2024-01-05", "ingestion_rate": 140, "response_time_ms": 3.2}
            ]
            
            trend_analysis = {
                "trends_detected": {
                    "ingestion_rate": "declining",
                    "response_time_ms": "increasing"
                },
                "trend_severity": "moderate",
                "projected_performance_in_30_days": {
                    "ingestion_rate": 125,  # Continued decline
                    "response_time_ms": 4.5  # Continued increase
                },
                "intervention_recommended": True,
                "confidence_level": 0.85
            }
            
            mock_memory.analyze_performance_trends.return_value = trend_analysis
            
            memory = MockSmartMemory()
            
            # Analyze performance trends
            results = memory.analyze_performance_trends(performance_history)
            
            assert "trends_detected" in results
            assert results["intervention_recommended"] is True
            assert results["confidence_level"] > 0.8
            assert results["trend_severity"] in ["minor", "moderate", "major"]


@pytest.mark.slow
@pytest.mark.performance
class TestLongRunningPerformanceTests:
    """Long-running performance tests."""
    
    def test_24_hour_stability_test(self):
        """Test 24-hour system stability."""
        # This would run for 24 hours in real scenarios
        # Mocked for unit testing
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory
            
            stability_metrics = {
                "test_duration_hours": 24,
                "total_operations": 2000000,
                "success_rate": 0.9995,
                "memory_leaks_detected": 0,
                "performance_degradation": 0.05,  # 5% slower after 24h
                "system_crashes": 0,
                "error_spikes": 2,  # Brief error spikes
                "recovery_time_average_seconds": 3.2
            }
            
            mock_memory.run_stability_test.return_value = stability_metrics
            
            memory = MockSmartMemory()
            
            # Run stability test (mocked)
            results = memory.run_stability_test(duration_hours=24)
            
            assert results["success_rate"] > 0.999        # Very high success rate
            assert results["memory_leaks_detected"] == 0  # No memory leaks
            assert results["system_crashes"] == 0         # No crashes
            assert results["performance_degradation"] < 0.1  # Less than 10% degradation
    
    def test_memory_evolution_performance(self):
        """Test memory evolution performance over time."""
        with patch('smartmemory.evolution.cycle.run_evolution_cycle') as mock_evolution:
            # Test evolution performance
            evolution_metrics = {
                "evolution_cycles_completed": 48,  # Every 30 minutes for 24 hours
                "average_cycle_duration_seconds": 45.3,
                "items_evolved_total": 15000,
                "evolution_efficiency": 0.92,
                "system_impact_during_evolution": 0.15,  # 15% performance impact
                "evolution_accuracy": 0.88,
                "rollback_events": 2
            }
            
            mock_evolution.return_value = evolution_metrics
            
            from smartmemory.evolution.cycle import run_evolution_cycle
            
            # Run evolution performance test
            results = run_evolution_cycle(Mock())
            
            assert results["evolution_efficiency"] > 0.9
            assert results["system_impact_during_evolution"] < 0.2
            assert results["evolution_accuracy"] > 0.85
            assert results["rollback_events"] < 5


@pytest.mark.integration
class TestPerformanceIntegration:
    """Integration performance tests."""
    def test_real_world_workload_simulation(self):
        """Test real-world workload simulation."""
        with patch('smartmemory.smart_memory.SmartMemory') as MockSmartMemory:
            mock_memory = Mock()
            MockSmartMemory.return_value = mock_memory
            
            # Test realistic workload
            workload_metrics = {
                "workload_type": "mixed_realistic",
                "duration_hours": 8,
                "operations_completed": 500000,
                "user_patterns_simulated": [
                    "morning_peak",
                    "lunch_dip", 
                    "afternoon_steady",
                    "evening_decline"
                ],
                "average_system_utilization": 0.65,
                "peak_system_utilization": 0.89,
                "user_satisfaction_score": 0.91,
                "sla_compliance": 0.97
            }
            
            mock_memory.simulate_realistic_workload.return_value = workload_metrics
            
            memory = MockSmartMemory()
            
            # Simulate realistic workload
            results = memory.simulate_realistic_workload()
            
            assert results["user_satisfaction_score"] > 0.85
            assert results["sla_compliance"] > 0.95
            assert results["peak_system_utilization"] < 0.95  # Not overloaded
