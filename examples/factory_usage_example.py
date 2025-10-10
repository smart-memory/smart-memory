"""
Example demonstrating unified factory pattern usage for memory and store creation.

Shows how to use MemoryFactory and StoreFactory for consistent initialization.
"""

import logging
from smartmemory.configuration import MemoryConfig
from smartmemory.memory.memory_factory import MemoryFactory, MemoryType
from smartmemory.stores.factory import StoreFactory, StoreType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_memory_factory():
    """Demonstrate MemoryFactory usage."""
    print("=== MemoryFactory Demonstration ===")

    # Get configuration
    try:
        config = MemoryConfig()
        print(f"✅ Loaded configuration successfully")
    except Exception as e:
        print(f"⚠️ Could not load config: {e}")
        config = None

    # Show registered memory types
    registered_types = MemoryFactory.get_registered_types()
    print(f"📋 Registered memory types: {[t.value for t in registered_types]}")

    # Create individual memory instances
    print("\n🏭 Creating individual memory instances:")

    for memory_type in registered_types:
        try:
            memory = MemoryFactory.create_memory(memory_type, config)
            print(f"✅ Created {memory_type.value} memory: {type(memory).__name__}")
        except Exception as e:
            print(f"❌ Failed to create {memory_type.value} memory: {e}")

    # Test convenience methods
    print("\n🎯 Testing convenience methods:")
    try:
        semantic = MemoryFactory.create_semantic_memory(config)
        print(f"✅ Semantic memory via convenience method: {type(semantic).__name__}")
    except Exception as e:
        print(f"❌ Semantic memory convenience method failed: {e}")

    # Test batch creation
    print("\n📦 Testing batch creation:")
    try:
        all_memories = MemoryFactory.create_all_memory_types(config)
        print(f"✅ Created {len(all_memories)} memory instances in batch")
        for mem_type, memory in all_memories.items():
            print(f"   - {mem_type.value}: {type(memory).__name__}")
    except Exception as e:
        print(f"❌ Batch creation failed: {e}")


def demonstrate_store_factory():
    """Demonstrate StoreFactory usage."""
    print("\n=== StoreFactory Demonstration ===")

    # Get configuration
    try:
        config = MemoryConfig()
        print(f"✅ Loaded configuration successfully")
    except Exception as e:
        print(f"⚠️ Could not load config: {e}")
        config = None

    # Show registered store types
    registered_types = StoreFactory.get_registered_types()
    print(f"📋 Registered store types: {[t.value for t in registered_types]}")

    # Create individual store instances
    print("\n🏭 Creating individual store instances:")

    for store_type in registered_types:
        try:
            store = StoreFactory.create_store(store_type, config)
            print(f"✅ Created {store_type.value} store: {type(store).__name__}")
        except Exception as e:
            print(f"❌ Failed to create {store_type.value} store: {e}")

    # Test convenience methods for graph stores
    print("\n🎯 Testing graph store convenience methods:")
    graph_methods = [
        ("semantic_graph", StoreFactory.create_semantic_graph),
        ("episodic_graph", StoreFactory.create_episodic_graph),
        ("procedural_graph", StoreFactory.create_procedural_graph),
        ("zettel_graph", StoreFactory.create_zettel_graph)
    ]

    for name, method in graph_methods:
        try:
            store = method(config)
            print(f"✅ {name} via convenience method: {type(store).__name__}")
        except Exception as e:
            print(f"❌ {name} convenience method failed: {e}")

    # Test batch creation of graph stores
    print("\n📦 Testing batch graph store creation:")
    try:
        all_graph_stores = StoreFactory.create_all_graph_stores(config)
        print(f"✅ Created {len(all_graph_stores)} graph store instances in batch")
        for store_type, store in all_graph_stores.items():
            print(f"   - {store_type.value}: {type(store).__name__}")
    except Exception as e:
        print(f"❌ Batch graph store creation failed: {e}")


def demonstrate_factory_validation():
    """Demonstrate factory validation utilities (optional)."""
    print("\n=== Factory Validation Demonstration ===")

    try:
        # FactoryValidator may not be present in current build; make optional
        from smartmemory.memory.models import FactoryValidator  # type: ignore
    except Exception as e:
        print(f"⚠️ FactoryValidator not available: {e}")
        print("🔍 Basic validation via registry checks:")
        print(f"   - Registered memory types: {[t.value for t in MemoryFactory.get_registered_types()]}")
        print(f"   - Registered store types: {[t.value for t in StoreFactory.get_registered_types()]}")
        return

    # Validate memory factory registration
    memory_valid = FactoryValidator.validate_memory_factory_registration()
    print(f"🔍 Memory factory registration valid: {memory_valid}")

    # Validate store factory registration  
    store_valid = FactoryValidator.validate_store_factory_registration()
    print(f"🔍 Store factory registration valid: {store_valid}")

    # Test individual memory type creation validation
    print("\n🧪 Testing individual memory type validation:")
    try:
        config = MemoryConfig()
    except Exception:
        config = None

    for memory_type in MemoryFactory.get_registered_types():
        valid = FactoryValidator.validate_factory_creation(memory_type, config)
        print(f"   - {memory_type.value}: {'✅' if valid else '❌'}")


if __name__ == "__main__":
    print("🚀 Factory Pattern Integration Example")
    print("=" * 50)

    try:
        demonstrate_memory_factory()
        demonstrate_store_factory()
        demonstrate_factory_validation()

        print("\n🎉 Factory demonstration completed!")
        print("The unified factory pattern provides:")
        print("  ✅ Consistent initialization across all memory/store types")
        print("  ✅ Centralized configuration management")
        print("  ✅ Easy testing with mock factories")
        print("  ✅ Extensible registration system for new types")

    except Exception as e:
        print(f"\n💥 Factory demonstration failed: {e}")
        import traceback

        traceback.print_exc()
