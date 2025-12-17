import sys
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from abc import ABC, abstractmethod
from typing import TypeVar, Optional, List, Generic

# ----------------------------------------------------------------------
# 1. SETUP MOCKS BEFORE IMPORTS
# ----------------------------------------------------------------------
# Mock heavy dependencies that might be missing in execution env
sys.modules["fastapi"] = MagicMock()
sys.modules["fastapi.security"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["falkordb"] = MagicMock()
sys.modules["redis"] = MagicMock()

# Mock registry to prevent loading all stores
registry_mock = MagicMock()
sys.modules["smartmemory.stores.registry"] = registry_mock

# Mock MemoryItem to avoid deep dependencies in smartmemory
memory_item_mock = MagicMock()
class MockMemoryItem:
    def __init__(self, content=None, item_id=None, metadata=None):
        self.content = content
        self.item_id = item_id
        self.metadata = metadata or {}
memory_item_mock.MemoryItem = MockMemoryItem
sys.modules["smartmemory.models.memory_item"] = memory_item_mock

# Mock Corpus/Document to avoid service_common dependencies
corpus_mock = MagicMock()
from pydantic import BaseModel
class MockCorpus(BaseModel):
    id: str = "c1"
    name: str
    registry_id: str
    description: str = ""
class MockDocument(BaseModel):
    id: str = "d1"
    name: str
    corpus_id: str
corpus_mock.Corpus = MockCorpus
corpus_mock.Document = MockDocument
sys.modules["smartmemory.models.corpus"] = corpus_mock

# Mock Persistence Layer
TItem = TypeVar("TItem")
class PersistenceBackend(ABC, Generic[TItem]):
    @abstractmethod
    def save(self, item: TItem, **kwargs) -> Optional[str]: ...
    @abstractmethod
    def find(self, item_id: str, **kwargs) -> Optional[TItem]: ...
    @abstractmethod
    def delete(self, item_id: str, **kwargs) -> bool: ...

persistence_mock = MagicMock()
base_mock = MagicMock()
base_mock.PersistenceBackend = PersistenceBackend
sys.modules["smartmemory.stores.persistence"] = persistence_mock
sys.modules["smartmemory.stores.persistence.base"] = base_mock

# ----------------------------------------------------------------------
# 2. IMPORTS (Now safe)
# ----------------------------------------------------------------------
# Ensure path allows importing content under test
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.append(str(project_root))

from smartmemory.stores.corpus.store import CorpusStore
from smartmemory.stores.external.file_handler import FileHandler
from smartmemory.stores.json_store import JSONStore

# ----------------------------------------------------------------------
# 3. ADAPTER & TEST
# ----------------------------------------------------------------------

class JsonPersistenceAdapter(PersistenceBackend[MockCorpus]):
    def __init__(self, json_store: JSONStore):
        self.store = json_store

    def save(self, item: MockCorpus, **kwargs) -> str:
        self.store.add(item.model_dump())
        return item.id

    def find_one(self, model_class, **kwargs) -> MockCorpus:
        # Mock adapter logic: naively assume kwargs['id'] exists
        item_id = kwargs.get('id')
        data = self.store.get(item_id)
        if data:
            return MockCorpus(**data)
        return None
    
    # unused
    def find(self, item_id: str, **kwargs) -> MockCorpus: return None

    def delete_one(self, model_class, **kwargs) -> bool:
        item_id = kwargs.get('id')
        return self.store.delete(item_id)
    
    # unused
    def delete(self, item_id: str, **kwargs) -> bool: return False
    
    def find_many(self, model_class, **kwargs):
        return []

def test_isolated_components():
    print("Starting Isolated Verification...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # --- Test FileHandler ---
        print("Testing FileHandler...")
        fh = FileHandler(base_path=str(temp_path / "files"))
        
        item = MockMemoryItem(
            content="Hello World",
            item_id="test.txt",
            metadata={"uri": None}
        )
        
        uri = fh.add(item)
        print(f"File saved: {uri}")
        assert uri.startswith("file://")
        
        retrieved = fh.get(uri)
        assert retrieved['content'] == "Hello World"
        print("FileHandler Verified.")
        
        # --- Test CorpusStore ---
        print("Testing CorpusStore...")
        json_store = JSONStore(data_dir=str(temp_path / "meta"))
        adapter = JsonPersistenceAdapter(json_store)
        corpus_store = CorpusStore(persistence=adapter)
        
        corpus = MockCorpus(name="Test Corpus", registry_id="reg_1")
        c_id = corpus_store.add(corpus)
        assert c_id == corpus.id
        
        fetched = corpus_store.get(c_id)
        assert fetched.name == "Test Corpus"
        print("CorpusStore Verified.")
        
        print("SUCCESS: Components verified in isolation.")

if __name__ == "__main__":
    test_isolated_components()
