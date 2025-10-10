from typing import Dict, List, Optional

try:
    import chromadb  # type: ignore
except Exception:  # pragma: no cover - optional
    chromadb = None  # type: ignore

from .base import VectorBackend


class ChromaVectorBackend(VectorBackend):
    """ChromaDB-backed implementation hidden behind the backend interface."""

    def __init__(self, collection_name: str, persist_directory: Optional[str]):
        if chromadb is None:
            raise RuntimeError("chromadb package is not installed but vector backend=chromadb")
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, *, item_id: str, embedding: List[float], metadata: Dict) -> None:
        self.collection.add(embeddings=[embedding], ids=[str(item_id)], metadatas=[metadata])

    def upsert(self, *, item_id: str, embedding: List[float], metadata: Dict) -> None:
        self.collection.upsert(embeddings=[embedding], ids=[str(item_id)], metadatas=[metadata])

    def search(self, *, query_embedding: List[float], top_k: int) -> List[Dict]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        scores = results.get("distances", [[]])[0] if "distances" in results else None
        out: List[Dict] = []
        for i, id_ in enumerate(ids):
            item = {"id": id_, "metadata": metadatas[i] if i < len(metadatas) else {}}
            if scores and i < len(scores):
                item["score"] = scores[i]
            out.append(item)
        return out

    def clear(self) -> None:
        try:
            all_ids = []
            results = self.collection.get()
            if "ids" in results:
                all_ids = results["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
            # Try to recreate collection for a clean slate
            try:
                name = self.collection.name
                self.client.delete_collection(name)
                self.collection = self.client.create_collection(name)
            except Exception:
                pass
        except Exception as e:
            print(f"Warning: Chroma backend clear encountered error: {e}")
