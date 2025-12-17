from typing import Any, Dict, List, Optional, Union

from smartmemory.models.corpus import Corpus, Document
from smartmemory.stores.base import BaseHandler
from smartmemory.stores.persistence.base import PersistenceBackend


class CorpusStore(BaseHandler[Corpus]):
    """
    Store for managing Corpus metadata/Lifecycle.
    Uses a PersistenceBackend to save Corpus objects.
    """

    def __init__(self, persistence: PersistenceBackend[Corpus]):
        self.persistence = persistence

    def add(self, item: Corpus, **kwargs) -> Union[str, Corpus, None]:
        """
        Register a new Corpus.
        """
        if not isinstance(item, Corpus):
             raise TypeError("CorpusStore only accepts Corpus objects")
        
        # Save to persistence
        self.persistence.save(item)
        return item.id

    def update(self, item: Corpus, **kwargs) -> Union[str, Corpus, None]:
        """
        Update an existing Corpus.
        """
        if not isinstance(item, Corpus):
             raise TypeError("CorpusStore only accepts Corpus objects")
        
        self.persistence.save(item)
        return item.id

    def get(self, item_id: str, **kwargs) -> Union[Corpus, None]:
        """
        Retrieve a Corpus by ID.
        """
        return self.persistence.find_one(Corpus, id=item_id)

    def delete(self, item_id: str, **kwargs) -> bool:
        """
        Delete a Corpus.
        """
        return self.persistence.delete_one(Corpus, id=item_id)

    def search(self, query: Any, **kwargs) -> List[Corpus]:
        """
        List/Search corpora.
        """
        return self.persistence.find_many(Corpus)


class DocumentStore(BaseHandler[Document]):
    """
    Store for managing Document metadata.
    """

    def __init__(self, persistence: PersistenceBackend[Document]):
        self.persistence = persistence

    def add(self, item: Document, **kwargs) -> Union[str, Document, None]:
        if not isinstance(item, Document):
             raise TypeError("DocumentStore only accepts Document objects")
        self.persistence.save(item)
        return item.id

    def update(self, item: Document, **kwargs) -> Union[str, Document, None]:
        if not isinstance(item, Document):
             raise TypeError("DocumentStore only accepts Document objects")
        self.persistence.save(item)
        return item.id

    def get(self, item_id: str, **kwargs) -> Union[Document, None]:
        return self.persistence.find_one(Document, id=item_id)

    def delete(self, item_id: str, **kwargs) -> bool:
        return self.persistence.delete_one(Document, id=item_id)

    def search(self, query: Any, **kwargs) -> List[Document]:
        # Support filtering by corpus_id via kwargs or query
        filters = {}
        if isinstance(query, dict):
            filters.update(query)
        if 'corpus_id' in kwargs:
            filters['corpus_id'] = kwargs['corpus_id']
            
        return self.persistence.find_many(Document, **filters)
