# Backward-compatibility shim — use smartmemory.models.library instead.
# This module re-exports Library as Corpus and maps library_id to corpus_id
# for existing code that imports from this path.
from smartmemory.models.library import Document, Library

# Re-export Library as Corpus for backward compat
Corpus = Library

__all__ = ["Corpus", "Document", "Library"]
