"""Hybrid retriever combining semantic and keyword search."""
from typing import Any, List
import math


class Document:
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}


class HybridRetriever:
    """Combines dense vector search with BM25 sparse retrieval."""

    def __init__(self, vector_store, top_k: int = 5, alpha: float = 0.7):
        self.vector_store = vector_store
        self.top_k = top_k
        self.alpha = alpha  # weight for semantic vs keyword

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve top-k most relevant documents for a query."""
        semantic_docs = self._semantic_search(query)
        keyword_docs = self._keyword_search(query)
        return self._reciprocal_rank_fusion(semantic_docs, keyword_docs)

    def _semantic_search(self, query: str) -> List[Document]:
        """Dense vector similarity search."""
        # Placeholder — in production this calls embedding model + vector DB
        return [Document(f"Semantic result for: {query}", {"source": "vector_db", "score": 0.9})]

    def _keyword_search(self, query: str) -> List[Document]:
        """BM25 keyword-based search."""
        return [Document(f"Keyword result for: {query}", {"source": "bm25", "score": 0.7})]

    def _reciprocal_rank_fusion(self, lists: list, *more) -> List[Document]:
        """Combine multiple ranked lists using RRF."""
        scores = {}
        for rank_list in [lists] + list(more):
            for rank, doc in enumerate(rank_list):
                key = doc.content[:50]
                scores[key] = scores.get(key, 0) + 1 / (60 + rank + 1)
        return sorted(
            [Document(k, {"score": v}) for k, v in scores.items()],
            key=lambda d: d.metadata["score"], reverse=True
        )[:self.top_k]
