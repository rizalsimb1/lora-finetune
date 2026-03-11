"""Main RAG pipeline orchestrator."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generator, Optional
from .ingestion import DocumentIngester
from .retriever import HybridRetriever
from .generator import AnswerGenerator


@dataclass
class RAGResult:
    answer: str
    sources: list[dict]
    query: str
    latency_ms: float = 0.0


class RAGSystem:
    """End-to-end RAG system with ingestion, retrieval, and generation."""

    def __init__(
        self,
        vector_store: str = "chroma",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        collection_name: str = "documents",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
    ):
        self.ingester = DocumentIngester(
            vector_store=vector_store,
            embedding_model=embedding_model,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.retriever = HybridRetriever(
            vector_store=self.ingester.vector_store,
            top_k=top_k,
        )
        self.generator = AnswerGenerator(llm_model=llm_model)

    def ingest(self, path: str) -> int:
        """Ingest documents from a local path. Returns number of chunks indexed."""
        return self.ingester.ingest_path(path)

    def ingest_url(self, url: str) -> int:
        """Ingest content from a URL."""
        return self.ingester.ingest_url(url)

    def query(self, question: str) -> RAGResult:
        """Retrieve relevant context and generate a grounded answer."""
        import time
        start = time.perf_counter()
        context_docs = self.retriever.retrieve(question)
        answer = self.generator.generate(question, context_docs)
        latency = (time.perf_counter() - start) * 1000
        sources = [{"doc": d.metadata.get("source", "unknown"),
                    "score": d.metadata.get("score", 0.0)} for d in context_docs]
        return RAGResult(answer=answer, sources=sources, query=question, latency_ms=latency)

    def query_stream(self, question: str) -> Generator[str, None, None]:
        """Stream answer tokens as they are generated."""
        context_docs = self.retriever.retrieve(question)
        yield from self.generator.stream(question, context_docs)
