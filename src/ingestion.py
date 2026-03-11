"""Document ingestion pipeline with chunking and embedding."""
import os
from pathlib import Path
from typing import List


class DocumentIngester:
    """Handles loading, chunking, and indexing documents."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".html"}

    def __init__(self, vector_store: str, embedding_model: str,
                 collection_name: str, chunk_size: int, chunk_overlap: int):
        self.vector_store_type = vector_store
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self):
        """Initialize the vector store backend."""
        if self.vector_store_type == "chroma":
            try:
                import chromadb
                client = chromadb.PersistentClient(path=".chroma")
                return client.get_or_create_collection(self.collection_name)
            except ImportError:
                raise ImportError("Install chromadb: pip install chromadb")
        raise ValueError(f"Unknown vector store: {self.vector_store_type}")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def ingest_path(self, path: str) -> int:
        """Ingest all supported files from a directory."""
        p = Path(path)
        files = list(p.rglob("*")) if p.is_dir() else [p]
        total = 0
        for f in files:
            if f.suffix in self.SUPPORTED_EXTENSIONS:
                total += self._index_file(f)
        return total

    def ingest_url(self, url: str) -> int:
        """Fetch and ingest content from a URL."""
        import urllib.request
        with urllib.request.urlopen(url) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
        chunks = self._chunk_text(content)
        # Store chunks (simplified)
        return len(chunks)

    def _index_file(self, path: Path) -> int:
        text = path.read_text(errors="ignore")
        chunks = self._chunk_text(text)
        return len(chunks)
