"""FastAPI REST API for the RAG system."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import sys; sys.path.insert(0, "..")
from src.pipeline import RAGSystem

app = FastAPI(title="RAG API", version="0.1.0")
rag = RAGSystem()


class QueryRequest(BaseModel):
    question: str
    stream: bool = False


@app.post("/query")
async def query(req: QueryRequest):
    try:
        if req.stream:
            return StreamingResponse(
                rag.query_stream(req.question),
                media_type="text/plain",
            )
        result = rag.query(req.question)
        return {"answer": result.answer, "sources": result.sources,
                "latency_ms": result.latency_ms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest(path: str):
    count = rag.ingest(path)
    return {"indexed_chunks": count, "path": path}


@app.get("/health")
async def health():
    return {"status": "ok"}
