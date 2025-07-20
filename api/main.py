# api/main.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from api.schemas import QueryInput, QueryResponse, SourceInfo
from api.chat import router as chat_router
from rag.rag_chain import build_rag_chain
from utils.logger import setup_logger

logger = setup_logger()
app = FastAPI(title="RAG Chatbot API")
app.include_router(chat_router, prefix="/api")

@app.post("/query", response_model=QueryResponse)
def ask_question(payload: QueryInput):
    logger.info(f"📩 Received query: {payload.question}")

    filters = {"source_type": {"$eq": payload.source_type}} if payload.source_type else None
    chain = build_rag_chain(filters=filters)
    result = chain(payload.question)

    answer = result["result"]
    sources = []
    seen = set()
    for doc in result["source_documents"]:
        key = (doc.metadata.get("url", ""), doc.metadata.get("title", ""))
        if key not in seen:
            seen.add(key)
            sources.append(SourceInfo(
                title=doc.metadata.get("title", ""),
                url=doc.metadata.get("url", ""),
                source_type=doc.metadata.get("source_type", "")
            ))

    logger.info(f"✅ Answer returned with {len(sources)} source(s)")
    return QueryResponse(answer=answer, sources=sources or [])
