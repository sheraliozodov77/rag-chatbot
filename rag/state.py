# rag/state.py

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from pydantic import BaseModel

class ChatState(BaseModel):
    user_input: str
    chat_history: List[str] = []
    retrieved_docs: Optional[List[Document]] = None
    generated_response: Optional[str] = None
    session_id: Optional[str] = None
    long_term_memory_hits: Optional[List[Document]] = None
    metadata: Optional[Dict] = {}
    retriever: Optional[Any] = None  # Needed for LangGraph
