# api/schemas.py

from pydantic import BaseModel
from typing import Optional, List


class QueryInput(BaseModel):
    question: str
    source_type: Optional[str] = None


class SourceInfo(BaseModel):
    title: str
    url: str
    source_type: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]


class ChatRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    history: List[str]
    long_term_memory_used: bool
    sources: List[SourceInfo]
