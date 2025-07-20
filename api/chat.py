# api/chat.py

from fastapi import APIRouter
from api.schemas import ChatRequest, ChatResponse, SourceInfo
from rag.state import ChatState
from rag.langgraph_graph import get_langgraph
from uuid import uuid4
from utils.logger import setup_logger

logger = setup_logger()
router = APIRouter()
graph = get_langgraph()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    session_id = payload.session_id or str(uuid4())
    logger.info(f"🧠 New chat session: {session_id}")

    state = ChatState(user_input=payload.user_input, session_id=session_id)

    try:
        result_dict = graph.invoke(state)
        logger.info(f"🧠 LangGraph response received")

        result_state = ChatState(**result_dict) if isinstance(result_dict, dict) else result_dict

        if not result_state.generated_response:
            return ChatResponse(
                session_id=session_id,
                response="❌ LangGraph did not return a valid result.",
                history=[],
                long_term_memory_used=False,
                sources=[]
            )

        all_docs = (result_state.retrieved_docs or []) + (result_state.long_term_memory_hits or [])
        seen = set()
        sources = []
        for doc in all_docs:
            title = doc.metadata.get("title", "").strip()
            url = doc.metadata.get("url", "").strip()
            key = (title, url)
            if title and key not in seen:
                seen.add(key)
                sources.append(SourceInfo(title=title, url=url))

        return ChatResponse(
            session_id=session_id,
            response=result_state.generated_response,
            history=result_state.chat_history,
            long_term_memory_used=len(result_state.long_term_memory_hits or []) > 0,
            sources=sources
        )

    except Exception as e:
        logger.error(f"❌ LangGraph Exception: {e}")
        return ChatResponse(
            session_id=session_id,
            response="❌ Unexpected server error occurred.",
            history=[],
            long_term_memory_used=False,
            sources=[]
        )
