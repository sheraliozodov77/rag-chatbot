# rag/memory.py

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import InMemoryChatMessageHistory
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from rag.prompt_template import get_prompt_template
from rag.retriever import get_retriever  # For long-term memory

def get_short_term_memory(session_id: str):
    """Returns a buffer memory scoped to a session."""
    history = InMemoryChatMessageHistory()
    return ConversationBufferMemory(
        chat_memory=history,
        return_messages=True,
        memory_key="chat_history",
        input_key="user_input",
        output_key="generated_response"
    )

def get_long_term_memory_qa():
    """Returns a retriever-based QA chain for long-term memory recall."""
    retriever = get_retriever(k=5, filters={"source_type": {"$eq": "chat_memory"}})
    prompt = get_prompt_template()
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
