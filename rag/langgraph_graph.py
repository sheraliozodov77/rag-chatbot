from langgraph.graph import StateGraph
from rag.state import ChatState
from rag.memory import get_short_term_memory, get_long_term_memory_qa
from rag.retriever import get_retriever
from rag.prompt_template import get_prompt_template
from langchain_openai import ChatOpenAI 
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_core.runnables import RunnableSequence
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


# Node 1: Optional long-term memory recall
def retrieve_long_term(state: ChatState) -> ChatState:
    long_term_qa = get_long_term_memory_qa()
    result = long_term_qa.invoke({"query": state.user_input})
    state.long_term_memory_hits = result.get("source_documents", [])
    return state


# Node 2: Retrieve from vector DB (your RAG chunks)
def retrieve_documents(state: ChatState) -> ChatState:
    retriever = get_retriever(k=5)
    state.retriever = retriever
    state.retrieved_docs = retriever.invoke(state.user_input)
    return state


# Node 3: Generate answer with OpenAI using RetrievalQA

def generate_answer(state: ChatState) -> ChatState:
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        prompt = get_prompt_template()
        generation_chain = LLMChain(llm=llm, prompt=prompt)

        docs: list[Document] = (state.long_term_memory_hits or []) + (state.retrieved_docs or [])

        if not docs:
            state.generated_response = (
                "I'm sorry, I could not find relevant information to answer that question based on the current knowledge."
            )
            return state

        # ✅ Build the context string
        context = "\n\n".join([doc.page_content for doc in docs])

        # ✅ Run the generation chain with your prompt template
        response = generation_chain.invoke({
            "context": context,
            "question": state.user_input
        })

        state.generated_response = response.get("text", "❌ No response generated.")
        return state

    except Exception as e:
        print("❌ Error in generate_answer:", str(e))
        state.generated_response = "❌ An error occurred while generating the response."
        return state



# Node 4: Update short-term memory
def update_memory(state: ChatState) -> ChatState:
    memory = get_short_term_memory(session_id=state.session_id or "default")
    memory.save_context(
        inputs={"user_input": state.user_input},
        outputs={"generated_response": state.generated_response}
    )
    return state


# ✅ Node 5: Return final state
def return_final_state(state: ChatState) -> ChatState:
    return state


# Build the LangGraph
def get_langgraph():
    builder = StateGraph(ChatState)

    builder.add_node("retrieve_long_term", retrieve_long_term)
    builder.add_node("retrieve_docs", retrieve_documents)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("update_memory", update_memory)
    builder.add_node("return_final", return_final_state)  # ✅ added

    builder.set_entry_point("retrieve_long_term")
    builder.add_edge("retrieve_long_term", "retrieve_docs")
    builder.add_edge("retrieve_docs", "generate_answer")
    builder.add_edge("generate_answer", "update_memory")
    builder.add_edge("update_memory", "return_final")  # ✅ now goes to final
    builder.set_finish_point("return_final")  # ✅ sets END

    graph = builder.compile()
    graph.output_type = ChatState
    return graph
