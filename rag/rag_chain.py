# rag/rag_chain.py

from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from rag.retriever import get_retriever
from rag.prompt_template import get_prompt_template
from llm.llm_loader import load_llm
from langchain.llms import HuggingFacePipeline
from utils.logger import setup_logger

logger = setup_logger()

def build_rag_chain(filters: dict = None):
    logger.info("🔗 Building RAG chain...")

    retriever = get_retriever(filters=filters or {})
    prompt = get_prompt_template()
    llm = load_llm()

    if isinstance(llm, HuggingFacePipeline):
        logger.info("🤖 Using local LLaMA model via Hugging Face Transformers")
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    else:
        logger.info("🌐 Using OpenAI GPT model")
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    logger.info("✅ RAG chain ready")
    return rag_chain