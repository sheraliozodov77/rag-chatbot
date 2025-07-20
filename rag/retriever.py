# rag/retriever.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# ✅ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Validate index
index_names = pc.list_indexes().names()
if PINECONE_INDEX_NAME not in index_names:
    raise ValueError(f"❌ Pinecone index '{PINECONE_INDEX_NAME}' not found. Available: {index_names}")

print(f"✅ Using Pinecone index: {PINECONE_INDEX_NAME}")

def get_retriever(k: int = 5, filters: dict = None):
    embed_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # ✅ Use new PineconeVectorStore correctly
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embed_model,
        namespace="",  # optional: "" or your namespace
    )

    search_kwargs = {"k": k}
    if filters:
        search_kwargs["filter"] = filters

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    return retriever
