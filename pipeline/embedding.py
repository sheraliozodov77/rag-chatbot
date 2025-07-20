# pipeline/embedding.py

import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    embedded = []
    for chunk in tqdm(chunks, desc="🔢 Embedding chunks"):
        try:
            text = chunk["text"][:8191]  # OpenAI input limit
            res = openai_client.embeddings.create(
                input=text,
                model=EMBEDDING_MODEL
            )
            vec = res.data[0].embedding

            embedded.append({
                "id": chunk["chunk_id"],
                "values": vec,
                "metadata": {
                    "text": text[:500],
                    "title": chunk.get("title", ""),
                    "url": chunk.get("url", ""),
                    "source_type": chunk.get("source_type", ""),
                    "timestamp": chunk.get("timestamp", ""),
                    "chunk_index": chunk.get("chunk_index", 0)
                }
            })
        except Exception as e:
            print(f"❌ Embedding error for chunk {chunk['chunk_id'][:50]}...: {e}")
    return embedded
