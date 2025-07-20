# pipeline/storage.py

import os
import time
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
EMBEDDING_DIM = 3072  # match OpenAI text-embedding-3-large

pc = Pinecone(api_key=PINECONE_API_KEY)


def init_index():
    existing = [ix.name for ix in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"📦 Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(3)
    return pc.Index(PINECONE_INDEX_NAME)


def upload_vectors(vectors: List[dict], index, source_id: str):
    for vec in vectors:
        try:
            index.upsert(vectors=[vec])
        except Exception as e:
            with open("output/failed.txt", "a") as f:
                f.write(f"{source_id} :: upsert error :: {str(e)}\n")
    with open("output/success.txt", "a") as s:
        s.write(source_id + "\n")
