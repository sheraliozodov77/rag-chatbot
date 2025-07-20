# pipeline/processing.py

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


def is_valid_text(text: str) -> bool:
    return text and isinstance(text, str) and len(text.strip()) > 50


def chunk_documents(
    docs: List[Dict],
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> List[Dict]:
    """
    Accepts a list of raw documents with metadata and returns a list of chunked documents with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []
    for doc in docs:
        text = doc["text"]
        if not is_valid_text(text):
            continue

        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            chunked_doc = {
                "text": chunk,
                "chunk_id": f"{doc['url']}#chunk-{i}",
                "chunk_index": i,
                "source_type": doc["source_type"],
                "url": doc["url"],
                "title": doc.get("title", ""),
                "timestamp": doc.get("timestamp", "")
            }
            all_chunks.append(chunked_doc)

    return all_chunks
