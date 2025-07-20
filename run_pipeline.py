# run_pipeline.py

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.ingestion import ingest_entry
from pipeline.processing import chunk_documents
from pipeline.embedding import embed_chunks
from pipeline.storage import init_index, upload_vectors
from utils.logger import setup_logger

logger = setup_logger()

SUCCESS_LOG = "output/success.txt"
FAILED_LOG = "output/failed.txt"
ENTRYPOINTS_FILE = "entrypoints.txt"
CHUNKS_LOG = "data/raw_chunks.jsonl"

Path("output").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

def load_entrypoints(retry_failed: bool = False) -> list:
    if retry_failed and os.path.exists(FAILED_LOG):
        with open(FAILED_LOG, "r") as f:
            return list(set(line.split("::")[0].strip() for line in f if line.strip()))
    else:
        with open(ENTRYPOINTS_FILE, "r") as f:
            return [line.strip() for line in f if line.strip()]

def get_success_set() -> set:
    if not os.path.exists(SUCCESS_LOG):
        return set()
    with open(SUCCESS_LOG, "r") as s:
        return set(line.strip() for line in s)

def log_chunks_for_finetuning(chunks: list):
    with open(CHUNKS_LOG, "a", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk, f, ensure_ascii=False)
            f.write("\n")

def process_entry(entry: str, index):
    try:
        logger.info(f"Processing: {entry}")
        result = ingest_entry(entry)
        if isinstance(result, Exception):
            raise result

        chunks = chunk_documents(result)
        if not chunks:
            raise ValueError("No valid chunks extracted.")

        log_chunks_for_finetuning(chunks)

        embedded = embed_chunks(chunks)
        if not embedded:
            raise ValueError("Embedding failed.")

        upload_vectors(embedded, index, source_id=entry)

        with open(SUCCESS_LOG, "a") as s:
            s.write(entry + "\n")

        logger.info(f"✅ Success: {entry}")

    except Exception as e:
        with open(FAILED_LOG, "a") as f:
            f.write(f"{entry} :: {str(e)}\n")
        logger.error(f"❌ Failed: {entry} — {str(e)}")

def main(retry_failed: bool = False, max_workers: int = 4):
    entrypoints = load_entrypoints(retry_failed)
    success_set = get_success_set()
    to_process = [e for e in entrypoints if e not in success_set]

    if not to_process:
        logger.info("✅ Nothing to process.")
        return

    index = init_index()

    logger.info(f"🚀 Starting ingestion for {len(to_process)} sources...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_entry, entry, index): entry for entry in to_process}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f" Unexpected error for {entry}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry-failed", action="store_true", help="Reprocess failed.txt only")
    args = parser.parse_args()

    main(retry_failed=args.retry_failed)
