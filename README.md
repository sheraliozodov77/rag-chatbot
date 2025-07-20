# RAG Chatbot - Production-Grade LLM Application

This is a modular, production-grade Retrieval-Augmented Generation (RAG) chatbot project built using LangChain, Pinecone, LangGraph, vLLM, and other state-of-the-art tools. It supports document ingestion from multiple sources (websites, PDFs, DOCX), flexible embedding models, fine-tuning, and MLOps integrations.

## Features

- Modular ingestion pipeline for web and file content
- Chunking with RecursiveCharacterTextSplitter
- Embedding with OpenAI
- Vector indexing with Pinecone
- Batch embedding and upsert support
- Metadata tagging for traceability (source, title, timestamp, chunk index)
- Environment-driven model configuration
- Prepped for LangChain-based retrieval and LangGraph state management

## Project Structure

```
rag-chatbot/
├── ingest/
│   ├── firecrawl_scraper.py       # Website ingestion using Firecrawl
│   ├── file_loader.py             # PDF and DOCX ingestion
│   ├── cleaner.py                 # Cleans raw HTML/Markdown
├── embed/
│   ├── chunker.py                 # Splits long text into RAG-friendly chunks
│   ├── embedder.py                # Embeds text using OpenAI or local model
│   ├── indexer.py                 # Uploads vectors to Pinecone with metadata
├── scripts/
│   └── run_ingest_pipeline.py     # CLI for ingesting a URL or file
├── .env                           # API keys and config
└── README.md
```

## .env Example

```
# OPENAI
OPENAI_API_KEY=your-openai-key
EMBEDDING_MODEL=text-embedding-3-small

# PINECONE
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENV=your-region
PINECONE_INDEX_NAME=rag-chatbot-index

# FIRECRAWL
FIRECRAWL_API_KEY=your-firecrawl-key

# MISC
PROJECT_NAME=rag-chatbot
```

## Ingestion Usage

From the `scripts/` directory:

```bash
# Ingest website
python run_ingest_pipeline.py

# Ingest PDF or DOCX
python run_ingest_pipeline.py
```

Edit the source URLs or file paths directly in the script or parameterize them.

## Next Steps

- Implement LangChain retriever and generation logic (Week 4)
- Add LangGraph state-based workflows (Week 5)
- Connect to frontend/API for user interaction
- Add memory and monitoring
