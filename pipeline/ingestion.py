# pipeline/ingestion.py

import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Union

from dotenv import load_dotenv
from firecrawl import FirecrawlApp, ScrapeOptions
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import docx2txt

load_dotenv()
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)


def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return ' '.join(text.split())


def scrape_url(url: str) -> List[Dict]:
    print(f"🌐 Scraping: {url}")
    job = firecrawl.async_crawl_url(url, limit=20, scrape_options=ScrapeOptions(formats=["markdown", "html"]))
    while True:
        status = firecrawl.check_crawl_status(job.id)
        if status.status == "completed":
            break
        print(f"⏳ {status.completed}/{status.total} pages done...")
        time.sleep(5)

    results = []
    for doc in status.data:
        text = clean_html(doc.markdown or doc.html or "")
        if text:
            results.append({
                "source_type": "web",
                "url": doc.metadata.get("sourceURL", url),
                "title": doc.metadata.get("title", ""),
                "timestamp": datetime.now().isoformat(),
                "text": text
            })
    return results


def load_pdf(file_path: str) -> List[Dict]:
    print(f"📄 Loading PDF: {file_path}")
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return [{
        "source_type": "pdf",
        "url": f"file://{file_path}",
        "title": Path(file_path).stem,
        "timestamp": datetime.now().isoformat(),
        "text": ' '.join(text.split())
    }]


def load_docx(file_path: str) -> List[Dict]:
    print(f"📄 Loading DOCX: {file_path}")
    text = docx2txt.process(file_path)
    return [{
        "source_type": "docx",
        "url": f"file://{file_path}",
        "title": Path(file_path).stem,
        "timestamp": datetime.now().isoformat(),
        "text": ' '.join(text.split())
    }]


def ingest_entry(entry: str) -> Union[List[Dict], Exception]:
    entry = entry.strip()
    try:
        if entry.startswith("http"):
            return scrape_url(entry)
        elif entry.lower().endswith(".pdf"):
            return load_pdf(entry)
        elif entry.lower().endswith(".docx"):
            return load_docx(entry)
        else:
            raise ValueError(f"Unsupported file type or malformed URL: {entry}")
    except Exception as e:
        return e
