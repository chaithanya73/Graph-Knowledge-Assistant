"""
Text Extraction Service.

Handles extracting raw text from different file formats (PDF, DOCX, TXT).
Follows the Strategy Pattern — each file type has its own extraction method.

INTERVIEW: "Why not just read file.read()?"
- PDF files are binary — you can't just .read() them. They contain encoded text
  streams, fonts, and layout data. PyPDF2 parses the internal PDF structure.
- DOCX is actually a ZIP file containing XML. python-docx parses that XML.
- Having a unified extract_text(file, type) interface hides this complexity
  from the ingestion pipeline (Facade Pattern).
"""

import io
from typing import List
from PyPDF2 import PdfReader
from docx import Document


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file's binary content."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file's binary content."""
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from a plain text file."""
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Unified text extraction — routes to the correct extractor based on file extension.

    INTERVIEW: "This is the Strategy Pattern — the caller doesn't need to know
    which extraction algorithm is used. Adding a new format (e.g., .md, .html)
    just means adding a new function and an elif branch."
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif lower.endswith(".txt") or lower.endswith(".md"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: {filename}. Supported: .pdf, .docx, .txt, .md")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for vector embedding and entity extraction.

    INTERVIEW: "Why overlapping chunks?"
    - If we split at exactly 1000 chars, a sentence like "FastAPI uses Pydantic"
      might get cut: chunk1 ends with "FastAPI uses" and chunk2 starts with "Pydantic".
    - Neither chunk has the full relationship. The LLM would miss it.
    - Overlap (e.g., 200 chars) ensures sentences at boundaries appear in BOTH chunks,
      so the entity extractor sees complete context.

    INTERVIEW: "Why not split by sentences instead?"
    - Sentence splitting is fragile (abbreviations like "Dr.", "U.S.", "e.g." break naive splitters).
    - Character-based chunking with overlap is simpler and more robust.
    - Production systems use libraries like LangChain's RecursiveCharacterTextSplitter
      which tries paragraph → sentence → word → char boundaries in order.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap  # Move forward by (chunk_size - overlap)

    return [c for c in chunks if c]  # Filter empty chunks
