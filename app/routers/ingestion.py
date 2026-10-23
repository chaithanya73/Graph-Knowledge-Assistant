"""
Ingestion Router — API endpoints for uploading documents and text into the knowledge graph.

INTERVIEW: "Why separate routers per feature?"
- Single Responsibility: each router handles ONE domain (ingestion, query, graph).
- Team scalability: different developers can work on different routers without merge conflicts.
- Selective mounting: in microservices, each router could become its own service.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.query import DocumentIngestRequest, DocumentIngestResponse
from app.services.ingestion_service import ingestion_service

router = APIRouter(
    prefix="/ingest",
    tags=["Document Ingestion"],
)


@router.post(
    "/text",
    response_model=DocumentIngestResponse,
    summary="Ingest raw text into knowledge graph",
    description="Accepts raw text content, extracts entities and relationships using LLM, "
                "and stores them in the Neo4j knowledge graph.",
)
async def ingest_text(request: DocumentIngestRequest):
    """
    Ingest raw text → chunk → extract entities → store in Neo4j.

    INTERVIEW: "Why is this a POST and not a PUT?"
    - POST = create a new resource (the document and its entities).
    - PUT = replace an existing resource entirely (idempotent).
    - Each ingestion creates new graph data, so POST is semantically correct.
    """
    try:
        result = ingestion_service.ingest_text(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post(
    "/file",
    response_model=DocumentIngestResponse,
    summary="Upload and ingest a document file",
    description="Upload a PDF, DOCX, or TXT file. The system extracts text, "
                "identifies entities and relationships, and stores them in the knowledge graph.",
)
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload a file → extract text → chunk → extract entities → store in Neo4j.

    INTERVIEW: "How does FastAPI handle file uploads?"
    - FastAPI uses python-multipart under the hood.
    - UploadFile is an async wrapper around SpooledTemporaryFile.
    - For files < 1MB, it stays in memory. For larger files, it spills to disk.
    - We read the full content with await file.read() — for very large files (100MB+),
      you'd want streaming with chunks to avoid memory exhaustion.
    """
    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
    file_ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}",
        )

    try:
        file_bytes = await file.read()

        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        result = ingestion_service.ingest_file(
            file_bytes=file_bytes,
            filename=file.filename,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")
